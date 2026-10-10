# DeepSeek-V4 Flash DSpark execution

The installed checkpoint's upstream `inference/model.py` distinguishes base
generation from prediction. This document specifies the remaining executable
integration, not a claim that complete DSpark already runs.

## Base capture and main prefix

The 43-layer base decoder captures layers 40, 41 and 42 **after** each mHC
block. Each capture is the mean across four residual streams, accumulated in
F32 and rounded to BF16. Concatenation follows the official target order:
three 4,096-wide segments form one 12,288-wide input row. A learned base text
head is not equivalent to this unweighted capture.

`StreamMeanCapturePrecision` and its prepared frame already enter the parent
decoder queue before residual reuse. Placement retains local target metadata;
partial captures must be explicitly assembled before consumption. The separate
`RowProjectionNormPrecision` and `DeepSeekPredictionMain` owner execute the
actual `mtp.0.main_proj` E4M3/UE8M0 matrix and `main_norm` BF16 vector. Projection
rounds to BF16 before weighted F32 RMSNorm. Both stages share parent completion.

## Prediction attention is not ordinary causal attention

There are three `mtp.*` blocks. All have compression zero. The prediction block
has five draft slots: the seed token followed by four copies of noise token
128799. Embeddings expand to four residual streams. Model adapters own those
constants and checkpoint names; shared precision IR owns the arithmetic,
read sets, and explicit effects.

For initial prefill (`start_pos == 0`), each prediction block computes only its
main-stream KV projection, weighted normalization, rotary suffix and simulated
activation precision. It primes the 128-slot committed-main ring. It does not
execute the draft block's mHC/attention/FFN or sample draft tokens.

For prediction after prefill:

1. Compute main KV from the normalized captured main hidden state. The ordinary
   prediction call consumes one new committed main token.
2. Publish that main KV at `start_pos % window_size` before draft attention.
3. Compute draft queries/KV with absolute positions
   `start_pos + main_rows + draft_slot`.
4. Every draft query reads physical ring slots
   `0 .. min(window_size, start_pos + 1)` **and all five draft KV slots**.
   There is no triangular mask inside the draft block. A causal-window reader
   cannot substitute for this read set.
5. Include the learned attention sink in softmax normalization; inverse rotary
   and the grouped output projections remain subsequent operations.

Shared execution planning must express main publication and noncausal draft
consumption as separate effects. Prediction never publishes draft KV into the
committed-main ring. `CommittedDraftKvPrecision` now expresses this separate
read set; its AOT lowering and shared-ring frame build priming/prediction tables
once. `DeepSeekPredictionAttention` binds actual `mtp.*.attn` planes and joins
main projection/norm/rotary/simulation, draft query/KV transforms, publication
and noncausal attention. Its priming table has seven launches; prediction has
nineteen, including inverse rotary and both grouped output projections. The
result has five hidden-width BF16 rows and can write directly into a borrowed
enclosing mHC branch buffer without an extra copy. Output-A's BF16 parameter
boundary and Output-B's block-FP8 activation arithmetic use the existing shared
`GroupedAttentionOutputPrecision` and prepared frame. The KV matrix and learned
norm are uploaded once and borrowed by both
branches. Stage bounds use the three-block artifact count, not the config's
legacy value of one. Composite CUDA compilation and native ownership tests
pass. The standalone ring now passes independent GB10 correctness and sanitizer
checks; complete checkpoint prediction correctness remains unverified.
The base request's committed frontier remains authoritative;
prediction history must not independently advance or commit rejected drafts.
Prefill and draft counts/positions are separate prepared ports, not reinterpretations
of one mutable descriptor. Both phase plans are built at startup, without token-
step source generation, filesystem access, or argument-array construction.

## Prediction blocks and output dependencies

Ordinary prediction performs all three mHC attention and routed/shared FFN
blocks using the actual `mtp.0`, `mtp.1`, and `mtp.2` tensors. The prediction
integration uses an immutable model-owned block address, distinguishing a base
layer from a prediction stage. The address selects checkpoint/symbol namespaces
and the reference's absolute-layer routing law. Hyper-connection and packed-MoE
adapters consume this address while retaining the same shared precision plans
and prepared frames. Base entry points remain base-only; extending prediction
must not silently widen a base-layer API or substitute base checkpoint weights.
Priming executes only main-KV attention tables; prediction joins mHC prefix,
attention, mHC suffix and the routed/shared FFN envelope in its parent queue.
Compact banks and router weights are counted before upload, never expanded on
the host or rebuilt in the token path.

The prediction
head uses `mtp.2.hc_head_*` and `mtp.2.norm`, then the shared global vocabulary
matrix. Its output is five rows of F32 logits, not the base head's single last
row shortcut.

Markov sampling is sequential across those rows. Seed/previous sampled token
selects a BF16 256-wide Markov embedding, which projects to F32 vocabulary bias.
Add that bias to one row's logits, sample, and feed that token into the next
row. Computing all five biases from the initial seed is incorrect.

Confidence concatenates the head's pre-normalization BF16 hidden state with
the **corresponding previous-token** Markov embedding. The checkpoint's BF16
linear parameter is evaluated in F32 and returns a scalar without a sigmoid.
Confidence must not be computed from the normalized vocabulary-head input.

`DeepSeekPredictionHead` now binds these seven actual final-stage weight groups,
borrowing the shared global vocabulary matrix instead of uploading it again.
The learned frame exposes its three projection launches and pre-normalization
hidden view; its existing base four-launch suffix is unchanged. The shared
`SequentialMarkovPrecision`/prepared frame expresses the dependent sampling and
raw confidence laws. A complete head prepares three projection launches plus
four launches for each of five dependent rows (23 total), with one parent
completion. Seed/sequence RNG ports are explicit; positive-temperature
categorical sampling is not claimed random-seed-identical to PyTorch.
Native checkpoint/ownership tests and complete GB10 head compilation pass.
The independent chain numerical/sanitizer campaign is queued behind the current
base-model and ring campaigns; no physical chain or complete predictor pass is
claimed yet.

Draft verification and accept/reject/commit remain explicit scheduling and KV
ownership operations. The verified main model determines which tokens become
committed. A completed prediction alone cannot claim speculative generation or
its speed benefit.

## Verification state rollback

The base stage now declares its actual persistent allocations at preparation:
window payload/frontier, retained compressed values/frontier/counts, incomplete
learned-pooling numerators/gates/history, and the learned indexer's corresponding
pool/cache. Weights, query/output scratch and recomputed append descriptors are
excluded. A logical length reset alone is not rollback: tentative execution can
overwrite a ring slot or complete a previously partial compression group.

`compiler/state_snapshot_plan` purely places these declared sizes into a bounded
backup arena. `integration/device_state_snapshot` prepares a reusable device-only
copy transaction on the model stream. The CUDA backend resolves and leases the
regions once; forward and reverse copies share one reusable completion event.
Submission allocates no heap, computes no hash and stages no payload on the host.
Failed/partial submissions must drain before resource release. Backup capacity
is additional to the model's resident budget and is checked before allocation.

This is the physical state transaction, not complete speculative generation.
The predictor now declares each of its three committed-main ring/frontier
pairs, excluding draft output and recomputed descriptors. Its attached execution
owner also includes the explicit 16-byte RNG and prepares the same device-only
snapshot transaction. At the official 128x512 BF16 ring geometry, backup is
393,256 bytes for all three rings/frontiers plus RNG. This is additional budget,
not silently charged to an unspecified reserve. The snapshot must close before
the predictor whose allocations it leases. Contiguous Prime updates can append
later prompt chunks or replay an accepted prefix at the committed frontier;
Predict remains a one-main-row update and never publishes draft KV.

The verified-prefix algorithm, host request-frontier transaction, all-position
base head and inter-rank draft/result exchange must still be connected to these
state transactions. A restored base arena must not be described as a completed
DSpark acceptance loop. Model work must retire before rollback/commit; callers
must order all state mutation on the prepared stream or an explicit dependency.

The commit-pinned GB10 generic snapshot probe passed 128 rollback cycles and
commit/poll/budget/ownership checks (412 backup bytes). This proves the device
copy transaction, not a complete checkpoint DSpark acceptance result.

The working-tree ring campaign on .179 at
`/tmp/lunaflux-committed-draft-chunk-undo-20261010-v1` passed 262 independent
cases with maximum absolute error zero. It covers contiguous prompt chunks,
ring wrap, device-only undo of overwritten payload and frontier, replay of only
the accepted prefix, and rejection of a position gap without state mutation.
Memcheck reported zero errors and zero leaked bytes; racecheck reported zero
hazards; synccheck reported zero errors. This is component evidence, not a
whole-model acceptance or speculative-speed result.

## Completion requirements

- Execute the complete checkpoint base runner on both bounded Spark ranks and
  compare numerical/token output independently; a BOS smoke is not a tokenizer
  or chat-template correctness result.
- Consume complete ordered captures in the actual prediction queue, not only
  in an isolated prefix fixture.
- Test ring priming beyond 128 tokens, wraparound, exact committed positions,
  noncausal future-draft visibility, reset and cancellation without draft writes
  to committed storage.
- Execute all three prediction blocks with real compact checkpoint weights;
  validate Markov sequence dependencies, confidence arithmetic, and base-model
  verification/commit.
- Count model, prediction state, workspace, boundary storage and driver/process
  reserve before allocation. The bounded whole-model run remains 96 GiB per
  host with no swap allowed for its unit; component fixtures are capped at
  2 GiB. Do not materialize a full expanded expert bank on the host.

Native allocation/ownership tests and independent GPU component correctness,
memcheck, racecheck and synccheck are required for changed execution boundaries.
They do not replace real whole-model correctness or end-to-end benchmarking.
