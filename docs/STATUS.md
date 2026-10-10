# LunaFlux implementation status

This file records implemented behavior, not intended architecture. A phase is
not complete until every gate in [PLAN.md](PLAN.md) passes.

Formal phase ledger: Phases 0 through 4 have crossed their declared gates.
Phases 5 through 9 remain open. References below to a complete graph, a Ready
rank, startup admission, or a locally complete software slice are scoped
claims and do not close those five phase gates.

## Executable compact expert features — 2026-10-09

Update 2026-10-10 (DSpark local predictor composition): `DeepSeekPredictor` joins
the ordered capture projection, seed/four-noise embedding, all three checkpoint
prediction blocks and the dependent learned output head. Its separate priming
table skips draft computation. The seeded-block operation belongs to shared
precision IR with CUDA-specific implementation only in lowering. Aggregate
private weight/workspace budgeting precedes upload; captures, global tables and
result/RNG ports remain explicitly borrowed. The affected suite passes 59/59;
the complete predictor CUDA source compiles on GB10 under a 2-GiB/no-swap CPU
build unit. Artifacts are at `/tmp/lunaflux-markov-results-20261010-v2`.
The expanded 352-case seed/Markov oracle and sanitizers are queued, not passed.
Whole-checkpoint preparation/replay, base serving attachment, verification/commit
and independent whole-model correctness remain unfinished.

Update 2026-10-10 (DSpark dependent output head): generic immutable
`SequentialMarkovPrecision`, CUDA lowering and prepared frame now preserve
previous-sampled-token embedding dependencies, F32 bias addition, stable greedy
or explicit-seed/sequence categorical sampling, and raw confidence over
pre-normalization hidden plus the corresponding previous-token embedding.
`DeepSeekPredictionHead` binds seven actual `mtp.2` checkpoint groups and borrows
the base vocabulary allocation. Three learned collapse/norm/head launches join
twenty sequential row launches in a single parent queue. Existing base-head
suffixes remain unchanged. The 53-test affected precision/source/prediction suite
and the checkpoint-backed head fixture pass; warmed native fixtures allocate
nothing and retain explicit ownership. Complete head and greedy/stochastic oracle
sources compile on GB10 in 2-GiB/no-swap units, downloaded under
`/tmp/lunaflux-markov-results-20261010-v1`. Numerical/sanitizer execution remains
queued, not passed. Complete predictor orchestration, embedding, actual checkpoint
block execution and verification/commit remain open.

Base-run launcher correction: the live egress had loaded its checkpoint and
owned both listening ports, while the launcher was waiting for a journal message
that had not appeared. Readiness now checks both listeners against the current
user-unit PID. A bounded resume runner verified that ownership and the unchanged
commit-pinned executable, then started ingress without restarting egress or
recopying weights. Generation is still pending; this is not a numerical pass.

Update 2026-10-10 (DSpark prediction block composition): an immutable model-owned
base-layer/prediction-stage address selects exact checkpoint and symbol prefixes
and the absolute-layer routing law. Existing base APIs remain base-only. Each
`DeepSeekPredictionBlock` now joins mHC attention, the complete nineteen-launch
prediction attention output, mHC publication and routed/shared packed MoE in one
borrowing parent queue. Its separate prime table executes only main-KV priming.
Every compact bank, router, mHC and attention weight/workspace is counted before
upload; output residuals remain borrowed. Thirty affected plan/owner tests,
three upgraded checkpoint fixtures and the broader 98-test base-decoder/packed
execution aggregate pass with the documented migration warning exclusions.
All three composed block sources compile on GB10 under a 2-GiB/no-swap build
unit, retained at `/tmp/lunaflux-committed-draft-results-20261010-v5`.
This does not establish complete checkpoint prediction: three-block orchestration,
learned output/Markov/confidence, distributed capture and verification/commit
remain open, as does physical numerical validation of the composed blocks.

Update 2026-10-10 (DSpark complete attention output): prediction attention now
executes inverse rotary plus grouped Output-A and Output-B using the existing
shared `GroupedAttentionOutputPrecision`/prepared frame. It binds the actual
`mtp.*.attn.wo_a/wo_b` compact payload/scale planes alongside the six shared
query/KV/norm/sink groups. The phase tables remain seven priming launches and
now nineteen prediction launches. Output is five hidden-width BF16 rows;
enclosing mHC output may be borrowed with exact workspace subtraction and no
ownership transfer. Both targeted owner/plan tests and all five existing output
frame/adapter tests pass, exercising symbols, eight-group binding,
budget-before-upload, own/borrow output, warmed submission and abort/release.
The extended composite compiles on GB10 under a 2-GiB/no-swap build unit;
artifacts are at `/tmp/lunaflux-committed-draft-results-20261010-v4`.
Physical numerical validation remains pending behind the active whole-model
DeepSeek run. See the later block-composition update above; full prediction,
Markov/confidence and verification remain open.

Update 2026-10-10 (DSpark noncausal attention): shared
`CommittedDraftKvPrecision`, AOT source and prepared shared-ring frame now
separate initial main-KV priming from main-publication/all-draft attention.
`DeepSeekPredictionAttention` streams the exact six `mtp.*.attn` weight groups,
sharing KV matrix/norm between main and draft paths. Priming has seven ordered
launches; prediction has sixteen, including rotary and simulated KV precision.
Stage selection uses the three-block artifact count, not the legacy config
field of one. The expanded affected native suite passes 140/140, including the
new checkpoint-owner regression for exact summed budget, missing-stage rejection,
warmed allocation-free phases and abort/release.
Both ring kernels and the complete checkpoint attention source compile on GB10
under 2-GiB/no-swap build units. Build artifacts are retained at
`/tmp/lunaflux-committed-draft-results-20261010-v3`; no GPU numerical pass is
claimed yet because the full DeepSeek run has priority. Output projection,
prediction mHC/MoE, Markov/confidence and verified generation remain unfinished.

Update 2026-10-10 (DSpark main projection): generic `RowProjectionNormPrecision`
preserves packed block-128 E4M3/UE8M0 activation/weight arithmetic, an intermediate
BF16 boundary, and F32 weighted RMSNorm with final BF16 rounding. The prepared
frame reuses existing projection and RMSNorm renderers and parent completion;
native tests cover pre-allocation span/budget rejection, partial abort, 32 warmed
submissions without allocations or extra blocking synchronization, and release.
`DeepSeekPredictionMain` binds actual `mtp.0.main_proj` and `main_norm` planes,
accounts for compact weight/workspace bytes, and rejects incomplete/reordered
target-layer captures. The affected native precision/source/prefix/packed-execution
suite passes 137/137 with the documented toolchain-migration warning exclusions.
The independent GB10 fixture passes 36 numerical/replay
cases with measured zero maximum absolute error for both stages, memcheck with
zero errors/leaks, racecheck with zero hazards, and synccheck with zero errors.
The 2-GiB/no-swap artifacts are downloaded under
`/tmp/lunaflux-row-norm-results-20261010-v1`, executable SHA-256
`dd99984347df3e7669d82245bde677cc498db396c8c78f44831638237455f89b`.
This is a prepared prefix/component result, not a complete DSpark run or speed
claim. Prediction KV/read-set semantics, prediction blocks, Markov/confidence,
and verification/commit still require integration and whole-model validation.

Update 2026-10-10 (DSpark target capture): the shared precision IR now describes
BF16 residual-stream mean with F32 accumulation and one BF16 output round into
ordered hidden segments. A prepared generic frame joins captures to the parent
queue without a new completion or hot-path allocation. DeepSeek slices/stages
can explicitly enable official DSpark target-layer capture, append it before
residual reuse, and expose local ordered segment metadata for distributed
assembly. Default base execution remains unchanged. Native lifecycle tests
cover exact budgets, alias/bounds rejection, 32 warmed submissions with zero
allocations, partial abort and balanced release. The generated source passes
192 independent GB10 numerical/replay/inactive-row cases, memcheck, racecheck
and synccheck with zero errors/hazards and zero memcheck leaks. The bounded
2-GiB/no-swap run and downloaded executable/hash are retained under
`/tmp/lunaflux-stream-capture-results-20261010-v1`.
This connects capture, not complete DSpark prediction/Markov/confidence or
reference-correct whole-model generation.

Checkpoint run progress: exact `19c8f35a` builds as ARM64 and accepts the actual
43-layer DeepSeek config on both Sparks. Both executables have SHA-256
`04e6c3a07c8ad5ad5d42bac667a6522d777c756b046d9aebe216052917fd9dd4`.
The broader affected native stage/runner/packed-execution tests pass 92/92.
Both checkpoint copies, payload export, placement and rank AOT compilation have
passed. Placement assigns layers 0–26 to `.178` and 27–42 to `.179`, with planned
footprints 102,036,247,436 and 62,659,803,764 bytes within each 96-GiB ceiling,
including a 4-GiB reserve. The egress startup loader is active; generation has
not passed yet. The dedicated `.mbtx` runner waits for actual egress readiness
before starting ingress, without concurrent whole-model GPU campaigns.

MiniMax footprint correction: the management repository's approximately
465 GiB total includes duplicate root layouts and both FL2VA/Ref2VA variants.
A single variant is approximately 135 GiB: 63 GiB text encoder, 62 GiB
transformer, 9.8 GiB video VAE and 578 MiB audio VAE. Two-host component/lifetime
placement is therefore plausible without assuming a general weight-offload
requirement. Exact model/workspace placement and whole-model media generation
remain unfinished; aggregate disk sizes alone do not prove device feasibility.

Update 2026-10-10 (DeepSeek checkpoint stages): contiguous base-layer slices
now stream complete compact routed/shared banks and router weights and use at
most two reusable inter-layer residuals. `DeepSeekDecoderStage` adds embedding
only on ingress and the corrected learned head only on egress, with a single
ordered queue and two distinct boundary residuals. Its pre-upload footprint
covers owned weights, state, workspace and residuals; caller-owned request
ports remain explicitly separate. Token tables are now stage-owned
and included in that footprint: exact I64 checkpoint planes stream through bounded
chunks into checked I32 device allocations, without a whole-model host arena.
Payload scratch and per-table uniqueness rows are preflighted separately from
reader metadata/process memory. Split-cell chunks, exact device bytes, budget
rejection before allocation, malformed rows and cleanup pass native device-double
tests. The commit-pinned real-driver probe at `90f85532` now passes ten GPU
cases, exact I32 readback, rejected partial uploads and deterministic release.
Memcheck reports zero errors and zero leaked allocations. Downloaded artifacts
match remote SHA-256 under `/tmp/lunaflux-index-gpu-results-20261010-v1`.
Missing complete
checkpoint tensors are rejected during footprint preflight in component fixtures.
Only a prepared complete range may bind whole-model worker delivery.
The affected text/precision/stage suite passes 56/56; position/serial regressions
pass 19/19, including both rank port layouts. The broader affected native
aggregate also passes 145/145. Combined ingress/egress stage
source compiles under CUDA 13.0.88/sm121 in a 2-GiB/no-swap user unit. Sources,
module and journals were downloaded with matching SHA-256 to
`/tmp/lunaflux-deepseek-stages-results-20261010-v1`.
This does not establish real checkpoint stage execution. Real two-host checkpoint
generation and DSpark prediction execution remain integration work.
GLM's independent numerical comparison and MiniMax whole-model integration
also remain open; no model-wide speed or accuracy claim is made.

Update 2026-10-10 (compressed-row coordinates): shared immutable precision IR
now distinguishes cache-relative IDs from offset-relative IDs. DeepSeek's
joint attention reads separate window/current/compressed buffers, so its learned
selection and reader use cache-relative IDs. The former add-offset/subtract-offset
pair was redundant; no dynamic host offset tensor is required for that route.
The default offset-relative contract remains for existing joined-view callers.
The AOT renderer removes offset loads and the model adapter no longer accepts
an offset allocation. Both modes pass sixty GPU selection cases and 128 joint
attention steps, including null offset pointers for cache-relative kernels,
causal selection, ring wrap, replay and error behavior. Memcheck, racecheck and
synccheck all report zero errors/hazards; memcheck reports zero leaks. Complete
learned-indexer and compressed-sublayer sources compile on GB10. Both hosts use
2-GiB/no-swap user-unit limits; downloaded sources/binaries/cubins have matching
remote hashes under `/tmp/lunaflux-cache-coordinate-results-20261010-v1`.
These component fixtures do not establish complete-checkpoint generation.

Update 2026-10-10 (DeepSeek two-rank runner): stage-bound activation leases now
prevent resubmission/storage release during the common plaintext TCP/pinned-DMA
transfer. Rank and sender completion observers retire a submission only once.
Rank binding checks prepared request storage and token-output ownership.
Shared pure placement includes actual compact layer costs, checked I32 hash
tables, boundary text weights, four residual frames and caller reserve.
`cmd/deepseek_checkpoint` joins prepared stages, absolute-position ports, shared
rank control, bounded pretokenized prompts and greedy generation. The native
release executable builds and accepts the actual installed DSpark checkpoint
configuration (43 base layers, hidden 4096, vocabulary 129280). Lifecycle tests
cover leased closure/resubmission, invalid row counts, epoch overflow and
duplicate completion observation. The diagnostic executes base layers only;
DSpark prediction/MTP remains unimplemented in it. Actual complete-checkpoint
two-host execution and reference comparison remain unverified.
The affected runner/stage/shared control and pipeline native release suite passes
13/13 with the existing dependency warning-migration exclusions; the runner
builds locally. Checkpoint copies from `.175` to new directories on `.178` and
`.179` are running under separate 2-GiB/no-swap user units. They load no GPU
weights; first real stage execution remains pending.

Update 2026-10-10 (absolute request positions): the model-neutral serial frame
has an optional startup-selected position port. Chunked prefill publishes
`committed_start + row`; decode publishes the exact next committed position.
Only live row bytes are copied. Worker and rank sample/error slots derive from
the immutable input count, so the added port cannot be mistaken for output.
Legacy GLM frames retain eight ports and their original capacities. Repeated
position-enabled worker execution tests zero warmed allocations, no additional
blocking synchronization, shifted error delivery and deterministic release;
rank regressions exercise both port layouts and invalid samples/errors.
This connects request metadata, not whole-model DeepSeek numerical execution.

Update 2026-10-10 (DeepSeek text arithmetic): the output boundary now selects
explicit precision-IR laws for F32 normalization/affine multiplication followed
by one BF16 activation cast, and F32 accumulated vocabulary logits. GLM's default
rounding laws are unchanged. The independent GB10 probe passes live row counts
1/3/5, deterministic repeats, exact greedy output, inactive-row preservation,
invalid controls and balanced release with maximum absolute logit error zero.
Its fixture distinguishes the old paths at 16 normalization and 78 logit
rounding sites. CUDA correctness plus memcheck/racecheck/synccheck all exit zero;
sanitizers report zero errors/hazards. Runs use 2-GiB/no-swap user-unit limits.
Sources, executable and journals were downloaded with matching SHA-256 to
`/tmp/lunaflux-learned-text-laws-results-20261010-v2`. These are component
numerical results, not full-checkpoint DeepSeek/DSpark generation or throughput.

Update 2026-10-10 (DeepSeek complete base block): `DeepSeekAttentionEnvelope`
and `DeepSeekFfnEnvelope` join corrected F32 mHC controls, complete attention,
and routed/shared experts in `DeepSeekDecoderBlock`. Window, learned-ratio-4,
and all-causal-ratio-128 profiles execute 38, 67, and 47 borrowed effects in the
tested base-layer schedules. The output projection writes into borrowed branch
storage, removing an extra output buffer without adding a copy. Expert/router
banks remain stage-owned and are explicitly excluded from private frame budgets.
All three native device-double profiles pass 32 replays with zero warm heap
allocation, no additional blocking synchronization and balanced release; the
expanded affected release suite passes 93/93. Combined attention and complete
base-block modules compile on GB10 under CUDA 13.0.88/sm121 with 2-GiB/no-swap
limits. Both modules, sources and user-unit journals were downloaded with
matching SHA-256 to `/tmp/lunaflux-deepseek-block-builds-20261010-v1`.
This proves component composition and AOT buildability, not complete-checkpoint
GPU equivalence, whole-model generation or DSpark prediction execution.

Update 2026-10-10 (DeepSeek F32 mHC): checkpoint-backed attention and FFN
envelopes now consume the actual F32 control-function tensors, project the
unscaled residual in F32, then apply inverse RMS. A separate immutable numerical
law reduces combination[source,destination] residual products before adding the
branch product and publishing BF16. The previous BF16/pre-normalized and
destination-major contracts remain available for their existing callers; they
are not silently substituted for DeepSeek arithmetic. The new model adapter
streams the actual function/base/scale/norm tensors and contributes three prefix
effects and one suffix to the parent decoder queue.

The affected native release regression passes 66/66. Fake checkpoint tests
cover both envelopes, half-sized F32 banks, aggregate budget rejection before
allocation, zero warm heap allocation and deterministic release. A GB10 probe
passes ten live-row/reset/replay cases against an independent F32 CPU oracle;
428 non-symmetric-matrix witnesses distinguish the corrected orientation from
the old one. Memcheck, racecheck and synccheck exit zero with no errors, leaks or
hazards under a 2 GiB/no-swap ceiling. The executable SHA-256 is
`10091eb3e510a85d04d37a5cf0280b8709f1811eca7f11f2279817685c59f0d0`.
These are component numerical results, not complete DeepSeek/DSpark generation.
Complete decoder composition and the two-rank whole-model runner remain open.

Update 2026-10-10 (joint window/compressed attention): immutable
`WindowCompressedKvPrecision` distinguishes learned row selection from all-causal
compressed rows. Its common prepared frame keeps a single sink-softmax over
sliding-window and compressed operands, then copies/publishes the window only
after all queries have read it. The DeepSeek adapter now composes 41 effects for
ratio-four attention and 22 for ratio-128 attention; inverse rotary and learned
Output-A/B extend these to complete 44/25-effect attention sublayers. State,
scratch and compact checkpoint banks are summed before upload. All cache/indexer
append descriptors join the parent completion; no per-component completion,
warm allocation, filesystem operation or cryptography is added.

The affected native release regression passes 72/72. The official-shaped fake
driver test prepares both compressed variants and verifies one-byte aggregate
budget rejection, 32 steps, zero warm allocations, no added blocking sync and
balanced release. Actual GB10 joint-reader fixtures pass 64 steps covering causal
unions, learned offsets, ring wrap, reset, replay, empty/inactive rows, upstream
error and zero compressed capacity. Memcheck reports zero errors/leaks; racecheck
reports zero hazards; synccheck reports zero errors. Learned BF16 score/causal
top-k selection separately passes 30 GPU cases and those three sanitizer gates.
These are numerical component fixtures, not full-checkpoint decoder numerics.

Both complete Flash compressed-sublayer sources compile under CUDA 13.0.88 /
`sm121`, with cubin SHA-256
`a17a7397771d4f9980fa1d795d33400625806d56ec4d62a57ea18c3fb9cff8e4`
and `6d48fd986572135d424b56a84d4ad6cf36c5b2e1ef0a2715067b62648edfdfc7`.
Downloaded binaries/cubins match the remote hashes under
`/tmp/lunaflux-compressed-attention-gpu-evidence-20261010-v2`.
The initial export lacked a source separator and failed NVCC before GPU work;
that attempt is preserved, the separator is fixed and regression-tested.
An async fake-device fixture now takes its own exclusive test lane, fixing
cross-test enqueue-count interference without serializing production execution.
Complete DeepSeek mHC/MoE decoder composition, distributed ownership and
DeepSeek/DSpark generation remain unfinished. No speed improvement is claimed.

Update 2026-10-10 (real GLM template generation terminal): the corrected v2 run
of committed source `e7facf66` finished on both Sparks at 07:54:11 CST, with exit
status zero, two token IDs `785,1196`, and `finish=Length`. Both journals record
96.0 GiB peak / zero swap; both GPU processes released. The 13-token controlled
chat-template input was processed in order. Downloaded completion frames and
journals are in `/tmp/lunaflux-glm-template-generation-20261010-v2`. Transient
user units were collected after exit, so their later empty timestamps are not
the terminal evidence; the retained journals contain the actual run. This proves
real two-stage execution and orderly shutdown, not independent output accuracy,
general text tokenization or useful serving throughput. Each token step took
roughly 18–23 seconds in the correctness-first implementation. Independent
reference comparison remains required.

Update 2026-10-10 (checkpoint-backed learned indexer):
`CompressedIndexSelectionPrecision` supplies immutable compressed-row capacity,
causal visibility, top-k and aggregate scratch sizing. Its shared CUDA lowering
preserves separate BF16 dot/product/head-sum publications, consumes already
scaled learned head weights, and returns compressed row IDs rather than
expanding them into raw token positions. Selection consumes private reusable
score scratch to avoid scanning all previous winners for every candidate;
replay rewrites every score/output cell, including inactive rows. Equal scores
use a deterministic lower-row-ID tie break, not an assertion about PyTorch's
unspecified top-k tie ordering.

`DeepSeekLearnedIndexer` now composes six learned query effects, eleven learned
compressed-operand/cache effects and two score/selection effects. All nineteen
belong to the containing decoder queue. The adapter binds the actual Query-B,
head-weight and compressor checkpoint tensors and checks their summed compact
banks/state/workspaces before upload. The 67-test affected regression passes;
the new complete-owner fake-driver test measures 32 submissions / 608 effects,
zero warm heap allocations, no added blocking synchronization, exact one-byte
aggregate budget rejection before allocation and balanced deterministic release.
Native warning-denied checking passes with existing legacy warning exclusions;
public interfaces are refreshed. The standalone selection probe compiles for
GB10 `sm121` / CUDA 13.0.88 without warnings. Full checkpoint-geometry source
also compiles into a combined AOT module: source SHA-256
`544a2f7c461649d5367d5165efb44c641c14330fe02d6c6cc322827cfcc45789`,
cubin `fa8ebf37cbce1e1995a459024b7cb7a63578215e67e5474b1de230efef4cb267`.
Compilation uses a separate 2 GiB/no-swap CPU-only unit; it does not submit
GPU work or reload the live GLM run. Selection GPU numerical/sanitizer
execution has since passed as scoped above. This closes full-head
single-rank learned indexing's source integration, not distributed head reduction,
the complete DeepSeek/DSpark decoder or whole-model correctness.

Update 2026-10-10 (learned index queries): `LearnedIndexQueriesPrecision`
composes packed block-128 E4M3/UE8M0 query projection from the already-normalized
low rank, BF16 head projection/scaling, and the existing rotary/Hadamard/E2M1
transform. The global full-head normalization belongs to the immutable numeric
plan; hardware geometry remains in shared CUDA renderers. The DeepSeek adapter
binds actual `attn.indexer.wq_b.weight/scale` and `weights_proj.weight` rather
than substituting ordinary attention heads or fixture weights. Two compact
checkpoint banks and all five intermediate buffers count against the aggregate
budget. Six prepared effects share the caller's decoder queue.

The initial five focused tests pass; the expanded affected regression passes
59/59, including compact checkpoint binding/streaming. Thirty-two six-effect
submissions measure zero warm heap allocations, no additional blocking
synchronization and balanced release; short borrowed positions fail before
private allocation. Scoped warning-denied native checking passes with existing
legacy exclusions; interfaces are refreshed. New query-chain GPU correctness
and sanitizers remain pending while GLM owns both devices. The subsequent
learned indexer update above supplies weighted scoring/causal top-k; compressed
sparse attention and the complete DeepSeek/DSpark decoder remain unfinished.
This is not full-model correctness or a performance result.

Update 2026-10-10 (GLM template rerun): committed source `e7facf66` builds on
ARM in `/tmp/lunaflux-glm-current-arm-20261010.nrWP1kZm`, 209 tasks, approximately
503 MiB peak and zero swap. Both ranks use binary SHA-256
`2920bf418e925b3de8506980f054da2035c0a63fd59a03f61c9063b5fdefd2d6`.
The checkpoint's actual chat template and vocabulary produce the controlled
13-token prompt for user content `a`, reasoning effort low and the assistant
thinking prefix; this offline ASCII fixture is not arbitrary-text tokenizer
support. The first launch passed the wrong relative inventory path and failed
before payload loading on both ranks; those v1 journals are retained.
Corrected user units `lunaflux-glm-template-ingress-20261010-e7facf66-v2.service`
and `lunaflux-glm-template-egress-20261010-e7facf66-v2.service` are active,
authenticating the exact 121-shard checkpoint under 96 GiB/no-swap ceilings.
No additional GPU workload runs concurrently. Template output, independent
numerical comparison and a clean two-rank EOF/release result are not yet known.

Update 2026-10-10 (learned indexer cache operands): the same
`DeepSeekCompressedKvCache` owner now supports the indexer's distinct checkpoint
pool, width and numerical transform. `RotaryHadamardFp4Precision` describes
suffix rotary, normalized Sylvester Hadamard and blockwise E2M1 simulation,
including BF16 publication and both exact scratch tensors. CUDA lowering reads
each complete prior Hadamard stage before swapping shared ownership and uses
power-of-two `amax / 6` scales with even-code midpoint ties. Three prepared
effects feed the existing exact-word retained cache. Attention and indexer
paths share cache ownership/budget code without substituting one precision
contract for the other. The ordinary compact rotary renderer is also shared.

The new focused suite passes 5/5; the expanded affected regression passes 60/60.
Thirty-two eleven-effect prepared pool/transform/cache submissions measure zero
warm heap allocations, no additional blocking synchronization and balanced
release. The independent GPU numerical probe was repaired and retested: its
initial long-position case differed at an FP4 midpoint. The CPU reference had
rewritten `(1 / frequency_denominator) * position` as one division, changing F32
rounding before BF16/FP4 publication. The corrected oracle preserves the stated
operation order; lowering also uses the immutable normalized-Hadamard scalar,
not an approximate `rsqrt` substitution. Final `sm121` correctness passes with
largest error zero for the small 128-width/two-head fixture, including positions
0/3/1023, live 0/1/2/3, replay and exact FP4 midpoint ties. Initial failed runs
are preserved. Sanitizer completion is recorded below; this is component
correctness, not full-model correctness or speed.
This closes the indexer compressed-operand source path,
not learned index queries, head weighting/top-k, sparse attention, the complete
DeepSeek decoder or DSpark generation. Output remains simulated BF16, not
packed FP4 cache storage; no whole-model numerical or performance claim follows.

The final Hadamard/FP4 fixture also passes memcheck/full leak, racecheck and
synccheck with zero errors/hazards/leaks, under separate 2 GiB/no-swap user units.
CUDA is 13.0.88 on GB10 `sm121`; the selected binary is from the v3 directory
`/tmp/lunaflux-hadamard-fp4-v3-20261010.QELNUHrg`. Earlier v1/v2 correctness
failures retain their original source, probes and journals. Downloaded exact
sources, binaries and run records are under
`/tmp/lunaflux-precision-transform-evidence-20261010-v1`.

Update 2026-10-10 (persistent compressed attention cache):
`DeepSeekCompressedKvCache` now connects the learned pool and compact rotary/
precision transform to request-owned retained rows. The generic immutable
`RetainedRowsPrecision` plan and three-effect CUDA lowering publish length only
after exact-bit copies. Requested context bounds constrain both token pooling
and compressed storage; contexts shorter than a compression group have zero
logical rows, backed by one inaccessible sentinel row. Aggregate state/workspace
budgets are split across owners, not granted independently to every component.
Ten effects share the containing executor; no queue or blocking token-step
copy is introduced. The subsequent indexer transform is described above;
learned index queries, sparse attention and full decoder integration remain open.

The five new focused native tests pass; the expanded affected-package suite
passes 52/52. Thirty-two ten-effect fake submissions
measure zero warm heap allocations and no new blocking synchronization, with
balanced deterministic release. Short borrowed ports fail before private
allocation. GPU numerical and sanitizer tests for the entire checkpoint-backed
learned retained chain remain pending. An offline
`.mbtx` exporter and independent CUDA exact-word/publication oracle are ready
for empty chunks, reset, gaps, overflow, upstream errors and zero capacity;
the exported fixture is not a physical pass claim.

After GLM released both GPUs, the retained-row fixture passed actual `sm121`
exact-word/publication checking (capacity three: 19 steps; zero capacity: four)
and memcheck/full leak, racecheck and synccheck, all with zero errors/hazards/
leaks under 2 GiB/no-swap user units. This verifies the generic append primitive,
not the entire checkpoint-backed learned compressor chain.

Update 2026-10-10 (compressed attention operand transformation):
`DeepSeekAttentionCompressor` now joins checkpoint-backed pooling to standalone
compact-row RoPE and prefix-only block-64 E4M3/power-of-two simulation. The
immutable `RotaryCachePrecision` plan counts one output tensor, not a dummy
query plus KV pair. The prepared frame consumes the pool's compact counts and
group-start positions; seven effects share the containing queue and transformed
values retain BF16 storage. Query/KV and compact-row sources share the same
frequency and simulation lowering. Indexer Hadamard/FP4, persistent compressed
cache publication and full decoder integration remain separate unfinished work.

The five new targeted native tests pass across semantic plans, generated source,
all Flash/Pro layer ratios and prepared ownership. Thirty-two seven-effect
submissions measure zero warm heap allocations, no added blocking synchronization
and balanced deterministic release. Scoped native checking and formatting pass
with the existing legacy warning exclusions; generated interfaces are refreshed.
The larger affected-package regression subsequently passed 110/110. New
whole-chain GPU correctness/sanitizer evidence is pending; neither a full
DeepSeek/DSpark generation nor performance improvement is claimed.

Update 2026-10-10 (GLM real placement and AOT): the 121-shard real checkpoint
scan/export now finishes with exit zero. Actual executable-owner accounting
places layers `[0,25)` at 102,064,091,128 bytes and `[25,45)` at 91,516,526,804
bytes, each within 103,079,215,104 bytes including a 2 GiB reserve. Both exported
stages compile offline to `sm121` cubins on their assigned Sparks under 8 GiB /
no-swap user units. Both real checkpoint ranks are now starting on `.178/.179`
under separate 96 GiB / no-swap ceilings. The ingress requests at most two
greedy tokens with history 64; startup is running, not positive GPU generation.
Local placement/source evidence is
`/tmp/lunaflux-glm-real-placement-20261010.C0igHeAy`.
The actual tokenizer uses non-normalizing regex ByteLevel BPE with
`\\p{N}{1,3}` numeric groups, not Qwen's NFC/single-digit profile. This input
feature remains unsupported; it must not be silently tokenized as Qwen or raw
BPE. A new model-neutral offline prompt producer and `token-frames` CLI now
write canonical chunks from supplied IDs without scanning weight shards. The
wire/generation aggregate passes 152/152. This first controlled diagnostic uses
single-byte text `a`, whose exact ID 64 and EOS 154820 were read from the actual
tokenizer vocabulary/added tokens; `ignore_merges=true` makes the single-byte
mapping direct. This is not arbitrary text/chat-template support. Two-rank GPU
execution and independent correctness remain open. The exact live user units
are `lunaflux-glm-real-ingress-20261010-60fd6750-v1.service` and
`lunaflux-glm-real-egress-20261010-60fd6750-v1.service`; future checks must query
those units instead of starting duplicate full-checkpoint loads.
Internal rank communication is plain TCP; TLS/signing/public deployment
work is not a prerequisite for this feature diagnostic.

The first real two-Spark diagnostic subsequently generated IDs
`154822,154822`, then retired the request with `finish=Length`; ingress exited
zero at 06:59:13 CST. These IDs are `[gMASK]`, not a validated textual answer.
The single `a` input does not carry the normal GLM chat/template prefix, and
independent full-model numeric/output comparison is still required. Egress
reported `RemoteChannelError::Disconnected` when ingress closed after its
acknowledged release, exiting one. Do not describe the entire campaign as a
clean pass or the output as correct. Both ranks released their GPU allocations;
unit peaks were 90.1/95.9 GiB with zero swap peak. Downloaded run records are
`/tmp/lunaflux-glm-first-generation-20261010-v1`.
Commit `51f691a2` fixes this shutdown distinction: only acknowledged release plus
an orderly next-frame-boundary EOF ends cleanly; EOF before release or in a
partial frame remains a failure. Native control/transport regression passes
11/11, including reuse after release. The fix is not yet rerun with real weights.

Update 2026-10-10 (checkpoint-connected learned pools): the model adapter now
binds attention and indexer compressor `wkv.weight`, `wgate.weight`, `ape` and
`norm.weight` to their actual separate shapes and BF16/F32 storage. Its prepared
owner streams these four banks without expanding BF16 parameters into a host
F32 arena, accounts for weights plus retained state plus frames, and contributes
five launches to the containing queue. Layer ratios and indexer widths are
supplied by the model adapter; the shared immutable pooling IR is unchanged.
This output is deliberately before compressed rotary/quantization/cache
publication, not completed compressed attention or indexer selection.
The adapter's native suite passes 7/7, including all Flash/Pro layer-ratio
combinations and bounded actual-shape checkpoint-plane streaming fixtures.

Update 2026-10-10 (learned gated compressor core): immutable
`LearnedGatedPoolPrecision` now separates per-feature pooling semantics from
CUDA lowering and prepared persistent-state ownership. BF16 input and learned
projection parameters compute in F32, learned F32 position biases enter the
featurewise softmax, pooled values publish BF16 before learned RMSNorm, and
compact output metadata carries group-start positions for later rotation.
Preceding-window overlap is a semantic parameter, not a model-name/CUDA branch.
Five effects join the containing queue; startup initializes retained KV/scores
using bounded 64 KiB chunks, and the frame exposes a sticky append error.
The focused plan/source/prepared-frame tests pass 5/5, the precision IR suite
passes 36/36, and 160 prepared submissions measure zero warm heap allocations
with deterministic resource balance. Native checks use the existing legacy
warning exclusions; this is not a clean aggregate warning-migration claim.

GB10 `sm121` component correctness and memory/leak, race and synchronization
checks pass under 2 GiB/no-swap user units. Ratio-4 overlap covers 21 tokens in
mixed 0/1/2/3/4/5-row chunks (five pooled outputs); ratio-128 covers 135 tokens
(one pooled output). Learned F32 retained state and final normalized BF16
outputs match the small independent oracle exactly; a gap leaves state unchanged
and poisons history. Local evidence is
`/tmp/lunaflux-gated-pool-20261010.CgLmwc6y`, remote evidence
`/tmp/lunaflux-gated-pool-20261010.bziLvoNa`; downloaded probe SHA-256 is
`7ec241a2c2bb0956a304837acc5b67acca78a0c2c62c07274f2c9603d1972fec`.
Checkpoint binding, compressed output rotation/quantization, cache publication,
learned indexer and complete decoder integration remain; no whole-model or
throughput result follows from this component test.

Update 2026-10-10 (learned attention output integration): model-neutral
`RotaryTensorPrecision` and `GroupedAttentionOutputPrecision` now describe
inverse suffix rotation and the two learned output projections, including
three bounded scratch frames. Output-A retains compact E4M3/UE8M0 checkpoint
storage but rounds each decoded parameter to BF16 before its F32 dot product;
Output-B retains the existing dynamic block-128 FP8 activation semantics.
These distinct numerical contracts are not collapsed into one projection mode.
`DeepSeekWindowSublayer` composes the twelve retained-window input/attention
effects and three output effects in the caller's single prepared queue. It
binds the actual `wo_a`/`wo_b` checkpoint planes and returns hidden-width output.
The containing decoder still owns mHC, residuals, MoE and distributed work;
compressed layers remain rejected, so this is not complete model execution.

The affected native aggregate passes 100/100 with the existing legacy-warning
exclusions; the separate wire/generation aggregate passes 110/110. Its
fifteen-stage prepared queue performs zero measured warm heap
allocations, and deterministic close balances resources. GB10 `sm121` component
correctness and memcheck/leak, racecheck and synccheck pass with 2 GiB/no-swap
user-unit limits. Small independent CPU fixtures cover base and YaRN inverse
rotation, live rows 0/1/2/3, positions through 1,048,575 and deterministic replay.
Final hidden outputs match the fixture oracle exactly; intermediate largest
absolute errors are 0.001953125 (inverse rotation) and 0.0009765625 (Output-A).
This is not real-checkpoint accuracy or throughput. Local evidence is
`/tmp/lunaflux-attention-output-20261010.vXsq00BH`, remote evidence
`/tmp/lunaflux-attention-output-20261010.ji2Z52GT`; downloaded binary SHA-256 is
`8d3bbe218bf4b6dfb30316cdeb2c38dc73104305bbe89fa65b705506d111e819`.

Update 2026-10-10 (retained shared-KV window execution): a model-neutral
`WindowSharedKvPrecision` plan now accounts for the retained ring, sticky
history/error state, append descriptor and BF16 attention output. Its prepared
frame contributes reserve, causal attention with learned sink, exact-bit ring
copy and history publication to the containing executor. Attention reads old
history before overwriting slots, including when a new prefill chunk exceeds
the window; reset, positional gaps and context overflow have explicit behavior.
`DeepSeekWindowAttention` binds the actual F32 `attn_sink` and composes twelve
launches with the checkpoint-backed query/KV and rotary prefix. Compression-zero
layers only are executable through this boundary; compressed layers are
explicitly rejected. Output is still before inverse RoPE and output projections,
so this does not constitute a complete attention sublayer or model runner.

The affected native aggregate passes 136/136, including the unchanged zero-warm-
allocation wire/rank/generation regressions and a twelve-stage prepared queue
with deterministic resource balance. GB10 `sm121` component correctness,
memcheck with full leak checking, racecheck and synccheck pass under 2 GiB
memory/no-swap user units. The latest numerical probe covers 256/13/1/127/256
token chunks, inactive rows, reset, sticky invalid positions and context overflow;
largest absolute error is 0.000122070312, and retained ring words match exactly.
The four-kernel 256-row/two-head/512-width/128-window probe measures 1.180608 ms;
this is not model throughput. Local evidence is
`/tmp/lunaflux-shared-window-20261010.usL8tzuN`, remote evidence
`/tmp/lunaflux-shared-window-20261010.gDOMLdyI`; downloaded binary SHA-256 is
`27aa9db8e64824976e736cf19049218342e84465b4915cb347f3de97e2a809d0`.

Update 2026-10-10 (checkpoint executable): `cmd/glm_checkpoint` now connects
bounded real-config/shard inspection, actual-footprint two-rank placement,
offline source export, prepared GLM stages, activation/control owners and
canonical request-frame execution. Each host uploads only its placed layers
and text boundary. It consumes pretokenized canonical prompt frames rather
than providing text HTTP serving; self-feeding greedy continuation is added
as described below.
Native compilation passes; whole-checkpoint/two-host GPU execution is untested.
The exact committed `60fd6750` archive now also builds natively on Spark `.178`
with the September 20 ARM toolchain, existing warning exclusions and two build
jobs, under an 8 GiB/no-swap user service. The toolchain is isolated under
`/tmp/lunaflux-glm-arm-build-20261010.WGAwEo3b`, not installed over the host's
older incomplete installation. Downloaded archives and build failures are
preserved. A clean dependency archive excludes macOS AppleDouble source
sidecars; those sidecars caused the earlier invalid-UTF8 build failure.
The resulting ARM executable SHA-256 is
`c8ffb596925dea7ac8a314fb44faee7f44f87aa92ddb234c1d56a07a4d5102d0`.
It passes the actual staged GLM config on `.178` and `.179` under 1 GiB/no-swap
user services, without checkpoint payload scans or GPU allocation. Its `.179`
copy is under `/tmp/lunaflux-glm-arm-run-20261010.vx7kMsKI` with matching hash.
This is executable
portability/config validation, not weight placement or generation. Local build
artifacts are `/tmp/lunaflux-glm-arm-build-20261010.nXXzcHQo`.
The installed real config initially failed on root ModelOpt quantization
metadata. The parser now validates that NVFP4 descriptor and explicit RMS
epsilon; the exact downloaded config (SHA-256
`5db46f44956e4a8a0cc8ed54b6d77bf99dd7c1ec90c58975d1952560768513d5`)
passes the native config-only runner: 45 layers, hidden 4096, vocabulary 154880.
Config regressions pass 5/5; startup reader/materializer regressions pass 18/18.
The expanded rank/control/checkpoint regression aggregate passes 129/129.
These checks do not validate weight placement or model generation.
Linux checkpoint-executable linking exposed omitted pthread/dynamic-loader
libraries, hidden by macOS. Explicit `-pthread -ldl` fixes the native Linux
build; no native ABI or kernel arithmetic changed. The CPU-only real-shard
placement job runs before any GPU allocation. The data node uses cgroup v1:
requested user-unit MemoryMax/MemorySwapMax are not per-service kernel limits.
A verified 4 GiB address-space limit bounds this CPU process instead; no
per-unit no-swap enforcement is claimed. At 11.2 GB scanned the process used
about 7 MB RSS. The first one-hour scan was stopped and replaced by a two-hour
bounded scan because measured authentication throughput would exceed its
original deadline. No completed placement or GPU result is claimed yet.
The installed management-node source has 120 main shards plus input scales;
the earlier arbitrary source halves are not checkpoint-verified placements.

Update 2026-10-10 (actual header failure): the two-hour CPU scan exited with
status 1 before placement. Its original pipe did not retain the error, but a
bounded metadata-only replay of all 121 actual headers reproduces
`TooManyTensors(120)` under the old 20,000-per-shard policy: the final 4.84 MB
input-scale shard contains 37,152 scalar tensors. The checkpoint command now
allows the same bounded 500,000 total population within any single shard;
32 MiB headers and file/aggregate limits remain. Generic overlap validation
uses an O(n log n) ordinal sort, preserving public header order and coordinates.
All 150,226 actual header records validate with the revised reader; an offline
binding comparison finds zero missing or shape/dtype-mismatched operands among
110,110 required text-decoder tensor planes. These are metadata previews, not
payload authentication, successful placement or GPU generation.

Focused checkpoint/reader tests pass 16/16, including authenticated 37,152-scale
loading, range-order preservation and overlap-oracle regressions. Native checks
use the existing legacy warning exclusions (25/29/79/92, plus deprecated API
warning 20 on the clean Linux archive); the repository-wide
strict warning migration is not complete. Header weights alone require
186,533,449,720 layer bytes plus 2,537,562,112 text bytes. The best contiguous
weight-only split is at layer 24: 95,391,266,368 / 93,679,745,464 bytes. Therefore
the previous 90 GiB budgets with 2 GiB reserve cannot work. Re-run authenticated
placement with live-host-headroom-derived budgets before GPU allocation; all
state/workspace/module/process overhead still needs the executable's accounting.
The exact `7643d611` source archive builds on the Linux management host with
the stated legacy warning exclusions. The reader's standalone Linux tests
also require explicit `-pthread -ldl` for the toolchain allocator/async runtime;
macOS had hidden that dependency. This is test/executable linking, not a changed
native ABI. A new CPU-only authenticated export uses 96 GiB per-host total
budgets, 2 GiB reserve, a 4 GiB address-space cap and a three-hour deadline;
stdout/stderr are retained in the user-service journal. Its placement result is
pending, and no checkpoint GPU allocation is authorized by the header preview.

Startup upload now supports scoped pinned/authenticated shard reuse across
layer owners, while retaining stamp/replay checks and deterministic file close.
Cross-shard duplicate detection uses an index rather than scanning every prior
tensor. Inspection and upload authentication remain separate startup passes.
No cryptography/filesystem work enters the token step. Full model numerical,
memory and performance validation remains open; no GPU weights were loaded.

Update 2026-10-10 (two-rank request execution): rank-local workers now upload
the same canonical request metadata without publishing partial model responses.
A plain-TCP stage-control channel carries prepare/run/retire/commit/release;
the model-neutral two-rank coordinator waits for both ranks and activation DMA,
commits request history, and publishes exactly one terminal completion. GLM
stages bind those rank effects; queue retirement can be observed by both the
rank and transfer owner without resubmission. Idle service time is unbounded,
while active frames and activation waits retain bounded deadlines.

Real-loopback/native-device-double tests cover chunked prefill, final prefill,
decode, terminal token delivery, byte-exact residual transfer, failed samples,
cancellation, and balanced resource release. Warm rank polling has zero
measured heap allocations; network async polling is not claimed heap-free.
The expanded final affected aggregate passed 106/106. No actual checkpoint
GPU generation or throughput claim is
added. The checkpoint-backed executable startup/placement now exists as above;
actual numerical generation remains. DeepSeek learned-head/complete decoder and
MiniMax bounded-memory whole-model composition remain open. TLS work is paused.

Update 2026-10-10 (self-feeding generation): model-neutral serial greedy
continuation now feeds actual completed tokens into successive canonical
decode frames, with fixed token storage and alternating frame owners. It
preserves the reserved page table/capability order, request/model generations,
sampling seed and completion slot; no synthetic recipe or KV ownership is
introduced. Full context/sample/sequence budgets are checked before continuation.
EOS/length stop before another submission; cancellation suppresses output while
the caller still drains/releases both ranks. Failed/foreign completions poison
the state and cannot be retried as success. `cmd/glm_checkpoint generate`
connects prompt chunks and this continuation to the existing loaded pipeline;
generated token-ID output is written only after request retirement. Text
tokenization/HTTP serving and actual-checkpoint GPU numerics remain untested.

Seven black-box continuation regressions pass, alongside the existing wire
suite (91/91 total). The loopback two-rank regression now produces 7→9→11 and
checks each actual token becomes the next input, remote-commit publication,
both histories and deterministic release. A warm allocation probe initially
found one temporary sampling object per decode. Shared canonical greedy-wire
emission removes it: fourteen decode continuations now allocate zero measured
heap objects. No network-poll allocation or physical speed claim is added.

Update 2026-10-10 (DeepSeek learned output): `LearnedTextIoPrecision` now records
normalized F32 learned sigmoid controls, ordered residual-stream reduction and
one BF16 round before the ordinary final normalization/head/greedy boundary.
The prepared frame contributes four launches to the caller's queue, with no
extra completion or warm allocation. The DeepSeek adapter binds actual
`hc_head_fn`, `hc_head_base`, `hc_head_scale`, `norm.weight` and `head.weight`
shapes/dtypes; GLM's unweighted mean remains unchanged. Invalid control/input
values overwrite live reduction rows with NaN rather than leave stale output.
The expanded affected native suite passes 109/109, including official-shaped
control-weight streaming and repeated prepared-frame resource balance.
A GB10 fixture passes exact greedy tokens, deterministic repeats, untouched
inactive rows and all tested numerical boundaries with zero observed error.
Memory/race/synchronization sanitizers report zero errors and no leaks. The
live cap check confirms 2 GiB memory / zero swap; measured peak is 66.2 MiB.
Generated source SHA-256 is
`07254f9c9d00de8791603886d46b91bf3a5ea2d7416e47b5f45f19e5ceafe938`;
probe source is
`c00452aff1ad8ed634311b2f735fa26970c45f0891416d44541120f6cb9a33a2`.
This completes a component boundary, not the DeepSeek decoder/DSpark model run.
The 43-test learned-head/precision/source subset also passes on Linux after
three native test packages explicitly link the allocator's system libraries.
Whole-checkpoint numerics remain untested.

Update 2026-10-10 (DeepSeek quantized query/KV prefix): a model-neutral immutable
precision plan, CUDA source and prepared frame now connect Query-A → learned
RMSNorm → Query-B → additional unweighted BF16 head norm, plus shared-KV
projection → learned RMSNorm. The adapter binds the actual five compact banks
without expanding FP8 weights; all scratch/weight bytes are accounted before
upload. The affected native suite passes 40/40, including official-shaped
checkpoint streaming and missing-layer regressions.
The warm six-stage queue allocates no measured heap objects and releases its
resources deterministically. The first physical compile found a missing
`cuda.h` include behind the CUDA_VERSION guard; that generator bug is fixed.
The corrected GB10 fixture checks all six intermediate boundaries over live
row counts 0/1/3, zero/nonzero input and repeated execution, with zero observed
absolute error. Memory/race/synchronization sanitizers report no errors/hazards
and zero leaked bytes. Source/probe/binary hashes match their downloaded copies.
Local artifacts: `/tmp/lunaflux-query-kv-20261010.HIcAa8zE`; generated source
SHA-256 `0de8979212d79fbc38debf43768d65ce79da6145f10bc0f2d54943cc7c9579c6`.
This is an unsharded component, not rotary/cache/attention execution or a
whole-checkpoint DeepSeek/DSpark result. Actual GLM checkpoint transfers to both
Sparks and its CPU placement scan continue independently; no GPU weights have
been loaded by those jobs.

Update 2026-10-10 (DeepSeek attention input execution): `DeepSeekAttentionInputs`
now composes the checkpoint-backed six-stage prefix with full-head adjacent-pair
rotary and blockwise KV simulation. A pure `RotaryKvPrecision` plan records
Base/YaRN frequency policy, BF16 publication, E4M3 encode/decode, block width,
amax floor and scale law. The model adapter disables YaRN for compression-zero
layers and uses compressed-theta YaRN otherwise, matching the installed DSpark
reference. Only the 448 non-RoPE components of a 512-wide KV head are simulated
in block-64 groups; the 64-component positional suffix remains BF16.
The combined owner budgets all eight scratch frames and packed weights and
returns eight borrowed launches, without an additional completion boundary.
The precision/source/checkpoint adapter subset passes 38/38 locally; the two
new composed executor regressions pass, including 32 warm steps with zero
measured heap allocations, no blocking sync and exact resource release. Native
checks retain the existing legacy warning exclusions, including reserved-name
warning 35 in an unrelated older fixture. The original broader executor run
finished 88/91: three serial allocation tests failed in native debug mode.
Debugger stacks and generated C identified boxed mutable scalar accumulators
in wire UInt64 reads, frame sizing and checksums; release tests already passed
8/8. Functional loop state and immutable checked offsets remove those debug
allocations without changing wire bytes or relaxing checks. The affected wire
and serial regressions now pass 93/93 in native debug, with the same zero warm
allocation assertions. The clean Linux rotary/query-KV subset passes 38/38.

The GB10 component probe passes base/YaRN and both scale policies at positions
0/65,536/1,048,575, zero/nonzero inputs, live rows 0/1/3 and deterministic repeats.
Query prefixes and simulated KV prefixes are word-exact against the CPU oracle;
maximum rotary absolute error is 0.0078125 (not a bitwise arithmetic claim).
Memory/race/synchronization sanitizers report zero errors/hazards. Physical
units use a 2 GiB cgroup-v2 memory ceiling and zero swap; this tiny test does not
load model weights. Local artifacts: `/tmp/lunaflux-rotary-kv-20261010.53vJG5vl`;
remote artifacts: `/tmp/lunaflux-rotary-kv-20261010.7JsIaHnK`. Source SHA-256
`83801ac789162edec6f206bf9508eff75edea4af4ffae21e2b826e8e8d1da84c`, probe
`521271bb8ae33317e04807fabbca7058a62d086958697d68d651ac36e56a90e1`, binary
`ec3ff9b09c48ede53080547763f668bc27f9868d0a1e644e1bdd3f5281b4c595` match downloads.
The initial race-check admission paused on residual 3% GPU utilization after
memcheck; no workload started then. After idle/no-compute-process verification,
only the remaining sanitizers resumed. Full sliding/compressed-cache mutation,
learned compressor/indexer, attention/output/mHC composition and whole-model
DeepSeek/DSpark numerics remain open. This is a component execution result,
not complete generation, packed FP8 cache serving or performance evidence.

The expanded affected native aggregate passed 221/221 locally. Exact commit
`ef95b5c0` also builds the checkpoint runner on Linux and parses the installed
GLM config successfully. Linux continuation/wire test linking exposed the
runtime allocator's pthread dependency on this pre-glibc-2.34 host; explicit
native system-library flags in the two test-owning packages fix it. The corrected
focused suite passes 91/91 on both macOS and Linux. These are not checkpoint
GPU numerical results. The full-shard CPU scan remains active (79.2 GB read at
29 minutes, approximately 19 MB RSS); completed placement is still pending.

Update 2026-10-10 (activation execution edge): a model-neutral host-staged
activation link now waits for producer queue retirement, downloads only live
rows through a pinned DMA lane, sends unchanged tensor bytes over plain framed
TCP, and publishes receiver completion only after upload retirement. Storage
is fixed and budgeted at startup. Step epoch/row checks prevent a different
step from entering device storage; they are execution semantics, not TLS or
artifact admission. A GLM binding retains an activation lease that blocks stage
resubmission/closure until transfer completion or explicit drained cancellation.
The producer's submitted row count must equal its transferred row count.

The real-loopback/socket + native-device-double regression passes byte-exact
transfer, shorter-batch tail preservation, wrong-step rejection and resource
balance. The updated official-shaped GLM first-stage fixture also passes stage
lease/reuse, live-row and cancellation checks. These are not two-host GPU or
real-checkpoint numerical results. Metadata propagation, remote stage control
and terminal token delivery are now implemented as described above; actual
checkpoint whole-model execution remains unfinished.

Both composed GLM CUDA sources (layers 0–21 and 22–44, one live-row capacity,
history 64) compile to sm121 cubins on .178 GB10 with CUDA 13.0.88. Each compiler
invocation uses a systemd user unit capped at 2 GiB/no swap/180 seconds; both
exit zero. CUDA reports existing unused-variable warnings. The intervals are
composition probes, not verified memory placements for actual checkpoint banks.
Downloaded cubins match the remote SHA-256 values:
`5a55ea7cadcffdfcff03b2ae5f01b20a19f8752d9627dde53d321a602f1c61d3`
and `1b87370c872e8d68b4a9b626eec839b1f72d335b672c17a2da2ac7475a424e10`.
Sources and downloads are preserved under
`/tmp/lunaflux-glm-stage-aot-20261010.MaAHmXzr` locally and
`/tmp/lunaflux-glm-stage-aot-20261010.SJAGDuv0` remotely. Source generation is
an offline exporter; no request-path JIT or new GPU inference claim is added.
The transfer package passes scoped warning-denied native checking with legacy
warning 79 disabled. Whole-worktree warning-denied checking is not green:
even with warning 79 disabled, existing qualification-test warnings remain.
The first aggregate transfer run exposed process-global device-double counters
being changed by concurrent async fixtures. Test-only device-lane serialization
retains the exact counter assertions; runtime execution is unaffected.
The corrected affected release-mode native aggregate passes 97/97 across the
partition/precision plans, serial frame, GLM adapter, packed-MoE execution and
plain remote channel. Regenerated post-transfer AOT sources retain the exact
hashes of the compiled snapshots above. No whole-checkpoint GPU inference or
new three-model throughput result is claimed.

Update 2026-10-10 (stage assembly): a prepared GLM stage now owns boundary text
weights, its connected decoder interval and distinct input/output residuals.
Ingress loads only embedding, egress only norm/head, interior neither; the same
generic numerical lowering is reused. A pure contiguous partitioner consumes
exact checkpoint layer footprints and per-host budgets, including boundary
weights and explicit stage overhead, without GPU allocation or a hardware/model
branch in the compiler. All layers must be placed exactly once. Prepared stage
geometry must match worker ports; partial stages cannot publish model tokens.
The official-shaped first-stage native regression streams embedding plus one
dense block into one 27-launch queue and executes 32 steps / 864 launches with
zero measured warm heap allocations or blocking waits. Budget-minus-one and
foreign geometry are rejected before stage allocation; active cancellation
releases owned resources. That fixture deliberately omits final norm/head, so
it verifies boundary-only binding, not real-weight numerical execution. Whole-
model cross-host request/activation transfer and checkpoint generation still
remain; this feature does not close GLM/DeepSeek/MiniMax end-to-end inference.
The affected native regression run passes 90/90, and the final first-stage
fixture rerun passes after the row-bound/retry check. Format/API generation and
ordinary native checking pass. Warning-denied checking still fails on thirty
existing dependency migration warnings; it is not counted as a green release.

Update 2026-10-10 (request bridge): a model-neutral single-contiguous-request
worker now stages validated request frames into prepared token/count/reset/row
ports, polls asynchronous input/model/output completion, and publishes the
existing completion protocol. GLM decoder slices can bind their effects and
sparse append-error ports. Chunked prefill/decode continuation, vocabulary/context
bounds, execution failure, cancellation and startup memory ceilings are covered.
The original bridge allocated three boxed wire values per step; reusable
epoch-retaining storage and Boolean polling remove all 96 allocations across
the 32-step native regression. Normal progress performs no blocking device sync.
The corrected transfer double also rejects host reads/reuse before completion.
This is not yet a complete checkpoint runner or a multi-host model worker.
Final affected release-mode native suite: 140/140 passed across worker-wire,
serial frames, GLM adapters, packed-MoE execution and text source generation.

Small GB10 GPU text ingress/egress now passes a CPU oracle at rows 5 / hidden 8 /
streams 4 / vocabulary 17 / selected outputs 2: embedding is bitwise equal,
normalization/logit maximum absolute error is zero on this fixture, greedy IDs
match, and repeated outputs are bitwise identical. Memcheck reports zero errors
and zero leaked bytes; racecheck zero hazards; synccheck zero errors. Each run
is bounded by 2 GiB host memory, no swap and 120 seconds. This validates the
generated text kernels, not full GLM dimensions, real weights, decoder arithmetic
or serving speed. Full worker inference and two-host execution remain open.
Local downloaded evidence archive: `/tmp/lunaflux-text-io-gpu-20261010.HOs9zeNG.tar.gz`,
SHA-256 `39518a22889cff4f148406d8fed805bbfda8d927b554f18935d16abf649ef9a1`.
Native warning-denied check is still blocked by existing dependency warning
migration; warning 79 suppression in affected tests is not a green release gate.

Update 2026-10-10 (text I/O): a generic immutable precision plan and prepared
execution frame now connect token embeddings and selected-row mean/RMSNorm/head/
greedy around the decoder queue. The GLM adapter binds the installed global
BF16 tensor names and preserves intermediate BF16 rounds. No DeepSeek learned
head parameters are invented. Selecting one output row after 4096-token prefill
uses 627,712 bytes of private normalized/logit workspace; embedding/head weights
and residual frames are separate costs. Native device-double tests exercise the
combined queue and cancellation/release. GPU text numerical/sanitizer validation,
full worker inference and two-host execution remain pending; this is not a
new serving benchmark or a claim that the three complete models now run.
The affected release-mode native suite passes 72/72; all six new text regressions
also pass independently. Native check and format/API checks pass. Warning-denied
check remains blocked by unrelated dependency warning migration; it is not
reported as green. No GPU workloads or deployment-hardening changes were made.

Update 2026-10-10: checkpoint-backed decoder slices now connect consecutive GLM
layers through at most two reusable four-stream BF16 residual buffers. Recurrent
and sparse layers have a queue-free prepared-frame mode; the slice resolves the
official KDA/DSA and dense/MoE schedule at startup and owns one model-level queue
and completion per step. Aggregate weight/bank/state/workspace budgets are checked
before device allocation/upload. DSA-containing slices remain single-request;
recurrent-only slices preserve independent request slots. All 45 layer plans
can be constructed without loading device weights. This is decoder execution,
not embedding/head/sampling, full worker inference or two-host transport.

The three-early-layer native fixture executes 32 steps / 2,496 launches with
zero measured hot heap allocations or blocking waits, checks total-byte
accounting, then cancels and releases its layer owners. The generic two-layer
frame test executes 32 steps / 1,024 launches with the same warm-path properties.
Fixtures use zero-filled checkpoint storage and a native device test double,
not actual-checkpoint GPU arithmetic. No TLS/signing/deployment hardening was
added; necessary memory accounting stays in startup rather than token execution.
The final affected release-mode native suite passes 68/68, including the sparse
layer's queue-free frame prepared inside a 37-launch slice. Native check, format
check and API generation pass. Warning-denied checking remains blocked by the
existing `internal/nccl/api.mbt:200` implicit-method migration warning; it was not
expanded into unrelated cleanup. No new GPU or deployment claims are made.

Complete sparse decoder composition is now implemented: the generic immutable
decoder plan adds output projection and a residual-stream envelope to retained
sparse attention. `IndexedSparseDecoderFrame` binds twenty-three launches;
`GlmSparseLayer` appends the existing routed/shared FFN and its second envelope
in one thirty-seven-launch executor. Recurrent and sparse layers share FFN
construction, routing and streamed expert-bank upload rather than maintaining
two implementations. The Flash adapter binds learned output projection and
pool APE at their actual BF16 shapes. This is feature wiring, not a claim of
full-model serving, efficient kernels or checkpoint numerical equivalence.
The affected release-mode native suite passes 62/62. A small generic block
fixture and an official-shape sparse checkpoint fixture each execute 32 steps /
1,184 launches without measured hot heap allocations or blocking waits, then
cancel and release resources. The latter uses a native device test double and
zero-filled sparse checkpoint files, not real GPU checkpoint arithmetic.
Ordinary native check, format check and API generation pass; warning-denied
checking is still blocked by existing unrelated dependency warnings. No GPU
workloads or deployment-hardening changes were performed for this composition.

Hidden-to-projected sparse execution now has a model-neutral immutable precision
plan, AOT source composition and startup-bound frame. Eight projections, two
low-rank RMSNorm stages, affine index-key LayerNorm and packed-KV splitting feed
request-owned history and pooling/index/attention in a single eighteen-launch
queue. The native regression performs 32 steps / 576 launches with zero measured
hot heap allocations/blocking waits, then cancels and closes resources. Fifty
focused native tests pass across precision, source, frame and GLM adapter paths.
The small GB10 GPU fixture matches all thirteen intermediate buffers exactly,
checks single-row versus multi-row projection bitwise agreement, and executes
hidden -> projections -> append -> sparse attention across prefill/decode with
the selected attention output matching the ordered CPU oracle. Memory/leak,
race and synchronization checks report zero issues. The Flash adapter binds
twelve exact checkpoint weights, including rank-one index LayerNorm scale/bias;
one query's projection frame is 181,056 bytes, weights 115,610,112 bytes.
These are component features/reference schedules, not a real-checkpoint or
whole-model throughput result. Index LayerNorm epsilon is explicit rather than
guessed from the unrelated low-rank RMSNorm or DeepSeek source. DSA block
composition is now present; checkpoint numerics, efficient schedules and model
worker execution remain.

Request-owned projected sparse history now retains index keys, pool gates and
attention K/V between prefill chunks and decode. Its immutable storage plan
separates history capacity from frame rows; device lowering orders reserve/reset,
parallel BF16 copy and publication before pooling/index/attention in one queue.
The small eight-query / 32-history fixture owns 1,216 bytes of history/metadata
plus 580 bytes pool/index scratch. Distinct request owners have separate storage;
idle preserves history, request reset clears old validity, and overflow reports
a device error rather than truncating. The native queue test runs 32 steps /
192 launches with no measured hot heap allocation or blocking wait and releases
after active cancellation. The affected suite passes 99/99; small GB10 tests
match the independent oracle exactly, with bitwise chunked-prefill/decode
agreement, request isolation, reset/reuse and clean memory/leak/race/sync checks.
This earlier step implements cache append, not batched slot
scheduling, whole-model serving or a performance claim.

Projected indexed sparse attention now composes pooling, deterministic index
selection and attention from a model-independent precision plan. Decode query
capacity no longer forces query/output/selection storage to match retained
history capacity. A one-query / 32K-history plan uses 8,204 bytes for selected
indices and 2,269,200 bytes total owned pool/index workspace. Histories and
projected inputs are borrowed and excluded from this workspace.

The prepared producer has three launches; an IndexShare consumer borrows its
selection for one attention launch and owns zero workspace. The combined native
regression runs 32 steps / 128 launches without measured hot heap allocations or
blocking waits, then cancels and releases owners in dependency order.
The 96-test affected native suite and small GB10 GPU fixture pass. The GPU test
uses 15 valid keys within capacity 32, three prefill queries and one decode query:
maxabs 0, bitwise prefill/decode agreement, zero inactive output, and zero memory,
race and synchronization sanitizer errors. These are correctness-grade serial
kernels, not a model throughput result. Complete DSA blocks,
decoder/worker integration and two-host execution remain unfinished.

Recurrent GLM MoE decoder composition now joins both mHC envelopes, KDA,
automatic grouped routing, packed NVFP4 routed experts, BF16 shared expert and
BF16 residual publication in one 34-launch queue. The generic MoE launch frame
reuses existing binding functions instead of adding another queue/completion
boundary. Its standalone collective-aware execution remains available for
partitioned banks; the new complete local composition requires all experts.

The official-sized sparse checkpoint regression streams all 288 experts' three
packed projections, scale planes and global scales, plus router/shared/envelope
weights. Aligned owned weights total 4,406,639,960 bytes; rows 32 / sequences 16 /
slots 32 workspace is 152,584,580 bytes, excluding borrowed I/O/module/stream.
Thirty-two native device-test-double steps execute 1,088 launches without
measured token-step heap allocations or blocking waits, and active cancellation
releases the complete layer. Installed layer-4 headers were checked directly.
The affected release-mode native suite passes 62/62 tests, including existing
dense, routing, precision and checkpoint-upload regressions.
The warning-denied check remains blocked by 30 existing implicit-implementation
promotion warnings in dependencies; ordinary native checking has no errors.
This is executable composition and checkpoint binding, not actual checkpoint
GPU numerical execution or whole-model performance. DSA, workers and two-host
full-model execution remain unfinished; no TLS/signing/admission work was added.

The GLM early dense decoder block now executes both mHC envelopes, KDA and
dense SwiGLU in one 26-launch queue. Separate gate/up weights produce a shared
BF16 product once, and down consumes it; no intermediate CPU handoff is added.
Generic precision IR owns geometry and numerical rounds, the prepared frame
owns its product/functions, and only the GLM adapter binds model names. All
seven additional tensors match installed checkpoint header shapes/dtypes.
At rows 32 / sequences 16 / slots 32, weights are 579,060,440 bytes and workspace
148,669,440 bytes, excluding borrowed I/O/module/stream. The native test double
runs 32 steps / 832 launches without measured token-step allocation or blocking
waits. Focused native tests pass 58/58. This completes early dense-block
composition, not a whole-model worker.

On idle .178 GB10, a tiny complete 26-stage numerical fixture matches the
independent oracle exactly for final residual output and shared MLP product;
maximum recurrent state error is 2.27e-8. Raw history is exact, prefill/decode
outputs and state are bitwise equal, and idle publication is untouched. Memory,
leak, race and synchronization checks report zero issues under a 2 GiB/no-swap
process cap. These are generated-kernel component tests, not actual checkpoint
weight execution or a serving benchmark. The current ordered projections remain
correctness-first. Recurrent MoE/DSA blocks, real-weight GPU execution, complete
three-model workers and two-host execution remain unfinished. No new TLS,
signing or admission work was added.

The GLM recurrent attention sublayer now includes its complete mHC envelope:
four residual streams → function/collapse → positive Sinkhorn/input RMSNorm →
KDA branch → residual-stream publication. Its nineteen checkpoint parameters
bind exact installed header shapes, with 276,275,820 weight bytes and 146,043,392
workspace bytes at rows 32 / sequences 16 / slots 32. The native test double runs
32 steps / 640 launches in one reusable queue with zero measured token-step
allocations or blocking waits; aggregate focused tests pass 49/49.

On .178 GB10, a small full 20-stage fixture passes both numerical post contracts
against an independent oracle. Residual output, collapse and input norm match
the oracle exactly; maximum F32 control/state errors are 7.62e-8 / 2.27e-8.
Prefill versus eight decode frames gives bitwise-identical output/state; idle
frames are untouched. Memcheck, racecheck and synccheck report zero issues.
This closes attention-envelope composition, not full-model inference: real
checkpoint GPU execution, dense/MoE sublayer composition, DSA, workers and
two-host execution remain open. No throughput improvement is claimed.

Checkpoint-backed KDA execution now has a GLM layer owner. Its fifteen named
parameters bind to the actual installed layer-0 header's BF16 matrices,
BF16 `[8192,1,4]` convolution tensors and F32 control vectors without transpose
or host expansion. The generic uploader owns exact allocations and streams all
planes through one scratch chunk. The full-sized native fixture uploads
275,481,088 weight bytes, prepares the 16-stage queue, executes 32 steps with
zero measured step allocations/blocking waits and closes all resources after
active cancellation. Total weights plus rows-32/slots-32 workspace are
420,733,440 bytes, excluding borrowed I/O/module/stream. This uses the native
device test double, not a real-weight GPU correctness result or model benchmark.

The complete recurrent attention branch now joins hidden-to-Q/K/V projections,
two-stage forget, beta sigmoid, F32 decay, two-stage output gate, request-owned
convolution/delta, gated RMSNorm and output projection in one prepared queue.
Its 16 launches have one completion boundary. Pure precision IR accounts all
frames; the GLM adapter supplies exact geometry and model RMSNorm epsilon.
Thirty-two native steps execute 512 launches with zero measured heap allocations
or blocking waits. The combined focused native suite passes 47/47.

On .178 GB10, the small complete branch passes an independent double oracle
(maximum absolute error 3.61e-9), exact raw-history checks and bitwise full-prefill
versus token-by-token decode output/state. Memory/leak, race and synchronization
checks pass. This is component execution, not a full-checkpoint model test:
checkpoint binding, surrounding mHC/MLP/DSA layers and workers remain unfinished.
The ordered-F32 projections parallelize outputs but are not tensor-core GEMM;
no throughput improvement is claimed. No TLS/admission work was added.

Request-owned Q/K/V short convolution now executes before recurrent update in
the same prepared queue. Generic precision IR owns raw BF16 history semantics;
the GLM adapter selects 8192 channels / kernel four. Request-slot continuation,
reordering and reset use the same metadata for all four kernels. The Program
owns three histories and convolved frame buffers under explicit aggregate
startup budgets. Thirty-two composed steps have zero measured heap allocations
and blocking waits. The current focused native suite passes 42/42.

On idle .178 GB10, channel counts 8, 129 and 8192 pass an independent double
oracle and exact raw-history checks. One eight-token prefill and eight decode
frames produce bitwise-identical outputs, convolution histories and delta state.
All three composed GPU cases pass memory/leak, race and synchronization checks.
No throughput speedup is claimed. This earlier convolution-only check did not
cover projections or output; the complete branch above now does. Full model
workers remain unfinished. No TLS/admission work was added.

GLM KDA now has a request-owned recurrent delta Program in addition to its old
serial diagnostic source. The model adapter builds a generic BF16/F32 precision
plan; CUDA lowering assigns sequence/head CTAs and value-column ownership.
Explicit cache slot IDs survive batch reordering and reset on request reuse.
The AOT Program owns persistent state and a reusable nonblocking execution queue.
Thirty-two native steps have zero measured heap allocations and blocking waits;
the focused native suite passes 36/36. Format and native interface generation
pass. Warning-denied checks still encounter unrelated existing deprecations.

Three GB10 shapes pass an independent double-precision recurrence and ordinary
bitwise output/state comparison with the serial reference. New-kernel memcheck,
racecheck and synccheck report zero errors/hazards/leaks. The actual GLM
64-head / 128-key / 128-value, eight-row recurrence has median 47.63 us in five
alternating pairs. This is **not** a whole-model or reference-engine comparison:
the baseline is a deliberately one-thread correctness source. The new official
shape still uses 255 registers and spills; it is not claimed optimal. The old
reference's invalid CUDA float literal was fixed after real compilation exposed
it. Its instrumented full-size memcheck timed out; that failed capture is kept,
and sanitizer checks of the new kernel retain the independent CPU oracle.

[Recurrent execution results](RECURRENT_EXECUTION_2026-10-09.md) record exact
scope and timings. The downloaded artifacts match remote SHA-256 identities;
archive hash is `4a4e7f0ed7b335bf7b51f8ae781b7cc110296bbcb3502b546c72aa86b27e48ab`.
Real-weight GPU KDA execution, DSA/DeepSeek
attention, three-model workers and actual two-host execution remain unfinished.
No additional TLS/admission work was performed.

Automatic routing now joins the same MoE Program queue: BF16-input router
projection, score/correction, grouped top-k, routed experts, optional reduction,
shared expert and finalization. The caller no longer needs to compute expert IDs
or selected weights externally. Numeric IR owns BF16/F32 weight representation,
sigmoid/sqrt-softplus, stable ties and exact compact routing workspace; CUDA is
only its final lowering. GLM binds the installed BF16 [288,4096] router and F32
[288] correction vector. Actual-size native checkpoint streaming is tested.

Focused native tests pass 33/33, including 16 automatic-router queue steps / 160
launches with zero measured token-step heap allocations and active cancellation.
On idle .178 GB10, 18 generated-router GPU cases pass an independent host oracle
for BF16/sigmoid and F32/sqrt-softplus, live rows 0/1/2, choice-only bias,
normalization, stable ties and untouched inactive rows. Memcheck reports no
errors or leaks; racecheck and synccheck report no errors. The probe sources and
logs are at `/tmp/lunaflux-routing-feature-A4VeVi` on .178. This is executable
module correctness, not a full-model or throughput claim. Full decoder/worker
integration and actual two-host execution remain unfinished. TLS/admission and
unrelated deprecation-warning migration remain paused.

DeepSeek's automatic routing now also enters that queue. Installed checkpoint
headers confirm BF16 [256,4096] router matrices: layers 0–2 have I64
[129280,6] token-to-expert tables and no bias; layer 3 onward has F32 [256]
choice bias and no token table. The adapter translates these into generic
TokenHash/GroupedTopK selection IR, not model switches in the executor.
TokenHash preserves table order and gathers normalized sqrt-softplus scores;
it does not replace learned routing weights with uniform weights. The prepared
Program borrows the existing startup-narrowed I32 table arena at a layer offset.

The current focused native suite passes 29/29, including official-sized router
checkpoint streaming and 16 hash-routing queue steps / 176 submissions with
zero measured token-step heap allocations. Thirty generated GPU routing cases
pass (18 grouped, 12 hash), covering table order, nonuniform learned weights,
invalid IDs and untouched inactive rows. Memory/leak, race and synchronization
checks report no errors. Artifacts are under
`/tmp/lunaflux-routing-hash.ti3e5w` on .178; archive SHA-256 is
`65b2e58544a6c02f3e67a92a37dd2e63f0f48658287981febe51bbdf6017786b`.
These are component tests, not a
claim of complete DeepSeek generation or three-model throughput. Native check
has no errors; warning-denied check still encounters unrelated existing
deprecation warnings, which are not expanded into a migration project here.

The generic expert path now executes dynamic BF16-to-FP8/UE8M0 activation
quantization before both gate/up and down. Precision IR explicitly describes
amax floor, block geometry, SwiGLU rounding, routing-score placement, and compact
workspace sizes. The DeepSeek adapter supplies FP4/FP8 checkpoint mappings and
its F32-product/score-before-down semantics; GLM continues using its original
BF16-activation route. Both share bounded expert-bank upload and the ordered
AOT executor binder. No additional TLS or deployment-admission layer was added.

Focused native tests pass 54/54. On the idle .178 GB10 Spark, the 128-wide
FP4-weight/dynamic-FP8 expert probe passes six ordinary/zero-input live-row
cases against an independent blockwise CPU oracle, with maximum observed
absolute error 0. Input FP8 bytes and UE8M0 scales also match exactly.
Memory, race and synchronization checks report no errors or leaks.
The original NVFP4/MXFP4/BF16 expert probes still pass. Host memory stays at
4.2 GiB used / 117 GiB available, with no remaining GPU process.

This is module correctness, not a full-model or speed claim. Whole-model GLM,
DeepSeek/DSpark and MiniMax execution integration is still unfinished.
Warning-denied checks remain blocked by existing MoonBit deprecation warnings;
ordinary native check, format and interface generation succeed. Those unrelated
warning migrations remain paused.

The next executable slice adds typed F32 all-reduce/all-gather through the
existing private NCCL ABI, retaining the BF16 facade. DeepSeek shared experts
bind their actual FP8 E4M3 weights and UE8M0 128-by-128 scale grids, separately
from routed FP4 experts. The same precision plan, bank uploader and packed MLP
renderer implement both; no model-specific runtime was introduced.

A reusable MoE queue now executes routed experts, optional in-place F32
reduction, replicated shared expert, then BF16 finalization. Reduction is issued
into the same stream before shared work; completion uses nonblocking polls.
The pure combination IR records whether the routed result rounds before the
shared add. Two native queue regressions pass, including 16 cycles each with
zero measured heap allocations, pending-reduction ordering, and single-rank
cancellation. An existing boxed mutable index in collective submission was
replaced by immutable branch selection. The supporting native tests pass 48/48
and device tests pass 20/20; the NCCL exact-TU ASan/UBSan gate passes.

The generated shared FP8 MLP/finalizer passes 12 additional GB10 cases covering
zero/nonzero input, live rows 0/1/2, both output rounding placements and untouched
inactive rows. Together with six routed cases, all 18 cases pass; memcheck,
racecheck and synccheck report zero errors. This does not verify actual two-host
NCCL execution or complete checkpoint serving. The bounded physical artifacts
are at `/tmp/lunaflux-shared-moe.66gstK` on .178; generated header SHA-256 is
`b6fd9bd2119b1a4a36bc8e5c299f9557123c0e5f6988f28f10465abc643af3c4`.

GLM shared experts now bind the installed Flash NVFP4 checkpoint's actual dense
BF16 gate/up/down matrices, without scale sidecars. A read-only header inspection
of layers 3, 44 and 45 confirms gate/up 2048-by-4096 and down 4096-by-2048.
The adapter supplies a replicated one-expert, unit-score precision plan and
streams the 50,331,648-byte bank through the common packed materializer. A native
official-sized BF16 fixture passes binding and bounded 4,093-byte chunk transfer.

MoeProgram now owns the complete startup preparation for routed/shared execution:
exact workspace budgeting, AOT function loading, compact intermediate allocation,
expert-map upload, shared unit routing, optional FP8 quantization buffers and
binding to the existing MoE queue. Caller-owned banks and input/output remain
borrowed; no resident decoded weight duplicate is created. BF16 and dynamic FP8
Program tests each run 16 cycles with zero measured token-step allocations,
112/176 kernel submissions respectively, and balanced explicit release. Budget
refusal, partial binding cleanup and active cancellation also pass. The affected
native tests pass 20/20; targeted native compilation has no errors. These Program
tests use a fake native device to verify wiring and lifetime, not GPU arithmetic.
The earlier physical kernel results above remain separate; this addition has not
yet been exercised as a complete three-model or two-host serving runtime.

Feature development is prioritized over additional TLS/admission expansion or
blanket warning migration. The remaining functional work is connecting these
programs to complete GLM/DeepSeek decoder blocks and the shared whole-model
execution/serving path, followed by complete model execution measurements.

## Latest sealed physical qualification — 2026-08-28

The exact final30 source archive SHA-256 is
`1b951694414dbd9fc9d796eb9532580d82e0bb6a101ad289194ec58cd8d12eaa`.
On Linux it passes the warning-denied 506-task native check, 2,297/2,297 native
tests, the pinned-executable and child-control sanitizer gates, the
activation-allocation gate, and the process/worker authority ABI boundaries.
The complete RTX 5060 Ti `sm120` campaign is sealed at
`/tmp/lunaflux-final30-current-source-physical-20260828` on the validation
host. `RESULT.txt` records `outcome=passed`, `exit_status=0`, and
`terminal_stage=complete`; the independently rechecked evidence manifest
SHA-256 is
`94852d96c82390ffa427d2ce630c89086bec4f64b8395c431ed31d2fccd9a8cc`.

The campaign rebuilt worker SHA-256
`e4aa8ae78538c166fdf65ea256151f1d2b8c6b4b0bc990948c8558fd7fed6c25`,
compiled the authenticated offline-only AOT set with CUDA 13.1.115, used no
request-path JIT, and passed baseline/prefix materialization, spawned physical
execution, native listener readiness, broad BF16 qualification, non-routable
OpenAI qualification, and physical prefix reuse. Listener evidence proves the
exact admission transition `Listening/Ready -> Connected/NotReady ->
Listening/Ready` while health remains healthy; it ends with one balanced
accept/disconnect, all 32 KV pages free, and closed listener/child authority.
Broad serving ends with five balanced accepts/disconnects, two isolated
rejections, four completions plus one cancellation, and exact resource balance.
GPU inventory returned to 15/22 MiB with no compute process. This is positive
single-GPU BF16 evidence only; it is not positive I8, FP8, tensor-parallel, or
NCCL evidence.

The Phase 2/3 v3 2,300-wave fast and timer replays both pass on the same Linux
source with 4,600 requests, 4,465 completions, 135 cancellations, 16
rejections, 41 balanced accepts/disconnects, and closed queue/active/KV/pending
and child resources. The exact-final29 24-hour run also passes. Its terminal
record reports 86,400,012 milliseconds, while the worker's single terminal
line measures 86,400,002 milliseconds across 43,200 cycles and 86,400 requests.
All required wave counters are positive, accepts equal disconnects at 833, and
resources close at `queue0,active0,kv_used0,kv_free8,pending0,child_closed`.
Every entry in evidence manifest
`a49609d819e143adf1e63d99f0ee97014a58d095baaffd0f9a3c2eb43ff8788f`
rehashes successfully, stderr is empty, and the verified local archive at
`/private/tmp/lunaflux-final29-phase23-soak-24h-20260828-pass.tar.gz` has
SHA-256
`3a105bb7ef9665c86e8dabec1808ef61da838c3579bcb4929e05948e623d4fb0`.
The preceding final7 attempt has no `RESULT.txt` because
host loss removed its transient unit; its zero-byte runtime logs and immutable
artifacts remain preserved as infrastructure-interrupted evidence.

## Prior final7 physical qualification — 2026-08-28

The latest sealed portable source archive is
`bd1c9a275435ab25a0fb11ce539fde1745945bf542e6389250d4383a50c6e313`.
Its Linux phase boundary passes `moon info`, `moon fmt --check`, warning-denied
native check, and 2,065/2,065 native tests. A fresh RTX 5060 Ti `sm120`
campaign rebuilt and authenticated the child and AOT set, then passed
spawned execution, one-request serving, broad bounded BF16 serving,
caller-authorized loopback Responses, a real-timestamp benchmark measurement,
and eight-token prefix reuse. Broad serving covered concurrent requests,
active-plus-waiting saturation, cancellation, foreign-model rejection,
malformed-frame isolation, same-owner post-malformed recovery, fresh-owner
restart, drain, 22 ordered events, five balanced accepts/disconnects, two
rejections, all 32 KV pages restored, and closed listener/child authority.

The sealed evidence manifest
`56599642163a25771c6d3b58c2a2f5023ac3c0ebf52b9cc1f75979a88f8f2ae6`
verifies completely. The pass archive is retained locally at
`/private/tmp/lunaflux-physical-evidence-20260828-final7-bd1c9a275435-rerun1-pass.tar.gz`
with SHA-256
`2d1441f36929ee8bdfeeac7bc5ffdc11e9fdb06de7ee6b7c9ed350cf30ad09c6`.
GPU memory returned exactly to 15/23 MiB and no compute process remained.
OpenAI evidence remains explicitly non-routable and not production-ready.

An independent lower-level sweep on the same source passed 128 CUDA primitive
cycles, all eight BF16 kernel families, paged attention, a 128-cycle captured
complete graph, the five-case/160-launch shape matrix, and two byte-identical
approved-model runs with `1031,2185,688,2844`. Its sealed manifest SHA-256 is
`72f6a42ce7d0ae3d8f9c5c70eb8a8091a38c0026ea681cb90540fa29477e1229`;
the local archive SHA-256 is
`8bb86a6cce09fc299bc5c6461b09cd3626cac850e420015fda2ad991afa27cb0`.
The product-owned Phase 7 diagnostic rejects the heterogeneous `sm120`/`sm75`
no-peer node with missing NCCL before context, allocation, communicator, or
rank authority. That exact-source Phase 8 probe rejects I8 on `sm120` before
context or allocation authority; current software now admits exact `sm120`, so
the probe records the historical policy rather than current target support.
Their local evidence archive SHA-256 values are
`855e253214b5d9ed727811dac66a3548f541a2a261d03fac30f37428f6fcdd90`
and
`d31bed457d6fd8a08f417def68581b3ea443e31911b4eaedf729d3b424386bb8`.
These are fail-closed passes, not positive tensor-parallel/NCCL or I8 evidence.

The final7 Phase 2/3 source used the digest-pinned v3 soak policy
`ab268f305c71658b53b9f8347eb77a79d1fd4c414fc01faa0492a48e3886751a`.
Its real-worker smoke and 2,300-wave fast diagnostic pass with balanced
resources and a closed child. The two historical v2 24-hour passes remain
valid only for their frozen source. Linux testing exposed and fixed the soak
harness's remaining macOS-only `/private/tmp` fixture root by making the runner
supply an explicit approved root. Its 24-hour v3 run was launched as systemd
user unit `lunaflux-phase23-soak-v3-final7-20260828.service`, with preserved evidence at
`/home/jiaanguo/lunaflux-phase23-soak-v3-evidence-20260828-final7-bd1c9a275435`.
Host loss removed the transient unit before any `RESULT.txt`; this is
infrastructure-interrupted evidence, not a pass.
Exact identities and the preserved non-promoted final5 static-gate failure are
recorded in
[PHYSICAL_VALIDATION_2026-08-27.md](PHYSICAL_VALIDATION_2026-08-27.md).

## Post-final7 qualification carried by final30 — 2026-08-28

The final30 source includes the post-final7 additions below and has been rebuilt
on the NVIDIA host. The full Linux suite, exact sealed-final30 physical
campaign, and exact-final29 v3 soak pass. Individual claims below stay
scoped: this does not promote public networking, comparative benchmarks,
Mistral/FP8 hardware support, OCI/SBOM, or general release readiness.

Final30 also closes several local authority gaps without manufacturing external
approval. Offline comparison verification now independently pins the live
trial driver and sanitizes its replay environment; fused-kernel candidate
evidence binds ordered trials, numerical tolerances, per-shape aggregates, and
static-resource audit identity. Phase 6 adds a versioned non-authoritative
`legacy-doctor --json`; Phase 7 rejects noncanonical failure-reason/rank pairs; Phase 8
adds an authenticated fixture-only Mistral correctness corpus; and Phase 6/9
release assembly binds the exact first-party Apache-2.0 license and canonical
product-license inventory. The sibling LunaNexa repository now has an opaque
authority-only adapter projection that retains the exact launch-manifest
identity. Positive performance, supported quantization hardware, homogeneous
NCCL, approved OCI/SBOM/provenance, actual deployment-issued verifier
key/policy and signed
receipt set, an approved promotion-catalog entry, TLS/auth, and reviewer
acceptance remain open.

## Current software-validated source state — 2026-08-29

The exact current tree passes the complete 2,448/2,448 native matrix, aggregate
dependency/debt boundary, and focused authority/allocation/performance gates
locally. A non-overwriting portable handoff is sealed only after these checks;
its digest belongs to the external handoff record so sealing does not create a
self-referential source hash. Everything below this heading remains
source/software state only: it is not part of final30 physical evidence, does
not change the exact-source scope of the completed final29 soak, and makes no
new NVIDIA, performance, sanitizer, or release claim until a named post-soak
current-source campaign verifies.

Performance work is now tracked independently in
[PERFORMANCE_ROADMAP.md](PERFORMANCE_ROADMAP.md). The current dense
Qwen3-0.6B BF16 concurrency-one, two-token measurement is about 25.46 ms versus
21.60 ms for SGLang and 11.63 ms for vLLM on the same comparison host. This is
a narrow decode-oriented datapoint, not a throughput claim. The actionable
gaps are the roughly 311 launches in the unfused Qwen step, a single
hand-written projection strategy, one-warp serial paged attention, production
fusion not yet selected by that Qwen deployment, and maximum-envelope graph
capture instead of shape buckets. Current feature work therefore prioritizes
shape-aware projection, phase-specific tiled attention, real graph fusion, and
execution-graph buckets; release-evidence work is not on this hot-path loop.

Advanced-family work after that sealed baseline is software-only. DeepSeek V4
five-profile, GLM 5.3 Full/Flash, and MiniMax H3 FL2VA/Ref2VA have exact config,
plan, weight, numeric, host/device-upload, reference, and partial inert AOT
evidence behind workload-specific packages. Semantic capability availability
is explicitly not called executable readiness. H3 now additionally has prepared
encoder/denoiser/decoder request composition and offline source export, described
in the [current code inventory](MINIMAX_H3_CODE_INTEGRATION_2026-09-26.md).
This has not been physically qualified as a complete model or deployed as a
production serving path. The family-level stages and gaps are tracked in
[MODEL_FAMILY_SUPPORT.md](MODEL_FAMILY_SUPPORT.md).

Subsequent local debt hardening removes every compiler-reported unnecessary
MoonBit annotation and makes warning 73 a permanent aggregate boundary. It also
centralizes exact `DeviceTarget`/numeric-capability matching for FP8 and I8,
replaces a stale caller-constructible external-protocol report with the private
fail-closed instance-policy join, and separates authenticated bootstrap-source
encoding from untrusted decode validation without changing wire bytes. The
descriptor admission now retains its inert worker-plan proof privately instead
of exporting an unused accessor, and two unused qualification-only runtime
owner methods are no longer public. Authority validators now describe the
actual private credential-policy join and the opaque snapshot-pinned worker
executable graph. A further source-only pass removes the last public raw-Bytes
device loader so device materialization is exclusively ApprovedRoot-backed and
digest-authenticated, removes a dead scheduler retirement counter while adding
generation-exhaustion coverage, and replaces optimizer-dependent Result/defer
cleanup envelopes in request preparation and incremental output with direct
fail-cleanup paths, and removes the same optimizer-dependent defer envelope from
blocking inherited-frame I/O. Native ordered-executor construction now retains every
returned event, graph, and graph-exec handle plus its parent leases until
destruction actually succeeds, including combined create/destroy failures. The
worker-service shutdown and close paths now retain explicit retryable lifecycle
state through direct failure catches, and the root-bound worker-process package
keeps its recovery startup contract private. The fused physical-campaign
source composition, admission, benchmark-qualification, cleanup, and hostile transaction additions
plus generic LunaTile translation, graph telemetry, and fused approval-boundary
work pass their focused warning-denied native and hostile/static boundaries.
The generic serial oracle now also anchors a deterministic inert
parallel SIMT candidate with exact block/warp/lane mapping, a uniform staged
pipeline, lifetime-planned shared storage, tensor-core eligibility recording,
and conservative cross-instruction global-write disjointness. It has no
compiler, device, manifest, runtime, or promotion authority. An isolated
exporter and campaign runner now compose deterministic double compilation,
serial-oracle comparison, memcheck/racecheck, and an immutable evidence seal,
but the campaign has not run on NVIDIA hardware.

The same LunaTile graph now feeds one narrowly typed exact-`sm120` BF16
row-major `m16n16k16` WMMA source candidate with F32 accumulation/output,
32-byte global alignment, one warp per tile, and an identity epilogue. Its
test-only probe/campaign binds deterministic CUBIN pairs, independent CPU and
serial-CUDA numerics, dual-disassembler SASS instruction counts, resource
bounds, memcheck/racecheck/initcheck, cleanup balance, and a non-circular
three-file seal. A separate release package admits the exact positional result
and critical manifest entries, then exposes an opaque evidence-aware lowering
join that retains the serial and SIMT fallback identities. The existing
evidence-free tensor-core requirement still rejects. Candidate, evidence, and
qualified-wrapper types remain manifest-ineligible and promotion-ineligible.
All source, fake-tool, hostile-substitution, and admission gates pass locally;
no NVIDIA execution, physical SASS/numeric result, performance win, production
consumer, or promotion is claimed for either new route.
Rooted single-worker graph telemetry now crosses the child boundary in a fixed
80-byte checksummed sidecar: completion is unpublished until the matching
sequence/path/shape/counters validate, replacement counters aggregate with
saturation, and native/OpenAI instance metrics consume monotonic deltas without
steady-path allocation. Tensor-parallel transport explicitly publishes
telemetry absence rather than a fabricated miss. A bounded opaque `bench`
command uses one digest-pinned live deployment admission for a fixed
token-0/two-output greedy request, records real monotonic events, and drains the
owner before emitting evidence with comparison and promotion authority absent.
Focused hostile/sanitizer and allocation suites plus the complete aggregate
dependency/debt sweep pass. These are
local source and architecture results; they do not extend final30 physical
evidence.

The current source retains qualification-only paths for positioned full-QKV
paged-KV write, read-only paged attention, and residual-plus-RMSNorm. It now
also has production-fast-path V2 admission, resource preparation, and execution:
legacy Llama and generic numeric-BF16/Mistral worker APIs can replace the exact
residual/RMSNorm span with one launch and the QKV/RoPE/attention span with a
fused writer plus read-only attention. Startup authenticates artifacts, plan
adjacency, fallback identities, raw-pointer ABIs, and live-device identity once;
the token step performs no canary, hashing, filesystem work, diagnostic
host/device transfer, or qualification scan. Residual V2 preserves the
authenticated graph policy; its diagnostic qualification ABI remains
eager-only. Missing optional artifacts retain
standalone execution. Descriptor-pinned optional artifacts now cross the exact
runtime descriptor/bootstrap into deployed children for Llama and numeric BF16;
the QKV composite binds only after child-side live-device identity
authentication. None of these V2 routes has current-source physical CUDA,
sanitizer, performance, or promotion qualification.

The lower production-V2 and literal spawned `device-greedy`/
`device-greedy-fused-v2` harnesses now reach those production boundaries and
compare a fixed eight-byte device result with an independent host full-logits
referee. Their software/static gates pass; they have not run on NVIDIA
hardware.

Kernel approval verification is now reachable from a startup-owned bounded
FD-7 key capability with no public arbitrary-key constructor. The parent
consumes, closes, and wipes the deployment key before child activation, then
creates a fresh one-shot attestation bound to the exact admitted manifest,
approved source, pinned launch identity, generation, and ordinal. The child
never receives the deployment key. Stale, substituted, absent, and replayed
attestations fail closed before `Ready`; absent external approval leaves
baseline inference standalone.

Paged profile capture now binds exact launch identity, mixed row/cache/
position/page geometry, a diagnostic-only page-table trace digest, and bounded
sorted operation counters. The LunaTile optimizer-promotion owner remains
inert and non-bindable: no generic fixture operation can become a real catalog
operation until an authenticated specializer maps its exact operands, shape,
and numerical contract, in addition to sealed paired evidence and external
approval.

The runtime now serves bounded bodyless `GET /healthz`, `GET /readyz`, and
`GET /metrics` observation. OpenAI Responses serves them on its inference
origin and creates no second control listener; native-framed mode retains a
separate loopback-only HTTP control listener. A fixed inherited Unix-stream
capability on descriptor 5 gives the opaque one-argument CLI one local,
allocation-free drain command whose accepted/idempotent/closed responses and
listener-first cleanup are tied to the singular runtime owner. A distinct
descriptor-6 Unix-stream capability now admits one bounded read-once inference
credential, closes the channel, and wipes its source scratch before runtime
resource construction. Instance-policy v3 binds loopback plaintext or exact
wildcard-only `private_network_plaintext` OpenAI Responses plus its credential
ceiling. LunaNexa parses the same-origin publication and maps a wildcard origin
to the inspected private container address before bounded health/readiness
probes and invocation. Native v1/v2 reject a credential and OpenAI v2 remains
fail-closed. Focused HTTP
tests, a real local loopback hostile-client test, the inherited-drain native
ABI, ASan/UBSan, allocation, CLI-activation, and lifecycle suites pass. This
does not implement TLS, public routing, external authorization, or generation
fencing; those deployment gates remain open and no public HTTP drain route
exists. The copied opaque API-auth policy, verifier aliases, and presented
Bearer/header/body parser storage are deterministically overwritten on owner
close; startup rejection paths wipe before abandoning ownership. Constant-time
credential comparison scales with the configured credential length rather than
the larger accepted-input ceiling.
Authenticated OpenAI JSON now lands directly in fixed typed handoff storage.
The same receipt remains bound from byte-one capture through authentication,
parsing, template expansion, semantic validation, and admission, with no JSON
tree, prompt `String`/`Bytes` copy, canonical request-frame render/checksum/
reparse, `GenerateRequest`, or `TextInput` intermediary.

Two locally proven qualification harnesses now cover broader owner lifetimes.
The context-churn campaign cycles fresh child owners across bounded context
ceilings, while the actual-context campaign sends independently fixed 8-token
and 9-token requests through the production spawned owner and checks exact
plans, KV state, terminal events, and cleanup. Their focused warning-denied
checks and tests pass. Neither campaign has run on physical CUDA, so neither is
long-context capacity, leak, soak, or performance promotion evidence.

The benchmark boundary now includes a real OpenAI Responses SSE observer, an
offline admission tool, and a deterministic external-process campaign owner
for exactly 81 observations: three engines, nine profiles, and three trials in
fixed Latin-square order. It accepts only digest-pinned replay, trial-driver,
correctness, live-engine-identity, and process-supervisor tools; keeps
credentials on inherited descriptors; seals raw captures, identity evidence,
correctness records, and complete process-cleanup receipts; and hands the set
back to the existing comparison authority. A driver cannot self-assert an
engine: the separate live verifier must bind the declared revision, image,
configuration, and executable digest. Hostile fake-engine, wrong-identity,
malformed-framing, failed-correctness, timeout, substitution, and no-overwrite
gates pass locally. The handoff deliberately reports
`comparison_authority=none` and no physical measurement claim. No approved
live vLLM/SGLang identities, external correctness/identity policy, live
81-trial capture set, or comparative result has been produced.

Phase 8 now has software-complete runtime routes for the bounded Mistral and
reusable FP8 additions. Strict `MistralForCausalLM` configuration and
content-bound metadata reuse the existing dense 21-operation stateless and
paged plan; exact BF16 weight binding/materialization plus strict regular and
materialized descriptors reuse the common BF16 worker/service owner without a
scheduler or KV branch. Its family, weight, descriptor, worker-wire, and
socket-backed runtime gates pass. A digest-authenticated fixture-only corpus now
proves exact dense-plan and 21-weight equivalence and replays logits, argmax,
and continuations without claiming production authority. A production-approved
artifact and correctness corpus, physical numerics, accuracy, memory,
performance, and readiness remain open.

The FP8 numeric owner admits only canonical `F8_E4M3`, rejects both E4M3FN NaN
codes in bounded chunks, validates positive finite scalar-F32 scales, and
performs two-pass authenticated final-arena transfer. The complete dense-Llama
W8A8 builder, exact file inspection, v2 launch ABI, deterministic CUDA 12 AOT
source, and closed artifact/symbol binder remain byte-compatible with the
pinned v1 route. Exact-shape v2 stays inert and single-shot. The additive
reusable PagedV4 v3 production route reconstructs a digest-pinned schema-v5
manifest, accepts only opaque externally approved CUBIN release authority,
re-admits model/plan/profile/target/raw ABI/workspace/KV limits in the child,
prepares the mixed graph executor, validates unique sentinel-initialized scale
cells before publication, and joins the existing rooted service through
regular or materialized descriptor and one-argument CLI paths. Hostile
substitution, cleanup/restart, materialized-route, and socket-backed software
gates pass. Exact `sm120` is now admitted across device, descriptor, bootstrap,
and CUDA-v2 software boundaries alongside `sm89`/`sm90`; adjacent targets fail
closed. No current-source physical FP8 numerical campaign has qualified this
route; sanitizer/leak balance, soak, accuracy, memory, performance, and release
promotion remain open.

Phase 5 now has a separate offline/startup profile-priority admission package.
It binds sorted stateless full-context observations, workload and profiler
digests, model plan, and device target into one canonical digest, then deterministically
selects the operation with the greatest attributed self time. This supplies a
bounded input to later optimization work only; it neither captures a profile
nor proves kernel correctness, benchmark improvement, or release promotion.
A disjoint paged path derives target, selected profile, and complete device-KV
layout from an admitted paged launch set, binds exact mixed-row/cache/position/
page geometry plus bounded sorted operation counters, and rejects uncovered
operations or substituted authority. Its page-table trace digest is explicitly
an unauthoritative raw-trace claim.

The following exact-current-source paragraphs are retained as historical
qualification records for earlier snapshots.

The next exact-source physical rerun was locally staged but had not started
while the mandatory Phase 2/3 soak owned the NVIDIA host; that soak has now
passed. The current-source runner executes the existing qualification-only
context-churn and actual 8-token/9-token modes with the authenticated launch
and freshly built worker, then requires canonical evidence digests and final
GPU/process balance. A
generic filesystem-only evidence helper now owns deterministic manifests and
read-only sealing without acquiring schema, admission, or promotion authority.
OCI verification additionally rejects regular files with hard-link aliases;
crash-recovery verification uses independent copies for the OCI subtree so the
same invariant is preserved. The external-process comparison campaign now
seals all 326 canonical invocations beside their 326 supervisor receipts, so
offline replay independently checks exact argument hashes, timeout/grace, and
credential-descriptor scope. The strict native check, all 2,253 native tests,
the two context source gates, immutable-evidence hostile gate, OCI and atomic
recovery hostile gates, current-source runner gate, and the exhaustive fake
81-trial comparison gate pass locally. None of these local results is physical
context evidence, a live cross-engine measurement, or an OCI promotion claim.

The latest exact-current-source archive is
`847c8493a4d43faf8517969be8780eace00380b69fa9c6b7894d3fd9e45b36ac`.
It passes 2,038/2,038 native tests on Linux and a fresh RTX 5060 Ti `sm120`
campaign that rebuilds the child, rematerializes both authenticated launches,
and passes spawned execution, native serving, caller-authorized OpenAI
Responses loopback qualification, a real-timestamp LunaFlux benchmark trial,
and physical eight-token prefix reuse. All 209 sealed physical files, source
stability, release digests, empty runtime stderr, and final GPU/process balance
were independently verified after download. Exact hashes and the preserved
pre-fix `TIME_WAIT` failure are recorded in
[PHYSICAL_VALIDATION_2026-08-27.md](PHYSICAL_VALIDATION_2026-08-27.md).

All TCP listener owners now share one `SO_REUSEADDR` bind policy so an
authenticated server-side close cannot block same-address restart in
`TIME_WAIT`. `SO_REUSEPORT` is never enabled and a concurrently live exact bind
still fails. OpenAI qualification remains explicitly non-routable: generic
production readiness stays false, TLS and a health endpoint are unclaimed, and
production v2 admission continues to fail closed until the missing deployment
owners and approvals exist.

The portable source snapshot with SHA-256
`99887e5f4687889fd30f3927508a4adc49ff0b1f052117f87bca9b8c069d9e83`
passes the Linux warning-denied native check (434 tasks) and 2,010/2,010 native
tests. On the RTX 5060 Ti (`sm120`), the CUDA primitive, eight BF16 kernel
families, paged attention, capture-required complete graph, five-case shape
matrix, and two byte-identical approved-model runs pass. Spawned execution,
the production native listener fixture, concurrent framed slow/fast-client
progress, the 10,000-request service balance campaign, the soak diagnostic,
and ten sanitizer/native-ABI gates also pass. The approved-model continuation
is `1031,2185,688,2844`, selected-logit maximum absolute error is
`0.0005459413`, and final GPU process and memory balance is exact.

The scheduler now snapshots exact request-local cached-token usage before slot
recycling and carries it through worker and online-session terminal
publication. The bounded spawned-prefix validator checks independent expected
tokens and exact cached-token evidence without inferring from global telemetry.
The startup-fixed native framed connection pool owns one service and semantic
stream across independent incremental clients; its physical Linux campaign
proves that a one-byte-at-a-time client cannot block a complete fast request.
The authenticated runtime owner now selects that pool, or the corresponding
OpenAI pool, whenever the admitted preparation/scheduler request-slot capacity
is greater than one. Exact capacity one retains the prior singleton behavior.
Both production routes share readiness, drain, cold-start, observation metrics,
and deterministic cleanup with their fixed-capacity owner; no per-request pool
construction was added.
The canonical benchmark runner now collects checked lifecycle and timing
records into digest-bound evidence without adding process, network, or device
authority.

This is supported-BF16 correctness and bounded-serving evidence, not whole-plan
promotion. Positive I8 remains open even though current software admission now
includes exact `sm120` with `sm89` and `sm90`.
The tensor-parallel software route now binds a nonzero canonical group-template
digest through descriptor, materialized preflight, child re-admission, and the
live generation/rendezvous group digest; fake multi-rank, substitution,
restart, and cleanup gates pass. Positive tensor-parallel/NCCL remains open
pending a homogeneous peer-capable node with NCCL. Physical prefix reuse and a
single LunaFlux listener measurement now pass for the pinned tiny BF16
workload. The complete LunaFlux/vLLM/SGLang
comparison remains open: the current adapter deliberately brands Chat
Completions SSE and cannot approve the production Responses protocol, the host
has no pinned baseline images or container runtime, and the nine-profile,
three-trial matrix has not run. The sealed final30 campaign rebuilt its child;
the older paragraph above remains historical evidence for the preceding
archive.

## Completed foundation

- Phase 0 product, architecture, lifecycle, benchmark, and debt contracts.
- Checked engine and request lifecycle vocabulary.
- Focused validated configuration records and deterministic startup decisions.
- A byte-bounded configuration JSON adapter that rejects duplicate, unknown,
  missing, mistyped, and out-of-policy fields before constructing focused
  records.
- A bounded extracted byte-BPE tokenizer foundation with exact byte decoding,
  explicit special-token policy, limits, and deterministic merge ranking.
- A reusable startup-preallocated `LunaTokenizerWorker`. Its opaque
  generation-bound input writer copies one scalar byte into fixed backing per
  outer preparation unit, then transfers authority to work that advances byte
  BPE by a fixed operation budget, uses resumable CSR merge lookup, preserves
  exact ranked group-merge semantics, and copies results only in budget-capped
  chunks. The proportional `Bytes` encoder is a compatibility facade over that
  same writer and is not a reactor quantum. Frozen-reference and
  positive-controlled release-C gates cover byte-work equivalence and the
  allocation-free warm path. A fixed-lane Luna request-preparation pool
  includes the tokenizer input backing in checked aggregate byte admission and
  drives input copy, tokenization, token transfer, and incremental-output setup
  from one central FIFO quantum owner. Terminal tokenizer errors clean their
  already-stale work alias before the pool releases remaining lane authority.
  Canonical raw-frame scanning and direct preparation now share that FIFO work
  owner: the first accepted byte captures one receipt clock, frame/scalar/input
  and semantic work stays budgeted in fixed lane storage, and the private
  framed View is released before Ready. Object-form materialization remains a
  compatibility path. A native one-shot loopback/TCP shell now binds one
  listener, accepts exactly one connection, closes the listener, preserves
  partial framed tails, and drives that coordinator through private fixed
  input and dual-view output storage. A typed awaiting-input result alone
  authorizes continuation reads for an incomplete receipt; queued terminal
  work pinned behind an active ticket remains semantic progress. Exact
  receipt/output-stall remaining
  time bounds each body read and event write; positive short writes confirm
  the authenticated Offer before Flight release, while zero/unknown writes,
  timeout, cancellation, rejection, and disconnect drain retained authority.
  Usage ACK and lower retirement remain explicit off-reactor maintenance.
  The endpoint is thread-confined and requires exclusive serialized handoff;
  its activity flags are not cross-thread synchronization. It is evidence for
  one connection, not a standalone reactor, reusable listener, multi-client
  server, TLS/HTTP adapter, or fleet ingress. Async runtime and socket
  allocation remain outside the narrower warmed payload/helper allocation
  proof.
- A reusable native-framed `LunaOnlineTcpServer` now retains one listener and
  one `LunaOnlineFramedService` across sequential connections. Its compatibility
  bind accepts one request per connection; its pipeline bind admits bounded
  coalesced or later frames into fixed request lanes on the same connection.
  The compatibility bind preserves fragmented first-frame tails and discards a
  coalesced second frame; the pipeline bind retains and admits that exact tail.
  Both confirm bounded partial writes before ACK/release, close the listener
  first on drain, and record finite instance metrics/log events before
  observation ACK. Real worker-process loopback evidence covers accept timeout reuse,
  rejection without response bytes, normal completion, cancellation, exact
  failure recovery ordering, post-restart listener reuse, disconnect at
  FailureUsage, and stale snapshot/resource cleanup. This is serialized native
  framed ingress, not HTTP/OpenAI, TLS, concurrent clients, or a fleet server;
  async runtime/socket allocation remains outside its warmed synchronous helper
  proof.
- The fixed-lane service balance gate now completes 10,000 monotonically
  identified requests in 1,000 waves on one rooted worker service, including
  natural completion, cancellation, queued expiry, terminal-ring
  backpressure, ten empty-worker recoveries, and final cooperative
  shutdown/reap. It reconciles request IDs, physical and pending work,
  publication rings, block tables, and KV ownership. Its approved-root fixture
  is now supplied as a canonical per-run temporary directory rather than the
  macOS-only `/private/tmp`; the same 10,000-request proof passes on Linux.
  This is finite release
  evidence separate from the 24-hour soak. The frozen v1 low-rate soak aborted
  on a harness overconstraint that required every balanced wave to enter the
  two-row histogram bucket; service, request, and KV balances were intact at
  the observed boundary. The v2 policy records two-live, batched,
  backpressured, scheduled-cancel, attempted-cancel, accepted-cancel, and
  actually cancelled waves cumulatively. It permits balanced non-batched waves
  and treats an exact typed coordinator rejection as a non-mutating missed
  cancellation. The digest-pinned v2 campaign started under launchd on
  2026-08-24 as `lunaflux.phase23.soak.v2.20260824T125416Z` and produced two
  independent `24h-pass` records. Both measured 86,400,002 milliseconds,
  completed 43,200 waves and 86,400 requests, balanced 833 accepts with 833
  disconnects, restored all eight KV pages, and closed with zero queued,
  active, used-KV, and pending resources plus `child_closed`; stderr is empty.
  The service, worker, source, runner, stdout, and stderr SHA-256 values are
  preserved with the campaign evidence. The exact keepalive label was removed
  after the second pass so a third run could not append another record.
  A separate immutable attempt at
  `/private/tmp/lunaflux-phase23-soak-launchd.VwfUIV` exited 134 after 4,204
  seconds. Its service hash matches its manifest, but the binary
  self-identifies as the already rejected v1 soak policy; the failure is the
  known v1 harness false negative and is not runtime promotion evidence.
- A separate serialized `LunaOnlineOpenAIServer` composes the bounded HTTP/1,
  bearer-authentication, OpenAI JSON translation, trusted framed-service, and
  semantic SSE engines without materializing an intermediate request object.
  It supports Responses and Chat Completions over sequential one-request
  connections, confirms every partial socket write before semantic delivery,
  records observation telemetry before ACK, and preserves request deadlines
  across HTTP, JSON, framed admission, and output phases. Real worker-process
  loopback evidence covers wrong credentials without an Admission, fragmented
  HTTP/JSON input, both routes, terminal Usage/Completion ordering, bounded
  partial output, rejection/error reuse, receipt expiry, and listener reuse.
  This remains a native, serialized HTTP/1.1 foundation: it is not TLS,
  keep-alive/pipelining, concurrent-client arbitration, or a fleet endpoint;
  async runtime/socket allocation is outside its warmed synchronous helper
  proof.
- A bounded `tokenizer.json` adapter for the selected raw ByteLevel-BPE
  contract and the exact closed SentencePiece-derived BPE profile shipped with
  the pinned upstream model, with duplicate-key rejection and explicit
  rejection of unsupported tokenizer semantics.
- A digest-pinned synthetic compatibility tokenizer for the selected 3,000-row
  reference model. It proves the supported ByteLevel-BPE path from `Luna*c` to
  exact token IDs, independently recorded logits, the greedy token, and a
  four-token continuation. This remains functional evidence for the narrower
  ByteLevel profile rather than trained-tokenizer quality evidence.
- Validated dense Llama-style BF16 dimensions and content identities, with the
  plan identity minted only after `ModelPlan` validates and owns the complete
  graph, execution mode, limits, KV/workspace semantics, capabilities, final
  output, and numeric-schema digest. Constructors accept no caller plan digest
  or model identity. Llama metadata is content-only, and `ModelPlan` is the sole
  authority that can mint an authenticated plan identity.
- A bounded selected-model architecture JSON adapter that rejects remote code,
  tied embeddings, and unsupported dense-Llama variants.
- Bounded decoded safetensors metadata validation, including checked tensor
  sizes, ranges, duplicates, overlaps, precision, and materialization budget.
- A bounded safetensors byte reader with checked little-endian header lengths,
  strict JSON schema, exact integer conversion, duplicate-key rejection, and
  complete payload accounting.
- An immutable architecture-neutral operation plan and dense Llama family
  builder with explicit shape, workspace, RoPE, activation, tensor-reference,
  and semantic kernel-capability contracts.
- Phase-8 numeric admission foundations now define a mandatory canonical model
  numeric schema with separate tensor-storage and operation-execution
  contracts. Operation schema v2 distinguishes BF16 graph activation input
  from finite-E4M3 internal activation compute and digest-binds the complete
  dynamic per-tensor F32 scaling convention. A closed device capability policy
  admits only exact compute pairs 8.9, 9.0, and 12.0. The FP8 kernel
  manifest cross-authenticates the exact model schema, observed target facts,
  AOT catalog/launch entry points, one admitted artifact bundle, and each FP8
  weight plus scalar-F32 scale identity. A final pure startup join rejects
  substituted or incomplete evidence before publication. The authenticated
  numeric-weight owner additionally admits exact `F8_E4M3`, rejects both
  E4M3FN NaN codes and invalid scalar-F32 scales in a bounded first pass, and
  copies only the reauthenticated snapshot into the final arena. The
  all-or-nothing dense-Llama FP8 builder binds every eligible projection and
  scalar scale into full-context or paged plan identity; the family file
  facade preserves typed plan/file failures and rejects mode/batch replay. A
  staged dynamic-scale ABI further joins the plan and manifest to an exact
  PagedV4 raw launch contract, complete dimensions and operand order, an exact
  four- or eight-byte Workspace, schema-derived scale cells, and a fenced
  lease lifecycle. Separate v2 offline lowering emits deterministic finite-
  E4M3 CUDA 12 source for simple projections and compound gated MLP, then a
  non-circular binder joins the candidate to exact contract, CUBIN receipt,
  artifact, module, and symbol authority. Exact-shape v2 remains inert.
  Reusable PagedV4 v3 now has schema-v5 manifest reconstruction, opaque
  external release admission, child-side re-admission, exact mixed-graph
  executor preparation, unique sentinel-checked scale evidence, and
  regular/materialized descriptor, worker, service, and CLI routing. Focused
  software gates pass, including exact `sm120` admission and adjacent-target
  rejection. No current-source physical FP8 numerical correctness,
  sanitizer/leak balance,
  soak, memory improvement, benchmark, or readiness claim has been made, so
  Phase 8 remains open.
- Symmetric-I8 weight-only v1 now freezes rank-two `[out_channels,
  in_channels]` storage with one owned rank-one plain-F32 per-output scale,
  implicit zero only, signed codes `[-127, 127]` with `-128` rejected, BF16
  activation boundaries, F32 accumulation, and BF16 output. Synthetic plan
  fixtures make multi-weight selection atomic. A distinct closed
  8.9/9.0/12.0 device feature exists solely as software target admission.
  Catalog v4
  exact-digest selection now binds each operation to one content-addressed AOT
  family and catalog-owned entry point. A pure numeric layout places the full
  parameter/scale/zero-point/codebook table, and an opaque static plan joins
  that exact layout schema, full model identity, selected entry point, and
  operation execution digest. Legacy v1/v3 reject I8, while the legacy device
  execution-profile bridge rejects v4. A distinct opaque catalog-v4 full-paged-
  graph launch boundary now derives the complete operand sequence from the
  model plan, places each schema-owned scale immediately after its weight, and
  authenticates the numeric schema digest, per-operation execution digest, and
  exact catalog entry point. The production dense-Llama builder now emits this
  policy atomically for every projection and the language-model head. A strict
  authenticated numeric-weight file loader binds the model, plan, schema,
  artifact, tensor layout, ranges, and per-output scales before allocation,
  reauthenticates the same immutable snapshot, streams directly into the final
  device layout, and returns retryable cleanup ownership on failure. Manifest
  schema/catalog v4 and bootstrap v3 retain the exact admitted memory plan,
  artifact bundle, launch sequence, and numeric authority through a serialized
  I8 executor. Runtime descriptor v2 and common worker bootstrap recipe v6 bind
  those inputs to an independently admitted instance policy, tokenizer,
  scheduler blueprint, worker executable, process limits, roots, and restart
  policy. Digest-authenticated launch schema v2 selects that closed recipe in
  the existing one-argument native command, with no legacy-loader fallback,
  and converges on the same framed service, listener, drain, restart, and
  cleanup owner. The child rebuilds the plan, re-observes the device numeric
  capability, reauthenticates weights, and publishes readiness only after the
  executor is complete. Host and fake-seam validation does not establish
  physical-CUDA numerical correctness, sanitizer/leak balance, device soak,
  accuracy, memory improvement, performance, hardware support, or production
  readiness, so Phase 8 remains open.
- Exact dense-Llama weight-manifest binding that validates safetensors names,
  shapes, completeness, and plan tensor-reference order before allocation.
- A single selected-Llama admission boundary that proves configuration,
  safetensors metadata, semantic plan, and exact tensor vocabulary agree before
  materialization.
- Bounded reference-bundle file admission for three explicit regular files,
  with per-file and aggregate pre-allocation limits, exact-snapshot SHA-256,
  tokenizer/model vocabulary agreement, payload-safe errors, and deterministic
  handle closure. Whole-file ownership is explicitly reference-only.
- Two-pass bounded weight materialization with identity preservation, complete
  preflight validation, zero-copy tensor visitation, exact complete-file
  source offsets, and an explicit host-copy reference representation.
- Deterministic aligned single-arena device-weight layouts plus a synchronous,
  bounded approved-root loader. Component-wise no-follow traversal pins a
  strict relative descendant; complete digest, metadata, and dense-Llama
  vocabulary admission precede device allocation. Inspection privately retains
  that locator, and later loading completely re-admits it before allocation.
  Transfer reuses fixed host storage, copies source-ordered chunks into final
  device regions, re-hashes exact copied bytes, rechecks the same-handle stamp,
  and retains no model-sized snapshot. A terminal source-file close failure
  closes any otherwise-ready allocation before publication. If that allocation
  close also fails, opaque cleanup authority is returned at `SourceClose` with
  idempotent retry rather than losing the partial owner.
- Deterministic Float reference kernels for dense-Llama embedding, RMSNorm,
  projections, split-half RoPE, causal grouped-query attention, residuals,
  SiLU-gated MLP, and the language-model head.
- An architecture-neutral offline plan interpreter that prepares immutable BF16
  weights once, executes validated operation plans, and performs bounded greedy
  generation. Pinned tiny-model fixtures cover non-degenerate attention and MLP
  paths with independently derived logits.
- A substantive `lunaflux legacy-config-plan` compatibility command that authenticates a bounded model
  configuration under an approved root, builds the exact paged-Llama semantic
  plan, and explains shapes, operation count, workspace, required kernel
  capabilities, and KV page/capacity decisions. It closes filesystem authority
  before publishing. The model-content digest is explicitly caller-declared,
  not weight-verified, and kernel-manifest resolution remains open.
- An exact Apache-2.0 upstream tiny BF16 Llama artifact pinned by immutable
  revision and SHA-256. End-to-end tests match 33 independently generated
  sampled logits, three argmax tokens, and two four-token greedy continuations.
- A compatible Apache-2.0 external ByteLevel-BPE corpus derived from pinned
  Hugging Face Tokenizers sources. Five independently generated cases cover
  ranked merges, spaces, multibyte Unicode, a literal zero byte, newlines, and
  byte-exact decoding.
- Bounded deterministic greedy sampling with stable lowest-token tie breaking,
  plus temperature/top-k/top-p stochastic sampling over fixed scratch storage.
  The stochastic path rejects non-finite logits before advancing its specified
  xorshift64 stream and preserves deterministic lowest-token tie ordering. The
  native and OpenAI paths now carry canonical temperature, top-k, top-p, seed,
  stop-token, and stop-string inputs through bounded parsing, framed transport,
  scheduler suppression, incremental UTF-8 matching, and device/fake-seam
  selection. Physical-CUDA numerics and production performance remain open.
- A private dynamically loaded CUDA Driver/cuBLASLt ABI with opaque handles,
  explicit retryable resource release, parent-child ownership, and a semantic
  public device inventory. Unsupported hosts report a typed unavailability
  reason without attempting device allocation. Opened contexts pin their
  ordinal and discovered capability.
- Checked offset host/device transfers and reusable row-major BF16 cuBLASLt
  GEMM plans with FP32 accumulation, explicit caller-owned storage, retryable
  descriptor cleanup, and injected-dispatch ownership/failure probes.
- Atomic native operation guards that interlock use, child construction, and
  explicit close across contexts and every owned child resource. Phase 1 GEMM
  synchronizes before returning so borrowed operands cannot outlive device
  work.
- Private bounded CUDA module and function loading plus a synchronous AOT
  launch seam. Launch dimensions, argument count, allocation ownership,
  offsets, byte counts, alignments, context identity, and close ordering are
  checked before or around the driver call.
- An immutable exact startup kernel catalog that selects by model operation,
  shape, capability, CUDA target, and workspace, using either the narrow
  implicit vendor BF16 GEMM semantic or a content-addressed AOT artifact.
  Missing, incompatible, or ambiguous support fails closed without JIT or
  fallback. Content-addressed bindings distinguish a module digest, semantic
  kernel family, and profile-specific entry-point identity. The vendor GEMM
  contract is limited to operation layouts representable by its narrow ABI.
- An immutable static device plan that validates the complete canonical numeric
  tensor layout before resolving operation inputs. It retains the exact model
  numeric-schema digest and per-operation execution digests. Legacy catalogs
  preserve content-only plan reuse; v4 requires full model identity and an
  exact catalog-owned AOT entry point before inert plan publication.
- A deterministic exact BF16 activation/workspace memory plan with liveness-
  checked slot reuse, retained terminal output, bounded aligned arena size,
  one startup allocation, and explicit close.
- Exact stateless device execution profiles for `FullPrefill` and
  `FullRecompute`. They precompute token-staging, weight, activation, workspace,
  operation, complete terminal-output, and final-row views for an exact batch
  and sequence shape without claiming cached state.
- Bounded AOT launch-contract admission for every exact profile and AOT-backed
  operation. Contracts pin exact dimensions, ordered semantic operand roles,
  byte/alignment claims, workspace, family, and entry point; they cannot carry
  arbitrary payloads, paths, or a JIT channel.
- Content-addressed artifact admission that verifies and owns each required
  module once per digest and maps all required stable entry points to bounded
  CUDA function symbols. Missing, duplicate, unreferenced, or digest-mismatched
  declarations fail before module import.
- Bounded, versioned AOT manifest and module-file admission under an approved
  read-only root. It pins the exact model, target, catalog, module digests,
  stable entry points, and function symbols; rejects unsafe path forms and
  unsupported file types; proves aggregate bounds before module-sized reads;
  and admits immutable snapshots without an extra full copy.
- Opaque approved-filesystem capabilities backed by a private native ABI.
  Component-relative `openat`, no-follow directory traversal, final regular-
  file checks, pinned positional reads, same-handle stamps, atomic operation/
  close exclusion, deterministic close, and payload-safe failures are
  implemented. A startup-only bounded snapshot holds one lease across its two
  stamps, exact positional read, and trailing-growth probe, with exactly one
  accepted payload allocation. It rejects truncation or size/mtime/ctime change
  before publication. Native and public capability representations are
  private. The streaming weight loader has adopted this authority. A reusable
  ordered model/kernel lease pair, sanitized fixed-FD rooted spawn, and child
  import are implemented. Initial root-bound admission no-follow traverses each
  encoded absolute model/kernel label and requires exact opaque directory
  identity with the corresponding caller-owned capability before retained-pair
  activation or spawn. Failures remain role-specific and payload-safe. The
  root-bound supervisor retains that exact pair, executable, limits, and
  canonical source across zero-argument replacement without ambient label
  revalidation, closing the pair only at instance/service retirement. The child reconstructs
  admission and readiness, then runs a serialized bounded plan/completion loop.
- Synchronous model-configuration file admission now consumes a caller-owned
  approved root plus typed relative locator, closes the immutable bounded
  snapshot source before publication, verifies an independent lowercase
  SHA-256 file identity, and parses metadata under an independently supplied
  model content identity. It exposes no ambient path or async filesystem API.
- The offline reference artifact aggregate now follows the same capability
  boundary: one caller-owned approved root plus one opaque three-locator source
  yields three immutable digest-pinned snapshots synchronously. Same-handle
  per-file/aggregate admission and deterministic file close complete before
  parsing or bundle publication. The CLI alone opens the absolute deployment
  root and closes it before reference execution or output.
- The internal device-worker bootstrap composition now authenticates a decoded
  source before fixed-role root acquisition, derives the bounded paged-Llama
  recipe, closes kernel authority before device preparation and model authority
  before exact readiness publication, and retains independent retryable worker
  and root cleanup authority after compound failure. Transport publication
  remains outside this owner.
- The device child consumes Configure then canonical source from
  the inherited private channel, reconstructs the fixed-root bootstrap into a
  live `DeviceWorkerOwner`, and writes Ready only after an exact readiness
  query and reusable loop storage.
  It then reads one bounded plan, acquires its completion writer before device
  mutation, finishes execution, and publishes the exact completion before the
  next plan. Clean idle waits without a first-byte deadline; partial frames are
  bounded and fail-stop. CPU ordering, positive-controlled release generated-C
  allocation, invalid-config no-Ready, native child-control ABI, and
  AddressSanitizer gates pass; physical CUDA readiness remains deferred.
- An exact prepared Phase-1 device executor that cross-checks model, target,
  catalog, profile, artifact, allocation ownership, physical range, and actual
  pointer-alignment evidence before importing code. It reuses token and
  activation storage, launch arguments, loaded functions, and vendor
  descriptors across synchronous full-sequence dispatches. Invalid tokens and
  unavailable or stale outputs fail closed.
- Explicit executor cleanup ownership across partial construction and partial
  close. A double construction/cleanup failure returns an opaque retryable
  cleanup guard; a close failure blocks further dispatch while preserving
  retry authority over still-open children.
- A distinct versioned paged model graph whose every operation consumes exact
  live-step counts, with positioned rotary input, page-table descriptors, and
  persistent layer-indexed split K/V state. Its catalog is all-AOT and binds
  exact runtime-input order, operation shape, layout version, tokens per page,
  entry point, dimensions, and physical operands before code is imported.
- Fixed-capacity device-step staging uploads counts last after complete host
  validation, authenticates scheduler-retained page generations before plan
  submission, and permanently poisons a staged owner after any partial transfer
  failure. A positive-controlled native release gate proves the warmed public
  descriptor `stage`/`finish` path performs no generated/runtime allocation.
- An owner-mediated ordered full-paged-graph executor that leases the
  caller-owned weight allocation and privately owns descriptor buffers,
  activation/workspace and persistent-KV allocations, stream, modules,
  functions, dimensions, and argument lists. Exact staged/executed capabilities
  enforce lifecycle order. It enqueues exact authenticated operations and waits
  on one reusable completion event; any partial launch/wait failure poisons and
  drains the executor, and partial construction or close retains explicit
  retryable cleanup authority.
- Owner-mediated BF16 terminal-logit readback and completion construction for
  that executor. Startup binds the retained LM-head row geometry and allocates
  fixed readback/logit/selection scratch; successful execution reads only
  producing rows for explicit host sampling, rejects non-finite logits, and
  applies counter-addressed stochastic selection from the exact request seed
  and output index. It appends to the exact-plan completion writer while leaving publication to
  the aggregate owner after executor finish. A positive-controlled native
  release harness now proves a warm public device-worker owner completes the
  production-reachable schema-v2 mixed/full-batch ordinary-prefill,
  final-prefill, decode, and greedy/stochastic lifecycle without MoonBit
  managed/array/string allocation or native resource creation. Nonuniform
  row-dependent fake BF16 logits and an independent scalar oracle cover varied
  row token counts, page CSR, completion slots, and stochastic selection. Fake
  launch/readback/non-finite faults also prove no completion publication, owner
  faulting, writer reuse, and cleanup. An authenticated bootstrap may instead
  select the AOT device greedy reducer, which returns one fixed eight-byte
  result per producing row and never falls back to host sampling when its exact
  embedded artifact is absent or mismatched. Sampling placement is fixed before
  token execution. Physical CUDA numerical correctness and promotion evidence
  remain open.
- A canonical transport-independent inference request and streaming-event
  contract with immutable token/text inputs, exact model identity, bounded
  sampling/stops/deadlines/cache scope, monotonic usage, and payload-safe public
  failure vocabulary. The first trusted receipt boundary can now capture the
  relative deadline exactly once as an opaque absolute monotonic deadline;
  scheduler admission never rebases it after parsing or tokenization delay.
- A canonical fixed-buffer request-v1/event-v2 wire for every `GenerateRequest`
  field and all five `StreamEvent` variants. Completed v2 carries an optional
  bounded UTF-8 terminal tail for unmatched incremental stop prefixes. It
  bounds hostile counts before arithmetic
  or allocation, validates exact lengths/checksums/reserved bytes and canonical
  option encodings, and publishes only epoch-bound opaque frames. A reusable
  `LunaFramedRequestWorkspace` accepts and validates request-v1 through bounded
  work quanta, then lends an epoch-bound scalar/indexed-byte view without
  constructing request payload objects. Its
  fixed-capacity one-frame incremental reader authenticates the 16-byte prefix
  and unsigned declared total before payload acceptance, rejects trailing or
  pipelined input, poisons on prefix/frame failure, and delegates final
  authority to the same scanner plus a synchronous object-form compatibility
  materializer. It is a transport-neutral codec foundation, not a listener or
  ingress claim.
- A bounded incremental decoded-output owner that copies exact tokenizer pieces
  into caller-owned fixed storage, validates UTF-8 across token boundaries,
  and withholds stop strings matched across token or code-point boundaries.
  Pattern tables and scratch are constructed before token stepping. The
  persistent Luna online instance drives this owner through acknowledged typed
  Luna events. `take_event` issues one opaque exact request/event-epoch credit;
  only its ACK advances the semantic owner. Usage ACK atomically creates a
  distinct Completed/Failed epoch, abort invalidates outstanding credit, and a
  cooperative canonical framed-event Workspace borrows only the semantic view.
  Its budgeted Work charges fixed header groups, individual payload/digest and
  checksum bytes, detaches one immutable epoch View only after final semantic
  authentication, and bounds each caller-owned chunk copy by the startup step
  budget. Copying or releasing framed bytes never ACKs the semantic credit;
  real online evidence pins that credit through complete framing and copying
  until explicit ACK. Positive-controlled release-C gates cover the warm online
  and cooperative framing paths. The fixed-lane preparation pool now feeds this
  boundary. This owner deliberately excludes socket operations; the native
  one-shot endpoint, reusable native-framed Server, and serialized OpenAI HTTP
  Server compose partial writes and transport ownership. TLS, keep-alive, and
  concurrent-client arbitration remain open.
- A startup-preallocated, generation-authenticated
  `LunaRequestSemanticStorage` foundation can now stream structurally bounded
  stop-token, stop-string, cache-scope, permission, and inference-limit data
  into budgeted validation and a sorted stop-token index. Its scalar/byte View
  exposes no collection, string, epoch, or release authority. Incremental
  output is the first real consumer: it copies one authenticated stop byte per
  charged setup unit, retains explicit failed-Work cleanup, and detaches only
  after final View authentication at lease transfer. Its preallocated one-View
  slot is included in request-preparation pool reference-cell admission. The
  coordinated object-request path now streams stop tokens, UTF-16 stop text,
  cache scope, and permission into this storage under the pool quantum. Output
  setup receives only the full semantic View; the scheduler receives a narrow
  token-only projection and holds an independent preallocated retention until
  slot recycle. The online worker holds the second retention through healthy
  request retirement or successful recovery close, so claim release cannot
  invalidate scheduler semantics early. Builder-structural failures precede
  Work and do not claim full multi-error equivalence with
  `StopConditions::new`.
- A request-admission bridge now composes that reader with trusted
  monotonic time: the first valid nonempty append samples exactly once before
  byte mutation, and one successful take retains an immutable request plus its
  once-derived absolute deadline without exposing timestamp/deadline getters or
  a detachable frame/receipt capability. The live fixed-lane preparation pool
  authenticates the selected model, tokenizer digest, and exact inference
  envelope; advances retained UTF-8 tokenization, token copying, and
  incremental-output setup through bounded FIFO quanta; and returns a
  revocable `LunaPreparedRequest` shell around one exact claim. Each lane's
  token, semantic, and output storage remains generation-pinned through claim
  release after lower retirement. Saturated or draining admission does not
  consume the receipt; deadlines, cancellation, work ceilings, stale aliases,
  and explicit discard/release are covered. Ready, failed, prepared, and
  claimed results intentionally pin their lane until the outer owner disposes
  of the capability. The legacy facade performs the same semantic-storage and
  View-based output contract synchronously, including blocking tokenization and
  semantic validation; it is not a reactor quantum. Direct framed preparation
  now consumes the scanner View without constructing a `GenerateRequest`: it
  enforces the hard pre-frame and exact request deadlines, compares the selected
  model digests bytewise before input mutation, and accounts receipt bytes plus
  scanner/import work under the same lane ceiling. The native one-shot endpoint
  and both serialized reusable Servers compose this path; concurrent
  multi-connection ingress orchestration remains open.
- A transport-neutral `LunaOnlineFramedCoordinator` now exclusively composes
  one direct-framed preparation pool, one persistent online instance, and one
  cooperative framed-event workspace under a shared monotonic clock. It owns a
  single ordered byte stream: accepted counts leave every unconsumed tail with
  the caller, fixed slots preserve request FIFO across preparation and active
  execution, and a later Ready or failed preparation cannot preempt an active
  request. Opaque byte Offers require exact partial confirmation; semantic
  event ACK occurs only after every framed byte is confirmed, while Usage ACK,
  terminal retirement, recovery, and shutdown remain explicit off-reactor
  maintenance. Disconnect and bounded output/rejection stalls revoke framed
  authority before draining partial, Ready, Prepared, active, and terminal
  owners. Real-worker evidence covers the Usage ACK boundary, later-rejection
  ordering, exact byte confirmation, owner reuse, and partial-receipt close;
  positive-controlled release-C and exact MBTI gates cover warmed transport
  paths and capability opacity. This coordinator itself is not a TCP listener
  or socket writer. The separate one-shot endpoint owns one connection and
  kernel partial-write retry; concurrent connection arbitration, host timer
  wakeups, and multi-client fairness remain the next network-adapter boundary.
- A native-only, private online-TCP output scratch now owns one dynamic `Bytes`
  payload backing and a retained mutable `FixedArray[Byte]` view of that same
  object. The narrow internal bridge borrows the original, increments its
  reference count once for the returned alias, and is callable only from the
  thread-confined service scratch constructor; static bytes, `BytesView`, and
  cross-thread manual reference-counting are outside the contract. Exact-TU
  ASan/UBSan evidence covers pointer identity, length, reference balance, and
  repeated reuse. Positive-controlled release-C evidence covers allocation-
  and bulk-copy-free warmed payload-buffer operations only; it does not claim
  that a future async TCP task or runtime scheduling is allocation-free. The
  scratch remains private foundation, not a listener, socket owner, or public
  transport capability.
- A backend-neutral worker protocol for exact prefill/decode rows, flattened
  token/page/capability tables, plan and model generations, completion slots,
  and typed completion records. Its reusable fixed-capacity plan and completion
  buffers authenticate lifecycle epochs; scalar row drafts avoid temporary
  table construction; final prefill explicitly returns the first generated
  token; provenance-bound row capability recipes preserve exact model-plan
  operation order; and whole-build checkpoints can roll back several committed
  rows. Submitted-work validation remains separate from the scheduler's
  current-generation publication check. The scheduler owns distinct paired A/B
  plan and completion buffers, while the production root-bound service admits
  one outstanding plan for its serialized child. It issues exact-epoch completion writers and
  retires completed owners strictly in plan-sequence order. Canonical bounded
  little-endian plan and completion frames now detach that protocol from heap
  owner capabilities: each receive validates checksum, exact counts/ranges,
  identities, sampling fields, outcomes, and canonical table coverage before
  replacing an epoch. Startup-sized open-addressed identity tables provide
  O(1) duplicate and completion-slot lookup without steady-state allocation.
  Plan construction fills each scalar draft and its retained tables in one
  pass. The service
  authenticates each received completion frame against the retained exact plan
  and converts it into the paired scheduler completion owner without retiring
  work, preserving publication-backpressure retry. An exclusive worker-side
  writer now produces canonical completion frames directly from exact received
  plan rows with abort/stale-epoch safety, so no scheduler completion owner is
  needed across this boundary. Reusable device-step staging also consumes the
  validated plan frame directly without a scheduler plan owner in the worker.
  Its post-execution path binds logits to the exact retained frame owner and
  epoch, applies validated scalar sampling fields, and freezes a canonical
  completion frame directly; no scheduler plan or completion owner is
  reconstructed in the isolated side.
  The private native layer now provides shell-free descriptor-pinned
  exact-executable spawn. Linux admission snapshots and hashes one no-follow
  descriptor, copies the authenticated bytes into an exactly sealed memfd,
  and later duplicates that opaque capability into `fexecve(5)` without a
  second pathname open. Unsupported live-spawn platforms fail closed. Parent
  signals are blocked before fork; the child resets every catchable
  disposition while blocked, closes descriptors 6 through `UINT_MAX`, and
  installs its empty final mask immediately before exec. The public raw-path
  preparation surfaces have been removed; test fixtures authenticate through
  the same admission and remain outside generated production APIs. The layer
  also provides an
  inherited socketpair endpoint authenticated by construction, fixed-buffer
  exact I/O with an unbounded idle first-byte wait and bounded partial-frame
  deadlines, bounded wait/terminate, and deterministic kill/reap close. A legacy
  protocol supervisor owns distinct A/B plan and
  completion frame buffers, enforces monotonic submission and oldest-first
  receive, pins a validated response until explicit post-publication
  retirement, and permanently fails malformed sessions. The production worker
  channel now has a separately linked child-side frame runtime and gates
  covering three monotonic echo exchanges, A/B/A reuse, retained
  response inspection, EOF, zero exit, and reap. That legacy echo child
  performs an exact checksummed
  Configure/BootstrapSource/ParentApprovalAttestation/Ready handshake that binds model
  identity, an admitted-bootstrap SHA-256 derived from graph/artifact evidence,
  a bootstrap-source SHA-256 derived from canonical `EncodedBootstrapSource`
  bytes, model generation,
  exact process-visible device ordinal, predecessor, worker limits, and
  inference limits. It canonically decodes and checks the source and consumes
  the exact launch-bound one-shot parent attestation before the fixture
  publishes protocol readiness. Startup
  cleanup retains explicit authority after a double failure, and submission
  rejects a foreign model generation before transport mutation. The echo child
  returns deterministic protocol completions rather than opening CUDA; the
  separate device child owns the real serialized execution loop. Closed-child
  recovery now retains validated responses, retires exact unreturned
  submissions in sequence order, and derives a replacement predecessor only
  after all obligations are discharged. The real-process gate proves a clean
  replacement, an abandoned sequence 4, and successful continuation at
  sequence 5. A new thread-confined worker service encapsulates scheduler and
  root-bound process owners, checks its single production credit before
  scheduler mutation, records plans before transport, validates received
  retirement authority before scheduler mutation, and stages each frame once
  into a paired typed completion owner. Publication pressure retains a private
  accepted-flight state and returns a nonallocating scalar backpressure result;
  retry neither rereads the wire frame nor reopens or advances the completion
  owner. The service commits bounded worker failures before abandonment and
  replaces the child only
  after the outstanding obligation retires and every surviving active request
  is terminally invalidated with balanced host KV release. Waiting requests are
  preserved because they own no device bytes. An independent retained
  `WorkerServiceBinding` pins the expected bootstrap and bootstrap-source digests, device ordinal,
  and inference envelope; each join/restart also verifies exact scheduler-owned
  model identity, generation, predecessor, and worker-protocol limits. Its
  owned-instance factory now consumes an immutable scheduler blueprint and
  ordinary approved roots, constructs both mutable owners internally, and
  publishes a one-shot ready-or-cleanup shell. The alias-taking constructor is
  fixture-only. Worker buffers, child ownership, executable snapshot, and
  Configure/source/parent-attestation/expected-Ready frames are preallocated before root
  acquisition; native spawn, scalar handshake validation and cleanup, and
  owner publication allocate no managed objects with rooted authority live.
  The owned service now also makes a permanent Raw-versus-Online family
  choice. Its restricted epoch lease authenticates startup-bounded streaming
  request lanes, exact generation/position/publication order, monotonic
  admission/deadline
  time, cancellation cuts, worker-loss recovery, and deterministic close
  without exposing scheduler/process/handle owners. The alias-free
  `LunaOnlineInstance` prepares that worker once, consumes externally prepared
  `LunaPreparedRequest` claims into fixed lanes under fresh opaque tickets,
  publishes through one canonical event credit, and provides exact request
  retirement plus off-reactor abort/recovery/instance-shutdown paths. Healthy retirement
  preserves the worker, lease epoch, publication cursor, and plan history;
  worker/protocol/device failure cooperatively replaces the worker before the
  affected request's recovered terminal is published. The instance owns no
  tokenizer or raw receipt state; Busy and Draining leave prepared authority
  retryable, exact preparation mismatches fail before lower mutation, and
  destructive claim transfer rejects retained-alias replay.
  The borrowed ordinary roots remain caller-owned. Normal allocation-free
  Token output and natural Maximum/StopToken Usage+Completed-v2 bundles are
  implemented; stop tokens are counted but suppressed. Incremental string-stop
  matching now withholds stop/post-stop bytes, commits an exact cancellation
  cut when possible, and translates only its authenticated terminal to
  Usage+Completed(StopSequence); final-token natural-terminal precedence is
  covered without cancelling an already-terminal request. Caller cancellation
  now defers behind pinned credit and publishes Usage+Completed(Cancelled), and
  automatic owner-clock expiry publishes Usage plus canonical nonretryable
  Failed(deadline_exceeded), with natural terminal precedence. Decoder/output
  rejection and authenticated worker/device loss now retain exact terminal
  authority and publish Usage plus fixed payload-safe Failed(output_invalid,
  nonretryable) or Failed(worker_unavailable, retryable). Reactor-safe progress
  reports when explicit off-reactor recovery/reap/drain is required. Ingress
  transport remains open;
  this is not online-serving readiness.
  Its
  real-process gate proves output-publication backpressure, worker death, failure retirement,
  non-reusing restart, post-restart completion, and balanced KV ownership.
  The full-graph physical blueprint and artifact bundle now derive the
  admitted-bootstrap digest
  from bounded canonical module, symbol, launch, layout, operand, device-step
  envelope, and exact device-assignment evidence.
- An aggregate `engine/device_worker` readiness owner. Its inert admission
  retains the independently expected startup contract and exact bounded model
  path, weight, device, memory, kernel, artifact, and bootstrap evidence.
  Preparation compares the complete received contract before resource opening,
  opens and verifies the exact visible device ordinal/target, loads verified
  weights internally, prepares the complete paged executor, and exposes only
  the exact readiness contract while context, weights, and executor remain
  live. Executor/weights/context cleanup is dependency ordered, retryable, and
  retains authority after compound preparation/cleanup failure. The production
  child imports its fixed approved roots, reconstructs and authenticates the
  encoded source through `engine/device_worker_bootstrap`, derives `Ready` from
  the live `DeviceWorkerOwner`, and forwards bounded plan/completion frames
  through that owner. The real-child gate currently proves hostile startup and
  failure behavior; positive spawned-child execution with physical CUDA and
  numerical output remains unproven.
- A generational fixed-page KV metadata `PageAllocator` with preallocated
  arrays, an intrusive FIFO free queue, separate active and cached references,
  exact-run rollback, terminal-generation retirement, invariant diagnostics,
  and a randomized ownership model test.
- A fixed-capacity request `BlockTableArena` with generational table ownership,
  preallocated dense page mappings, reusable rollback/release buffers, and
  randomized ownership fixtures. Canonical inline `PageIdStorage` and
  `BlockTableIdStorage` retain optional identities without boxed option cells;
  they do not acquire or release the underlying resources.
- A compressed startup-preallocated token radix with full-page-only
  longest-prefix reuse, complete cache identity salting, fixed entry/node/token/
  page/scope arenas, transactional result and publication buffers, active
  references, and deterministic priority/LRU zero-reference eviction. The
  scheduler integrates lookup, page/block-table acquisition and rollback,
  final-prefill publication, cache-disable/read-only/read-write policy,
  fairness-bounded reusable-depth priority, and bounded cache telemetry. The
  full salted-root and `(root,parent,first-token)` child indexes are balanced;
  every arena uses an `O(1)` free stack, the exact zero-reference victim is a
  min-heap, duplicate pages are rejected by fixed-scratch heapsort, direct
  protector metadata avoids descendant scans, and compaction visits only the
  changed ancestry. Waiting slots retain logical candidate evidence only;
  activation exact-revalidates and transactionally acquires references with
  rollback at every scheduler checkpoint. The radix package owns no device
  memory and imports no device implementation.
- Focused scheduler and cache configuration records for token/request budgets,
  chunked prefill, waiting-age policy, the two-plan-buffer invariant, bounded
  generated-token publication, physical page/block-table limits, prefix
  metadata arenas, reference bounds, and cache layout identity. Startup-only
  runtime capacity resolution now checks the scheduler, cache, model-shape, and
  worker envelopes and materializes canonical page-allocator and block-table
  limits. The scheduler subsequently authenticates exact intermediate-prefill,
  final-prefill, and decode row recipes against the loaded model generation and
  capability-cell capacity.
- A deterministic single-owner scheduler foundation with fixed request slots,
  an intrusive FIFO waiting queue, globally unique request generations, bounded
  terminal notices, and tokenized admission against the resolved runtime plan.
  It validates model identity, duplicate IDs, context/page envelopes, exact
  receipt-relative deadlines, and recipe provenance; the core request type
  still rejects nonempty stop strings until the incremental owner is composed;
  cancellation and deadline sweeps preflight publication and identity capacity
  before mutation. Its bounded `build_next` path activates FIFO requests only
  after submission, reserves decode resources before prefill policy, emits
  prefill rows before decode rows, and uses exact plan, block-table, and page
  checkpoints for rejected-build rollback. Intermediate prefill, final prefill,
  and decode use separate authenticated capability recipes. Distinct A/B plan
  owners permit two outstanding submissions in scheduler and echo fixtures;
  the production root-bound service restricts the serialized device child to
  one. Paired completion owners use a
  fixed slot index and exclusive writers; full-batch retirement authenticates
  current versus cancelled/expired work, preflights both publication rings and
  KV release, publishes final-prefill/decode tokens in plan order, enforces
  token-stop/maximum-output terminals, and resets the exact plan side. A shared
  monotonic publication sequence makes the separate token and terminal rings
  fail closed against out-of-order dequeue, including cancellation cuts. A
  single owner-selected dequeue now returns the exact globally oldest typed
  publication without exposing a ring choice. Normal
  idle and two-owner pressure are flat allocation-free outcomes in generated
  native code; the positive-controlled warmed scheduler allocation gate passes.
- A bounded `lunaflux reference` command that opens one approved absolute
  deployment root, validates three strict relative locators and digests, loads
  an admitted host snapshot, closes the root, and produces offline greedy
  tokens.
- A reusable depth-bounded JSON duplicate-key guard for map-backed parsers.
- A Phase 5 kernel-program and offline-release foundation spanning
  semantically neutral LunaTile validation/serialization, a generic
  deterministic serial-reference CUDA translation for affine copy,
  asynchronous copy, barriers, pipeline stages, MMA, reductions, and
  elementwise operations, and exact BF16 family reference
  lowerings for embedding, RMSNorm, positioned RoPE, residual, projections,
  gated MLP, and mixed paged attention, one closed deterministic CUBIN builder,
  and deterministic schema-v2 full-graph manifest production. Pointwise,
  projection, and paged-attention artifacts use a non-circular candidate ->
  offline compile -> final-contract binder, and bundle admission accepts only
  those opaque bound forms. Authenticated startup may select the embedded AOT
  greedy reducer, while temperature/top-k/top-p remain on the preallocated host
  path; no sampling placement decision occurs in a token step. The execution owner now enqueues the
  exact BF16/I8 graph and waits once rather than synchronizing every operation.
  Its startup-only CUDA graph seam can capture that exact prebuilt function,
  geometry, and operand sequence into one reusable graph exec on the retained
  stream. Optional ABI absence falls back only under the explicit fallback
  policy; required capture fails closed. No warmed path builds or updates
  nodes, and graph/event close failures retain dependency leases for retry.
  Opaque graph telemetry derives captured hits or ordered-eager misses from the
  actual private executor mode, retains the authenticated selected shape, and
  updates allocation-free saturating counters only after successful completion.
  Fake-driver and ASan/UBSan evidence pass. A capture-required sm120 add-one
  probe also passed 128 exact launch/poll/close cycles without selecting eager
  fallback. A typed BF16 release producer now turns exact two-build CUBIN
  evidence into catalog-v3 families, full launch contracts, strict family
  bindings, a canonical schema-v2 manifest, and an exact kernel-root plan. Its
  two-layer fixture proves 21 operations reuse nine physical modules without
  catalog ambiguity. The authenticated model-plan candidate export and strict
  offline compilation join are complete. Authenticated graph-memory accounting
  now distinguishes eager-only absence from a v2 declared capture upper bound,
  includes the bound exactly once in the startup ceiling, and rejects missing,
  unsupported, under-ceiling, or overflowing joins. A separate profile-priority
  admission deterministically binds stateless full-context observations to the
  immutable plan and target and identifies the largest self-time operation. It
  also has a disjoint paged path derived from an admitted launch set, with
  exact mixed-row/cache/position/page geometry, a trace-only page-table digest,
  and bounded sorted operation counters. It has no profiler, compiler,
  kernel-promotion, or runtime authority. The separate evidence-gated LunaTile
  promotion owner remains inert until a real-operation specializer maps exact
  operands, shape, and numerical contract. Broad shape coverage and
  performance remain unclaimed.
  A native approved-tiny-model campaign harness now pins the exact upstream
  model content, upstream tokenizer, canonical single-row plan, legacy runtime
  recipe, and independently supplied child executable digest. It reaches the
  production spawned child/service preparation and an owner-preserving offline
  validator submits one fixed two-token framed request through the existing
  service boundary. Only the pinned `1031,2185` token events, consecutive
  one-row/one-token prefill/decode plan sequences, same-page KV residency, and
  terminal Usage/Completed facts can produce its opaque result. The validator
  deterministically disconnects, drains, closes, and reaps before returning;
  it exposes no service, worker, device, scheduler, or selected logits. This is
  The r14 physical campaign passed this pre-listener path. The later r17
  campaign crossed the production native loopback listener and passed the
  exact five-event request plus network/KV/listener/child balance. This remains
  one bounded request, not broad, concurrent, TLS, or performance proof.
  A separate product-owned offline exporter now lowers an authenticated
  `ModelPlan` into the exact canonical candidate declaration, generated CUDA
  sources, typed recipes, and independent SHA-256 inventory consumed by the
  offline BF16 builder. Publication is atomic and no-overwrite. Its native
  adapter authenticates and closes the approved model root before exporting
  the fixed sm120 envelope with an exact bounded compiler major/minor/patch
  identity. It invokes no compiler and owns no process, device, runtime,
  worker, or serving authority. The separately approved CUDA 13.1.115 campaign
  has compiled and executed the pinned model, while the exporter itself still
  confers no readiness.
  No production path imports compiler/process authority or performs JIT.
- A Phase 6 operator foundation spanning strict byte-bounded configuration
  documents, focused immutable records and startup explanations, inspection-
  safe kernel metadata, and `run`, `doctor`, `plan`, `bench`, and
  `inspect-kernels` preflights. A strict `lunaflux.runtime.v1` loader accepts
  an independently pinned descriptor plus separately approved model and kernel
  roots, then composes the existing model, weights, KV, execution-manifest,
  bootstrap, startup-contract, and inert device-worker admissions. It opens no
  device and retains no root; host plan and kernel inspection may succeed even
  when physical hardware admission does not. Legacy run/bench preflight always
  reports false readiness instead of inventing model, kernel, benchmark, or
  service state. The separate digest-suffixed one-argument `run` path admits a
  fixed launch envelope and exact instance policy, verifies its worker binary
  and assigned CUDA target, and composes the existing live worker/service/TCP
  owners. It can publish readiness only while that exact listener and lower
  service remain live; drain or failure clears readiness. The offline
  deployment-bundle assembler requires exact launch/model/policy/kernel/library
  inventories, preserves model bytes outside the OCI rootfs, never overwrites
  an output, and rejects substituted or ambient files, PTX/JIT content, and
  injected partial-transfer failure. `lunaflux validate-release` shares the
  live startup semantic join and authenticates descriptor, policy, model,
  kernels, tokenizer, bootstrap, and worker executable without device,
  process, or listener authority; it closes all approved roots before returning
  canonical digest evidence. The separate atomic materialization workflow now
  pins a digest-authenticated host preflight executable, claims one new output,
  assembles beneath that private claim, and maps each no-follow staged source
  capability to the exact final absolute label encoded by launch/bootstrap
  evidence. It publishes only after semantic evidence, tool/materializer
  digests, exact inventories, and source-target/no-overwrite flags verify.
  Semantic rejection, target substitution, and injected partial publication
  before verified-stage installation remove the exact empty claim and retain no
  output. Publication itself is now an explicit recoverable v2 transaction:
  canonical output and current uid are claim-bound, prepared evidence binds the
  exact inventories, and only the six-entry prefix state can resume. Crash
  fixtures cover every prefix transition; hard-link, symlink, FIFO, unknown,
  non-prefix, canonical-output, and inventory substitutions are refusal-only.
  Recovery never recursively deletes an output. This does not prove the supplied
  tool's build provenance, an OCI image, SBOM, reviewer approval, or physical
  inference.
- Restrictive top-k sampling now uses fixed-scratch `O(V log K)` selection with
  canonical logit-descending/token-ascending ties. Unrestricted and top-p
  sampling retain canonical `O(V log V)` ordering for exact floating-point and
  RNG replay. Whole-row finite validation precedes output mutation.
- Multi-request online publication routing is a startup-preallocated one-probe
  table authenticated by exact worker generation and live request identity;
  terminal retirement clears the exact mapping only after lower retirement.
  Native and OpenAI servers share one private telemetry bridge with exact-once
  unsigned prefix-counter deltas, current gauges, plan histograms, KV-total and
  shape drift detection, and sticky degradation.
- The inert benchmark-evidence package now admits at most 8,192 canonical
  terminal request records per trial, rejects reordered/missing/malformed or
  overflowing measurements, and derives deterministic integer nearest-rank
  p50/p95/p99 queue, TTFT, service, decode, and end-to-end summaries. Separate
  domain-bound raw and summary digests retain every failure/rejection and feed
  the existing complete three-engine/nine-profile/three-trial comparison.
  This is evidence structure, not a runner or performance claim.
- Phase 5 now also has two isolated block-128 fused AOT families: QKV with
  positioned RoPE and paged split-K/V write, and residual addition with the
  immediately following RMSNorm. Exact plan adjacency, BF16 shape/layout,
  target, strict compiler, profile, fallback identities, operands, alignments,
  source, recipe, toolchain, numerical policy, and dispatch canary are bound.
  A source-owned qualification exporter derives their exact canonical envelope
  from the authenticated model plan and KV layout and publishes it atomically
  without compiler, device, runtime, or promotion authority. An independent
  test-only scalar referee covers four page-boundary shape classes with fixed
  tolerances and exact-cell corruption detection. The existing closed offline
  builder accepts both exact fused recipe schemas, and a typed candidate-only
  binder requires two byte-identical CUBINs plus the canonical receipt and a
  separately pinned receipt digest while binding source, recipe, compiler,
  target, geometry, symbol, and ABI. Scalar differential, page-edge, canary,
  deterministic lowering, hostile substitution, static-resource, exporter,
  builder, and compiled-binding tests pass locally. A source-only physical
  campaign now composes deterministic double builds, the public device facade,
  exact-cell scalar comparison, memcheck and racecheck, typed evidence
  admission, and independent artifact/measurement/outer seals without a
  self-referential digest. `CAMPAIGN_RESULT.txt` is covered by the outer
  manifest. Admission binds the two asserted inner manifest digests but does
  not inspect or certify filesystem modes. Hostile fake-tool transactions reject
  dirty or ambiguous sanitizer output, runtime stderr, tool identity drift,
  noncanonical numerics, partial output, overwrite, and unsafe output names.
  A separate fixed-shape paired benchmark evidence package remains inert and
  makes no timing claim. Production V2 lowers away qualification canaries and
  supplies exact startup-admitted runtime artifacts. Both legacy Llama and
  numeric-BF16/Mistral worker APIs prepare the residual/RMSNorm and
  QKV/RoPE/KV-write/read-only-attention replacements while retaining the
  standalone path when optional artifacts are absent. Descriptor-pinned
  optional artifacts now reach deployed Llama and numeric-BF16 children, with
  the QKV aggregate admitted only after live-device identity authentication.
  Physical CUDA correctness,
  sanitizer/race, benchmark-win, mixed-workload, and reviewer evidence remain
  open.
- A repository guard for package direction, sibling-product independence,
  CUDA ABI ownership, Python exclusion, temporary debt markers, and source-size
  review thresholds.

## Explicitly not implemented or not yet proven

- Phase 5 now has deterministic CUDA source emission for the initial BF16
  families, a physically exercised offline CUBIN builder, non-circular
  candidate-to-final binding for pointwise/projection/attention, and a complete
  typed reference-kernel bundle producer. All eight BF16 reference families
  and the capture-required primitive graph lifecycle passed scoped sm120
  correctness probes. A complete tiny twelve-launch paged-BF16 graph then
  passed 128 required-capture cycles with mixed prefill/decode KV writes,
  fourteen independently refereed boundaries, exact fixture logits/token,
  closed resources, and empty stderr. A bounded five-case shape matrix then
  passed 40 captured graph launches. The pinned upstream tiny BF16 model also
  passed physical sm120 execution from its canonical 21-operation offline AOT
  build: eleven selected logits matched within `0.0005459413`, greedy tokens
  were `1031,2185,688,2844`, same-page KV persisted, and resources closed.
  The pinned spawned worker/service path now has one physical two-token
  pre-listener execution pass with exact plan/KV balance and child cleanup.
  One bounded native-listener request now passes physically. Optimized kernels,
  broader production shapes and contexts, leak/soak coverage, and
  kernel/end-to-end benchmarks remain open. External
  deployment approval is binding evidence. LunaFlux does not independently
  verify the deployment's detached-signature scheme; it authenticates the
  startup-supplied approval binding privately.
- The exact tokenizer shipped with the pinned upstream model is now admitted by
  a closed SentencePiece-derived BPE profile covering its normalization, BOS
  template, complete byte fallback, fused-unknown policy, and ordered decoder.
  Its independently recorded token IDs and empty-input text-to-model reference
  execution pass through the authenticated production loader. General
  SentencePiece variants remain rejected; production use still requires an
  independently approved tokenizer artifact and deployment identity.
- The streaming device loader now uses pinned approved-root and regular-file
  capabilities rather than ambient string paths. Deployment approval and
  read-only mount policy remain external trust inputs. Low-level fixed
  descriptor-role inheritance, production supervisor retention across restart,
  and child reconstruction from those roles are implemented and proven.
- The 2026-08-27 NVIDIA campaign physically validates the private CUDA
  primitive boundary on an RTX 5060 Ti: driver loading, 4 KiB transfers,
  events, exact BF16 GEMM, PTX module/function loading, direct and ordered AOT
  launches, and deterministic closure passed 128 cycles with unchanged device
  memory. A generated residual-add CUBIN subsequently passed another 128 exact
  BF16 ordered-executor cycles, and the digest-pinned reference paged-attention
  fixture passed mixed prefill/decode output and KV-write comparison with
  maximum absolute error `0.0104694`. A final current-fixture campaign compiled
  and numerically checked embedding, RMSNorm, RoPE, residual, QKV, dense output,
  gated MLP, and LM-head kernels for three adversarial live tokens. A separate
  capture-required graph campaign passed 128 cycles without eager fallback. A
  later complete synthetic mini-Llama graph composed twelve real AOT launches
  for 128 captured cycles and matched all fourteen CPU-refereed boundaries,
  both KV arenas, final logits, and greedy token exactly. These campaigns closed
  with empty runtime stderr. The final integrated source snapshot
  `8351d804...41ae8` repeated that graph result and also passed 1,902/1,902
  Linux native tests, the portable 10,000-request Phase 2/3 balance proof, all
  BF16 release and I8 software gates, CUDA ABI, both warmed device-worker
  allocation gates, ordered-executor ASan/UBSan, Phase 2/3 diagnostic, and
  release-evidence gates. Its I8 physical probe produced an explicit
  fail-closed result before resources were opened: that sealed source admitted
  only `sm89|sm90`. Current software admission also includes exact `sm120`, so
  the older rejection remains historical rather than current policy. Eight native ASan/UBSan probes
  and the CUDA/NCCL ABI gates also pass; the MoonBit runtime itself was not
  sanitizer-instrumented.
  The later clean-source campaign `d3d2a556...7064e` added approved tiny-model
  BF16 numerical evidence on the same sm120 device, with authenticated weights,
  AOT artifacts, required graph capture, same-page KV persistence, and closed
  resources. Its final Linux native check was warning-free and 1,936/1,936
  tests passed. That historical snapshot's hardware sweep repeated the
  five-case BF16 shape matrix, its then-exact sm120 I8 rejection, bidirectional no-peer inventory, and
  the CUDA-peer/topology/NCCL static sanitizer gates. Physical NCCL remains
  unavailable. This is not positive I8, performance, or general readiness
  evidence. A later r14 campaign separately proves the pinned spawned-worker
  pre-listener service path.
  A product-owned Phase-7 physical topology diagnostic now composes the CUDA
  inventory, canonical directed peer matrix, strict typed topology admission,
  and dynamic collective-runtime admission into deterministic immutable
  schema-v1 evidence. The mixed sm120/sm75 no-peer machine is an expected
  exit-zero rejection rather than degraded tensor parallelism; missing
  `libnccl.so.2` remains a typed diagnostic and grants no communicator.
  See [PHYSICAL_VALIDATION_2026-08-27.md](PHYSICAL_VALIDATION_2026-08-27.md).
- The final 2026-08-28 integrated archive
  `cfef45b...60bb` passed warning-denied Linux native check and 1,957/1,957
  native tests. Its OCI, deterministic bundle-assembly, atomic
  materialization, deployment-boundary, and spawned-execution static contract
  gates passed with empty stderr. The same source repeated the approved tiny
  BF16 model on sm120 with authenticated weights and AOT artifacts, required
  graph capture, maximum selected-logit error `0.0005459413`, greedy tokens
  `1031,2185,688,2844`, same-page KV persistence, explicit closure, and no
  remaining compute process. The root-bound concurrent OpenAI pool also passed
  slow/fast-client progress, connection reuse, retirement, and drain. This is
  historical r9 result did not include spawned-worker physical CUDA. The later
  r14 campaign closes that narrow token-level pre-listener gap; positive I8,
  tensor parallelism, general serving readiness, and benchmarks remain unpromoted.
- Physical and numerical proof for the new paged path. The semantic graph,
  host page/table ownership, reusable device descriptor, persistent device KV
  allocation, exact all-AOT launch ABI, artifact admission, physical blueprint,
  single-wait ordered full-graph dispatch, and aggregate readiness owner
  are implemented. The synthetic twelve-launch graph and pinned tiny-model
  21-operation graph have now passed real-CUDA logits, sampled-token, KV-write,
  capture, and closure validation, and a bounded five-case shape matrix passed.
  The r14 approved deployment crossed the spawned child and owned framed-
  service path for one pinned two-token request, with exact `1031,2185`
  output, consecutive prefill/decode plans, one-page geometry, balanced
  retirement, empty stderr, and full child/device cleanup. A later r17
  campaign crossed the actual native loopback listener with exact event order,
  one accept/disconnect, restored 32-page balance, and closed listener/child
  authority. Broader shape/context, full leak/soak, TLS, and comparative
  benchmark gates remain open; the latest final7 campaign closes the bounded
  broad-listener qualification described above.
  A fresh portable current-source r20 requalification then passed all 12
  bounded physical cases on the same sm120 GPU: the warning-denied Linux check,
  primitive and BF16-family probes, paged attention, 128-cycle complete graph,
  five-case/160-launch shape matrix, and two byte-identical approved-model
  runs. The selected-logit error remained `0.0005459413`, tokens remained
  `1031,2185,688,2844`, resources closed, and no compute process survived. A
  separate listener rerun repeated the exact five-event stream and balance.
  That snapshot again rejected I8 on sm120 and the heterogeneous sm120/sm75
  no-peer topology before resource authority; current software admission has
  since added exact sm120. These results do not
  add cached-prefix, positive I8, homogeneous tensor-parallel/NCCL,
  concurrency, soak, or performance evidence.
  The
  stateless reference interpreter remains the correctness oracle and
  deliberately recomputes full sequences.
- TLS, HTTP keep-alive/pipelining, or broad spawned-service paged/cached
  execution and listener concurrency. The public in-process
  `LunaOnlineInstance`
  retains one worker across startup-bounded concurrent request lanes and implements
  generated-text decoding, sampling-result consumption, exact
  one-credit Token/Usage/Completed/Failed publication, cancellation, deadlines,
  request retirement, explicit drain, and cooperative failure recovery with a
  canonical failure terminal before the restarted worker is reused.
  The
  scheduler registry/lifecycle, worker, sampling, page-allocation, block-table,
  prefix-index, and runtime-capacity foundations exist, and the scheduler,
  root-bound service, spawned device child, and device-worker aggregate are now
  connected. The native pipeline server proves two live requests, a real
  two-row/two-token plan, deterministic global event order, cancellation
  isolation, and balanced lane/KV retirement on one socket. The reusable
  native-framed and OpenAI HTTP servers prove listener reuse and bounded
  telemetry. The bounded root-bound OpenAI connection pool now proves that a
  slow client cannot block a fast client, and proves exact connection reuse,
  retirement, and drain. There is still no TLS, HTTP keep-alive/pipelining,
  public-network promotion, or broad context/profile evidence. The latest
  final7 campaign adds bounded physical concurrent-request, saturation,
  rejection/recovery, restart, drain, and one-request latency qualification;
  it does not make a general production-latency claim.

Canonical `lunaflux doctor`, `lunaflux plan`, and
`lunaflux inspect-kernels` now take the same digest-suffixed deployment operand
as `run` and `bench`. They authenticate the recipe-specific runtime-instance
join, close every root, and publish only root-free evidence without opening a
device, spawning a child, or binding a listener. Malformed and unpinned
operands cannot fall through to weaker model-root preflight. The older host,
MODEL-only, separate-root, and caller-declared configuration diagnostics remain
only under visibly named `legacy-*` compatibility commands and never report
readiness. `lunaflux reference` remains the digest-pinned correctness executor.
A compiler/JIT-free OCI source contract, exact
context/base verifier, Linux-only build wrapper, hostile static gates, and
deployment runbook now exist. Their structural packaging, assembly,
materialization, and deployment-boundary gates passed on Linux in the sealed
final7 campaign, but no approved Linux/CUDA image, base/builder provenance,
final-rootfs/SBOM scan, reproducible final digest, or physical image readiness
is proven. A separate deterministic final-release inventory now joins the
externally approved OCI digest, SBOM, license inventory, provenance, kernel
manifest, rootfs scan, runtime contracts, and exact source identity while
preserving exact authenticator/tool bytes and replaying every approval.
Hostile substitution, symlink, hard-link, cross-subject replay, authenticator
substitution, no-overwrite, deterministic replay, and partial-publication
gates pass locally. The external authenticator allowlist and actual approved
artifacts are not fabricated by this wiring, so this is not a completed
image/release claim. Doctor preflight is not service activation. All incomplete
paths retain production readiness as false.

## Next first-release gate

The authenticated single-GPU BF16 loader, complete paged execution path,
spawned worker, bounded native listener, pinned cached-prefix case, exact
resource balance, and one real-timestamp LunaFlux measurement now have physical
evidence. Loopback health/readiness and the inherited local drain capability
are now implemented and locally proven. The remaining first-release work is
physical and publicly routable approval for those controls, deployment-owned
TLS/authentication/generation fencing, physical context-churn/actual-context
and broader device-leak coverage,
the full pinned and counterbalanced LunaFlux/vLLM/SGLang comparison, an approved final
OCI with SBOM/provenance/rootfs scan, and an actually routable opaque LunaNexa
deployment with reviewer acceptance. Full MoonBit-runtime sanitizer
instrumentation and broader physical concurrency/race evidence also remain
open. Positive I8 and tensor-parallel/NCCL evidence are later capability gates,
not blockers for the initial single-GPU BF16 release.

The sibling LunaNexa repository now also has an authenticated rollout-scoped
ten-subject promotion verifier with deployment-injected HMAC-SHA256 key
ownership, constant-time receipt authentication, deterministic key wiping,
private catalog construction, and hostile substitution/replay coverage. The
legacy caller-claim inspector remains deliberately non-routable, and no public
catalog or approval-input constructor exists. Production keys and policies,
issued signed receipts, and an approved catalog entry remain deployment-owned;
none is committed, so no deployment has been promoted.

LunaFlux external approval is now capability-safe: public signed evidence is
inert, the admitted approval is opaque, and the HMAC-SHA256 verifier exact-binds
immutable rollout policy, manifest bytes, detached envelope, and approved
source with closed-state rejection, constant-time tag comparison, and key/pad/
digest wiping. No public arbitrary-key constructor exists. The trusted
startup verifier-key handoff and per-child one-shot attestation bridge are now
integrated, and legacy worker bootstrap approval claims still fail closed.
Promotion nevertheless remains inert because no deployment-issued key/policy,
signed receipt set, approved catalog entry, sealed performance evidence, or
real-operation mapping is committed. Baseline inference does not depend on
this promotion seam and is unaffected.
