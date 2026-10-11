# Independent GLM / DeepSeek checkpoint correctness — 2026-10-11

Internal ordinary/speculative agreement and successful resource cleanup are
not independent numerical correctness. This work keeps that distinction and
does not add production authentication, hashing, TLS or release prerequisites.

## MiniMax schedule correction and matched decoder inputs

Source comparison found a concrete schedule mismatch, independently of text
encoder rounding. Installed original H3 uses CPU F32 `linspace(1,0,N)`, applies
the modality shift in F32, and evaluates `N-1` intervals. Native previously used
the generic F64 training-floor grid ending at `1/1000`, appended zero, and ran
`N` evaluations. Thus the earlier five-step media runs are not original-schedule
executions, even when all outputs are finite.

The family-neutral shape contract now declares its sigma-grid law. MiniMax
selects inclusive-zero F32; other plans retain their existing training-floor
law. Pure startup planning computes the scalar grid and interval cardinality;
device lowering consumes the prepared flow table without model branching.
The AOT request helper no longer assumes every grid has `N` intervals. Existing
plan associations distinguish the changed law; no new payload scan, runtime
cryptography or serving validation is introduced.

The pure planner matches **2,174 actual original CPU sigma outputs bitwise**
at counts 2, 5, 10, 20, 50 and 1,000 with shifts 3 and 12. The non-exact linspace
points require one final F32 multiply-add rounding; the subsequent shift
operations round separately. Original outputs and comparison are preserved at
`/tmp/lunaflux-minimax-original-sigma-20261011-v5`. The test-only bridge imports
the installed original routine; it does not substitute a new reference formula.

The selected-conditioning captured-latent run completed on Spark `.178` at
`/tmp/lunaflux-minimax-joint-latents-selected-20261011-v1`: five old-schedule
evaluations, all fifty layers, both VAEs, finite media and canonical terminal
latents. It is an isolation fixture, not a corrected-schedule quality pass.
Downloaded latents are standardized, before mean/std conversion.

An initial audio comparison incorrectly paired the older standalone native
**destandardized zero** with the original adapter's **standardized zero**.
Its 97.5% waveform relative L2 is an input-mismatch result, not attribution of
an AudioVAE defect. The original replay now has an explicit destandardized-input
mode for that standalone fixture; joint generated latents continue through the
original adapter's normal standardization conversion. Identical-input replay
must precede decoder-error and media-quality claims.

That identical-input audio replay is now complete on Spark `.179`. Original
raw decoding versus native standalone destandardized zero has **0.053442%**
relative L2, RMSE 0.0000493934 and maximum absolute error 0.000781611 over
331,200 finite waveform values. The normal original adapter also decoded the
captured **standardized generated** latent from the selected-conditioning joint
run: relative L2 **0.071764%**, RMSE 0.0000250496, maximum absolute error
0.000357319. The separate reports are under
`/tmp/lunaflux-minimax-upstream-audio-raw-zero-20261011-v2` and
`/tmp/lunaflux-minimax-upstream-audio-generated-20261011-v1`.
Neither numerical replay establishes intelligibility or semantic media quality.

The schedule correction passes the complete working-tree native suite,
**5,004/5,004**, with unchanged migration warning exclusions. AOT integration
fixtures retain their numerical/source assertions; only plan-dependent
association snapshots change. Separate request preparation and execution tests
exercise four intervals from five points, including the final zero endpoint.

## Original-scale reference rerun and bounded upstream conversion

### Original-reference full-vocabulary capture

The subsequent `v9` original DeepSeek MP2 execution completes on both Sparks
without OOM, with all 36,943 rank tensors loaded and no missing/unexpected keys.
It uses the previously disclosed GB10 sixteen-head launch adapter, not an
unmodified original launch. Literal input IDs match native at all 144 positions.
Both original ranks produce bitwise identical 129,280-value prefill logits.

In the pre-correction capture at that identical prompt frontier, original chooses `33310` (32.385582), while
native's retired physical row 5 chooses `19923` (31.737299). Original scores
`19923` at 31.466326; native scores `33310` at 31.683079. The matched full-vocabulary
RMSE is **0.604927**, relative L2 **11.4684%**, and maximum absolute error
**3.865801**. Original generates `33310,1`; excluding its reporting EOS leaves
**0/1 same-prefix generated tokens matched**. This is a confirmed differential,
not parity. Reference ranks agree with each other; rank disagreement cannot
explain this capture.

Reports and logits are under
`/tmp/lunaflux-deepseek-upstream-reference-20261011-v9`, including
`NATIVE_REFERENCE_LOGITS.txt`. The native capture remains separately at
`/tmp/lunaflux-dspark-run-logits-20261011-v2/logits.f32`.
`scripts/compare-checkpoint-logits.mbtx` selects the explicit physical row and
does not confuse captured capacity rows with the logical prompt frontier.

Source inspection finds an actual precision-contract difference: original
sparse attention rounds each online KV64 block's unnormalized probabilities to
BF16 before P×V, retains an F32 denominator, and adds sink mass after the final
block. The pre-correction native capture uses global F32 probabilities without that BF16
intermediate. This is an actionable arithmetic difference, **not yet a measured
attribution of the full-logit error**. Block boundaries and masked index slots
also matter; a simple final probability cast cannot reproduce online rounding.

The subsequently pushed `4ecd2040` correction explicitly carries the original
online BF16 probability law through precision IR and sparse-attention lowering.
Actual generated component kernels pass correctness and CUDA diagnostic gates.
The real MP2 rerun at `/tmp/lunaflux-dspark-run-probability-20261011-v2` now selects
`33310` (32.257904), matching the original first generated token. Whole-vocabulary
relative L2 moves from 11.4684% to **11.2971%**, but maximum absolute error worsens
from 3.865801 to **4.363262**. Thus this fixes the observed first-token mismatch,
not full numerical parity, EOS equivalence, or multi-prompt correctness. See
`DEEPSEEK_PROBABILITY_PRECISION_2026-10-11.md` for the corrected capture.

The completed original-reference diagnostic partitions initial prefill into valid
one-token continuation calls. This is explicitly a shape ablation, not the
unchanged whole-prompt baseline. Source inspection ruled out six-row continuation:
the original `get_window_topk_idxs` and `get_compress_topk_idxs` use decode-row
read sets whenever `start_pos > 0`, so six-row chunks would change causality,
not merely rounding. Those preparation directories were never run on GPUs.
The optional observer adapter accepts only one-row increments, captures the last
literal prompt frontier, and leaves subsequent decode invocations unchanged.
Both GPU ranks completed with exit zero and identical logit captures. Original
whole-prompt versus one-token prefill differs by **14.4805% relative L2** while
both select `33310`; corrected native versus one-token reference differs by
**11.7744%**. This does not prove that native's six-row execution is numerically
equivalent or that partitioning explains every difference. The separate full
comparison and remaining limits are recorded in the probability-precision report.

The subsequent **native one-row** run completes too, with all 144 literal IDs
and the unchanged six-row physical AOT capacity. Comparing its active row zero
against the original one-token reference still leaves **12.4704% full-vocabulary
relative L2**, RMSE 0.686261 and maximum error 4.995758. Both choose `33310`,
but matching logical prefill partition has not solved numerical equivalence.
This capture requests only one new token, so EOS agreement remains open.

The original-layout GLM scale adapter completed both real TP-rank requests in
`/tmp/lunaflux-independent-glm-20261011-v3`. The 13-token prompt still matches
8/8 native tokens. The 20-token prompt now matches 6/8: its first same-prefix
disagreement is native `13` versus reference `448`. The previous 7/8 result
used the installed backend's collapsed gate/up global scale and must not be
treated as a better or more faithful numerical result. A teacher-forced request
with the original 20 IDs plus the six native generated IDs retains this exact
frontier. Both reference containers are stopped and both GPU owners released.

The subsequent `/tmp/lunaflux-independent-glm-20261011-v4` expert-parallel run
matches **8/8 tokens on each literal input (16/16 total)**. It preserves the
separate original gate/up global scales and uses complete experts, matching
native's expert partition instead of TP-splitting each intermediate sum. Both
reference containers are stopped and GPU owners released. This closes the
observed bounded greedy disagreement; it does not establish full-logit or
all-input numerical equivalence. Persistent sparse-MLA FP8 KV remains a
reference limitation.

Expert decomposition also changes rounding: the ordinary TP reference splits
the intermediate dimension, whereas the native plan partitions complete
experts. The next isolated reference uses `--enable-expert-parallel`, retaining
the same original-scale adapter, literal prompts and memory ceilings. Its
completed with the result above; neither scale preservation nor
expert-parallel startup alone establishes model parity.

DeepSeek's official `inference/convert.py` retains both full rank dictionaries
in RAM. The test-only MoonBit counterpart,
`scripts/convert-deepseek-reference-checkpoint.mbtx`, streams payloads in at
most 1-MiB chunks. It applies the same MP2 row/column split and expert ownership,
views the original packed expert bytes as safetensors `F4`, and performs only
the official block-128 FP8/UE8M0-to-BF16 `wo_a` conversion. It never expands all
experts to BF16 or changes original files. Unit and complete-file fixtures
cover both ranks, nibble preservation, row/column splits, scale blocks and
BF16 ties. The actual checkpoint's metadata gives 36,943 tensors and
85,397,296,888 payload bytes per rank, including 46 `wo_a` conversions and
17,664 packed FP4 tensors. Real conversion runs under a 4-GiB/no-swap CPU unit;
both real conversions completed without OOM. The initial original-CLI attempt
`/tmp/lunaflux-deepseek-upstream-reference-20261011-v1` then failed before
inference because the converter confused Torch's packed-byte shape with
safetensors' logical F4 shape. A separate C++/CPython probe calls the installed
upstream `save_file`/`load_file`: packed `[2,4]` serializes as F4 `[2,8]` with
the **same eight payload bytes**, and loads back to the original Torch shape.
The converter now expresses that distinction and has a regression. The new
`--repair-packed` mode rewrites metadata and streams the unchanged payload into
new rank directories; it never mutates the failed file or original checkpoint.
Both repaired rank files now pass the upstream reader. The bounded original
generation comparison completed with the explicit hardware adapter below.

The original `load_model` also builds an owning CPU tensor dictionary while
the entire device model is already allocated. On Spark's unified memory that
duplicates roughly 80 GB; the `v2` test was stopped before OOM when available
RAM fell to about 18 GiB. The test-only C++ allocation adapter now copies one
named mmap tensor at a time into those same original device parameters, then
releases its CPU view. It leaves original forward/sampling arithmetic unchanged.
A shared-parameter CPU fixture compares its tensor values and alias/missing-key
result against upstream `save_model`/`load_model` exactly. A real rank's first
tensor also passes the independent reader. The attempted pread adapter in `v3`
failed on tensor access before loading; `v4` retains the upstream mmap backend
and progressed through 9,728 tensors before the test was stopped with about
13 GiB available: releasing tensor views did not release the reader's backing
map. A bounded device-copy probe isolates this lifetime: the original persistent
map exceeds a 4-GiB CPU limit before 1,536 tensors; reopening every 128 tensors
keeps resident CPU memory around 632 MiB through all 4,096 probe tensors. The
`v5` loader applies this backing-store lifetime fix while retaining original
Torch copy/conversion operations. Both ranks finish loading **36,943 tensors**,
with zero missing/unexpected keys and approximately 32–34 GiB host memory
available. Neither rank is OOM-killed. The original CLI then fails at its first
sparse-attention launch: Tilelang requests **104,448 bytes** dynamic shared
memory; an independent GB10 device query reports only **101,376 bytes** opt-in
capacity. This is an original-reference hardware launch failure, not a native
token mismatch. Logs remain at
`/tmp/lunaflux-deepseek-upstream-reference-20261011-v5`.

A scheduling-only one-stage probe still requests the same 104,448 bytes and
fails; its separate `v6` artifacts are preserved. The next bounded reference
port instead splits independent attention heads into groups of sixteen. It
retains the entire original Tilelang kernel verbatim, including KV64 online
softmax partitioning, two pipeline stages, sink arithmetic and output dtype.
This hardware adapter changes the wrapper's work decomposition, not weights or
model equations; it is explicitly **not an unmodified original launch**.
The sparse-only launch probe succeeds with finite outputs on Spark `.178`.
The full MP2 reference completed under `v7` on both Sparks with exit zero and
`OOMKilled=false`; both loaded all 36,943 rank tensors and released GPU owners.
Actual input IDs match all 144 native IDs. Its first generated token is
`33310` ("hello"), versus native `19923` ("Hello"). The CLI always appends a
reporting EOS, which is excluded from the one-token comparison. The preserved
command requests eight new tokens and has adequate context capacity; stopping
after one content token implies that the original loop sampled EOS on its next
step. The marker alone is not evidence of this, and the native one-token run
still cannot establish EOS agreement.
Thus **0/1** same-prefix tokens agree. This reproduces the independent backend's
capitalization disagreement, now without its FP8 sparse-KV substitution. It
is a measured unresolved native/reference differential, not a correctness pass.
Commands, logs and `COMPARISON.txt` remain in the separate `v7` directory.
The offline adapter is `scripts/prepare-deepseek-spark-reference.mbtx` and the
runner requires the explicit `--spark-head-tiles` option. Original checkpoint
sources remain unchanged and mounted read-only.

`scripts/run-deepseek-upstream-reference.mbtx` runs the checkpoint's unmodified
`generate.py`, original arithmetic/config fields, MP2 and literal user text,
changing only allocation capacities to one sequence/256 tokens. Its test-only
C++ observer captures the original function's actual input/output token lists;
the comparison requires identical literal inputs and excludes only the CLI's
appended reporting EOS. The observer preserves original exceptions. Runtime
uses the existing CX7 link (`10.0.0.1/2`) and the actual per-node NIC names,
96-GiB/no-swap limits, and no concurrent GPU workload. The CPU observer probe
and original CLI import/help checks pass on both Sparks; these are not GPU
model correctness results.

`scripts/prepare-deepseek-upstream-dependency.mbtx` separately prepares the
missing original fast-Hadamard dependency in an isolated test directory. It
changes only the dependency's setup target list to SM121 because CUDA 13 no
longer compiles SM75; numerical source and serving images are untouched.
The original SM121-targeted Hadamard wheel built successfully. This dependency
build is not a production runtime dependency or a correctness
pass. The upstream reference is required to remove the installed reference's
persistent-FP8-cache and routing-rounding mismatches, not to rerun those
mismatches and call the result faithful.

## Matching reference discovered

Both Spark `.178` and `.179` already contain
`ghcr.nju.edu.cn/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-e3-20260907`.
Despite the image tag, its installed vLLM source contains both `Glm5Next` and
DeepSeek-V4 implementations. The older January SGLang image and local source
trees are not matching references for these architectures.

The new offline MoonBit runner is
`scripts/run-independent-checkpoint-reference.mbtx`. It mounts the original
staged checkpoints read-only, uses two tensor-parallel ranks, literal input
IDs, an eager 256-token context, one sequence and 512 MiB KV allocation.
Each container has a 96-GiB/no-swap limit; no GPU workloads overlap these
references. These are correctness experiments, not comparative throughput.

GLM's original configuration declares FP4 weights and **null input activation
quantization**. The runner translates that to the reference's `W4A16_NVFP4`
mode, preserving the original exclusion list. It does not substitute EXL3
weights or silently introduce FP4 activation quantization. `ABLIT=0` disables
the image's optional load-time weight edits; reasoning stop suppression is
also disabled. No checkpoint file is changed.

## Execution status

The first GLM attempt, `/tmp/lunaflux-independent-glm-20261011-v1`, failed
before weight loading. The concrete cause was Gloo advertising loopback
addresses between two machines, not a numerical discrepancy or OOM. The
runner now explicitly selects their shared `enP7s7` interface for Gloo/NCCL.
The second attempt retains its own commands and results under
`/tmp/lunaflux-independent-glm-20261011-v2`.

The GLM reference now executes both original literal prompts across both real
Sparks. The 13-token prompt matches all eight native generated tokens. The
20-token prompt matches its first seven, then differs: native token `34542`
(" Numbers") versus reference `26682` (" Testing"). Reference top-five scores
include the native token, but its teacher-forced prefill score ranks it 11th.
The teacher-forced and incremental reference scores themselves differ because
they use different execution shapes; neither score is a native logit capture.
Raw completions, log probabilities and prefix reports are retained in `v2`.
Both reference GPU owners are released before the subsequent experiments.

Strict original-arithmetic GLM equivalence remains unresolved for the concrete
reference limitations below; DeepSeek also has a measured differential below.
Model architecture recognition or a running server is not a correctness pass.
Reference requests must use identical literal IDs to the native captures,
retain token/log-probability outputs, and distinguish greedy agreement from
teacher-forced numerical comparison. Existing native cancellation and
retained-weight results remain valid in their narrower scope.

DeepSeek attempt `v1` exits before loading weights: its SM120-family sparse
MLA implementation requires an explicit FP8 KV dtype instead of `auto`.
Attempt `v2` supplies `--kv-cache-dtype fp8` to that independent implementation;
it does not change LunaFlux's KV dtype or checkpoint. Both ranks load the
original 48 shards, with rank zero reporting 73.85 GiB of model memory. Both
reference ranks execute literal requests on the original checkpoint, then exit
zero with `OOMKilled=false` and release both GPU owners.

### DeepSeek independent results

The one-token BOS input reproduces both available native tokens, `5,223`.
The 144-token input uses the exact literal IDs from the native retained-request
cancellation run. Native emits `Hello! How can I assist you today?` and the
reference's eight-token window emits `hello! How can I help you today`.
Their first token differs (`19923` versus `33310`), so later positional matches
are not same-prefix equivalence. At the first token the native choice is the
reference's second candidate, 0.25 logit units below its winner.

A teacher-forced reference receives the original 144 IDs **plus the native
first five tokens**. On this identical prefix it prefers `help` (`1694`) over
native `assist` (`8233`) by 0.75 logit units. The 4.5-unit gap from the ordinary
reference completion used a different capitalization prefix and must not be
used for native arithmetic attribution. Raw JSON and comparisons remain in
`/tmp/lunaflux-independent-deepseek-20261011-v2`.

This is an independent checkpoint differential, not a strict numerical pass.
The SM120-family reference uses mandatory FP8 sparse-MLA KV storage and
DeepGEMM MXFP4 experts; native uses its existing BF16 KV/math route. No native
logit or same-input layer comparison has yet isolated the contribution of
those boundaries. Internal ordinary/speculative agreement does not close this
remaining numerical investigation.

## MiniMax follow-up

The existing independent fifty-layer encoder corpus remains the starting
point, including its larger long-input and digit-row differences. Matched
operator tests already show exact rotary application and very small projection
differences on the original three-token caption. Native arithmetic can be
closer to the wider F64 probe than library BF16 arithmetic; bitwise imitation
of one reduced-precision library path is not automatically a numerical fix.

The separate reference driver now exposes CPU-only `rotary-tables` to check
the F32 frequency/angle and BF16 cosine/sine tables across wider positions.
This isolates table generation from propagated hidden-state differences and
does not add a reference dependency to serving.

The production pure table planner is exercised separately by
`benchmarks/minimax_rotary_probe`. At all positions 0–139, both its cosine
and sine tables match the independently generated CPU ATen tables bitwise:
17,920/17,920 BF16 values in each, with zero absolute and relative error.
This rules out the table-generation hypothesis for the measured 140-token
text case; it does not claim GPU trigonometric equivalence or resolve the
propagated encoder differences. Results are in
`/tmp/lunaflux-minimax-native-rotary-20261011-v1` and
`/tmp/lunaflux-minimax-aten-reference-rotary-20261011-v1`.

Inspection of the actual installed upstream encoder also confirms that it
uses three separate Q/K/V GEMMs and separate gate/up GEMMs despite packed
parameter storage. The independent encoder already follows that operation
order. Switching its oracle to packed GEMMs would change the reference,
not fix the native implementation.

### Exact 140-token operator replay

A fresh native first-layer capture uses the original 140 literal IDs, not the
reference runner's older three-token default. The replay CLI now accepts those
IDs for matched-operand experiments and has a regression checking all 140 are
retained. Native execution exits zero with empty stderr and a 51,861,016,576-byte
user-unit memory peak. The independent replay exits zero and releases its GPU.

The wider independent F64 causal-GQA oracle uses the **same native Q, K and V**:

| Attention result against wider oracle | Relative L2 | BF16 cells identical |
| --- | --- | --- |
| Native | 0.001331% | 1,146,861 / 1,146,880 |
| Independent library BF16 | 0.073522% | 1,115,899 / 1,146,880 |

The first native/library difference is the Q projection. Other matched operator
differences remain small: K 0.047427%, output 0.027177%, MLP activation 0.023342%
and down 0.040891% relative L2. Matched residual additions are bitwise exact.
These are first-layer, same-input results, not a fifty-layer acceptance bound.
In particular they do not support rewriting the native attention math to
imitate the less accurate library result.

The fresh capture and replay are retained at
`/tmp/lunaflux-minimax-text-encoder-operators-long-20261011-v1` and
`/tmp/lunaflux-minimax-aten-reference-long-operators-20261011-v1`.
Wider and per-operator CPU comparisons are at
`/tmp/lunaflux-minimax-attention-precision-long-20261011-v1`.

### Fifty-layer reduction sensitivity

Two additional independent runs disable reduced-precision BF16 GEMM partial
reductions while preserving BF16 operator boundaries, original weights and the
same library attention. All fifty layers execute on the real Spark; these are
not simulated full-encoder results.

* Long input, 140 tokens: default and full-reduction reference final outputs
  match **all 716,800 BF16 cells**. The native/reference final difference remains
  0.9633% globally and 22.1058% on its worst row. The reduction flag alone
  therefore cannot explain or fix this case.
* Digits, 56 tokens: changing only the reference reduction policy changes its
  worst row by 13.8845%. Native versus that wider-reduction route has a worst-row
  difference of 27.9439%, versus 14.7532% with the default route. Switching the
  reference policy is not a native numerical fix.

These results demonstrate input-dependent amplification of small arithmetic
differences, but do not establish its acceptable impact on media quality or
rule out defects in every later layer. MiniMax numerical alignment is still
open. No production arithmetic change has been made without a diagnosed bug.

An additional fifty-layer reference changes only attention to independent
two-pass F64 math and keeps all BF16 projection/operator boundaries. Both real
Spark runs complete and release their owners; the independent native bridge
passes CPU causal-GQA smoke, ASan and LSan. This policy is attribution-only:

| Input | Native/default worst final row | Native/F64-attention worst final row |
| --- | --- | --- |
| Long, 140 tokens | 22.1058% | 48.2269% |
| Digits, 56 tokens | 14.7532% | 25.0560% |

Global native/F64-attention errors are 0.8759% and 0.6076% respectively.
Replacing the attention precision alone therefore does **not** resolve the
model differential, despite native first-layer attention's accurate same-input
result. Outputs and all 51 states are downloaded under
`/tmp/lunaflux-minimax-aten-reference-wide-attention-20261011-v1` (long) and
`v2` (digits). No acceptance tolerance is manufactured from these measurements.

## Native-logit localization

The explicit DeepSeek `egress-logits-dspark` command borrows the existing head
allocation, prepares a readback outside execution and copies the allocated
vocabulary rows only after the control session ends. Its runner's `logits` mode
requests one generated token, preserving the same literal prefill. Ordinary
generation prepares no additional pool and performs no additional readback.
This supports comparing native margins against reference scores instead of
guessing from decoded wording. The offline F32 inspector reports native
top-five scores and selected-token margins; separately rounding that head row
to BF16 is labeled an attribution-only view, not another runtime route.

The first physical capture completes with both ranks exiting zero and releasing
their contexts. It reveals an **observer bug**, not a model conclusion: copying
row zero alone cannot reproduce the emitted token when the prepared head has
multiple output rows. Its raw scores are retained in
`/tmp/lunaflux-dspark-run-logits-20261011-v1`, but must not be compared to the
first-generation reference scores. The correction copies the entire allocated
head after retirement. The offline inspector now requires an explicit output
row and checks that its greedy winner equals the actual emitted token before
reporting margins. A two-row regression rejects the misleading row-zero case.

The corrected run, `/tmp/lunaflux-dspark-run-logits-20261011-v2`, uses committed
`c8c64721437bbdfae9c9d53512224d791a761832`, the unchanged six-row AOT kernels and
the original 144 literal IDs (compared byte-for-byte before execution). The full
allocated head is downloaded. **Physical row 5 reproduces the actual emitted
token**, making this a usable native first-generation score capture:

| Implementation / diagnostic view | `Hello` (19923) | `hello` (33310) | Winner |
| --- | --- | --- | --- |
| Actual native F32 head | 31.737299 | 31.683079 | `Hello`, margin 0.054220 |
| Native head rounded to BF16, diagnostic only | 31.750000 | 31.625000 | `Hello`, margin 0.125000 |
| Independent reference, same literal input | second candidate | first candidate | `hello`, margin 0.250000 |

Absolute reference logits are not inferred from log probabilities. Their
pairwise margin is comparable; these opposite signed margins differ by about
0.3042 logit units. BF16 rounding of the captured native head alone does not
explain the winner reversal. The original checkpoint's `ParallelHead` explicitly
converts both its input and stored BF16 weight to F32 computation; changing
native head precision merely to imitate the serving reference is not a fix.
The remaining arithmetic attribution must occur before the final head, not in
the now-corrected row observer.

Both ranks exit zero, explicitly close their contexts and leave no GPU owners.
User-unit memory peaks are 58,820,612,096 bytes on `.178` and 37,478,076,416 bytes
on `.179`, below their 96-GiB limits, with zero unit swap peaks. These cgroup
figures are not complete CUDA unified-memory residency or a zero-host-swap claim.
The 69,114-ms generation window includes all 24 prefill frames and one generated
token; it is not steady-state token throughput. Strict independent equivalence
remains unresolved because of the reference arithmetic differences below.

The next two MiniMax interventions have now completed all fifty layers on the
real Spark for the original 56-token digits input. Both preserve BF16 operator
boundaries, original weights and literal IDs. All 51 states from each run are
downloaded under `/tmp/lunaflux-minimax-aten-reference-wide-projection-20261011-v2`.
Both containers exit zero, release their retained tensors and GPU owners, and
report no OOM. These are arithmetic attribution tests, not production fixes.

| Reference policy | Native/final global relative L2 | Native/worst final row relative L2 |
| --- | --- | --- |
| Original BF16 | 0.5081% | 14.7532% |
| F64 projection, original attention | 0.2901% | 14.9842% |
| F64 projection and F64 attention | 0.3371% | 18.2173% |

The worst row remains row 23. Wider projections lower the magnitude-weighted
global norm without fixing that row; widening attention as well increases its
difference. Neither result supports claiming that a projection/attention
precision switch fixes MiniMax. The earlier same-input layer replay still
distinguishes small local operator differences from substantially larger
propagated input sensitivity. Numerical alignment and media-quality impact
remain unresolved; no production arithmetic is changed to manufacture parity.

## GLM reference arithmetic caveat

The installed reference warns that gate/up global FP4 scales differ. Its
`ModelOptNvFp4FusedMoE.process_weights_after_loading` then selects
`w13_weight_scale_2[:, 0]` for both projections. LunaFlux retains the original
separate gate/up/down scale planes. Consequently an unmodified reference
completion is useful as a differential but cannot certify exact original-
checkpoint arithmetic. Its sparse MLA backend also selects FP8 KV storage.
These differences must be reported, not silently treated as native bugs or
fixed by altering the original checkpoint.

The original DeepSeek checkpoint's `inference/model.py`, `Expert.forward`, also
confirms `weights * (silu(gate) * up)` **before** conversion to BF16 and the down
projection. LunaFlux's immutable precision plan preserves that order. The
installed vLLM DeepGEMM FP4 expert performs activation quantization/down first,
then `deepgemm_unpermute_and_reduce` applies the routing weights. FP8 requantization
and BF16 rounding make these orders observably different. This is another
explicit reference arithmetic incompatibility, not permission to replace the
original native numerical contract to force token agreement.

The original DeepSeek `Attention.forward` also makes the KV boundary precise:
it applies rotary to the 64-lane suffix, simulates block-64 FP8 quantization on
the 448 non-rotary lanes, and publishes the result to a **BF16 persistent cache**.
Its comment explicitly distinguishes this simulation from future FP8 storage.
LunaFlux's `RotaryKvPrecision` preserves those boundaries; BF16 storage alone
does not imply that the quantization simulation was omitted. The installed
serving reference instead requires persistent FP8 sparse-MLA KV on this device.
The GLM scale collapse was independently reconfirmed in installed
`modelopt.py:1530–1555`: it happens before backend-format conversion, so merely
selecting another kernel backend cannot restore the discarded up scale.

## Source validation

### Original-layout GLM scale adapter

`scripts/prepare-glm-scale-reference.mbtx` now builds a separate test-only
ATen expert path that retains both original gate/up global scale planes and
does not perform the reference's Marlin weight permutation. It decodes only
the currently selected expert, with a conservative 2-GiB temporary bound,
instead of expanding the complete checkpoint to BF16. Routing and the rest
of the reference model remain external to this adapter.

CPU and real `.179` GPU probes pass. Fixtures cover all E2M1 magnitudes/signs,
distinct gate/up scales, selected-token routing and explicit BF16 boundaries;
deliberately collapsing the up scale changes the result. The CPU native probe
also passes ASan/LSan. Results are under
`/tmp/lunaflux-glm-scale-reference-20261011-v2`. The failed first schema
registration attempt remains separately preserved in `v1`.

A two-Spark original-checkpoint run is loading under
`/tmp/lunaflux-independent-glm-20261011-v3`, with the same original literal
prompts, eager context and 96-GiB/no-swap limits. This corrects the known scale
reference defect, not the sparse-MLA KV-storage difference. No full-model
numerical pass or production change is claimed before its outputs are compared.

### MiniMax normalization-only intervention

Two real Spark runs replace only the row/head RMSNorm square summation with
compensated F32 addition. Original weights, literal tokens, scalar launch
geometry, projections, attention and BF16 operator boundaries remain unchanged.
Both fifty-layer executions exit zero with empty execution stderr, download all
51 hidden states and leave no GPU owner. The original independent reference is
unchanged; neither library arithmetic nor its acceptance threshold is adjusted.

| Original input | Original native worst final-row relative L2 | Compensated native worst final-row relative L2 | Compensated final global relative L2 |
| --- | --- | --- | --- |
| Digits, 56 rows | 14.7532% (row 23) | 3.1238% (row 17) | 0.4532% |
| Long input, 140 rows | 22.1058% (row 5) | 9.0533% (row 6) | 0.7959% |

Digits row 23 specifically improves to 2.3349%. Reporting the new worst row is
important: that improvement alone must not be presented as the whole tensor's
error. The experiment isolates lost small square contributions in the original
ordered F32 accumulator as a substantial contributor, not the sole cause of
all numerical differences. Remaining propagated differences are still open.

The separate experimental AOT directories are
`/tmp/lunaflux-minimax-request-aot-compensated-{digits,long}-20261011-v2`;
native captures and final reports are under the corresponding
`/tmp/lunaflux-minimax-text-encoder-compensated-{digits,long}-20261011-v2`.
Layer reports are in `/tmp/lunaflux-minimax-compensated-{digits,long}-series-20261011-v2`.
The reused experiment manifest supplies original geometry/symbols only; it is
not proof of original source identity. These artifacts are separate from the
original AOT and are not a source-selected production validation.

The source fix expresses `SquareReductionLaw` in the backend-neutral precision
IR. The immutable MiniMax text plan selects compensated F32; CUDA lowering
consumes it in both row and Q/K-head normalization. Generic callers retain the
explicit ordered-F32 default. Compensation does not widen storage or move the
BF16 normalize-before-weight boundary. There is no token-path configuration,
new allocation, hashing, TLS or additional hardware-specific model branching.
Fresh ARM build of pushed `49b15d75f8678cf6177d30b229258dabf62e2fb8` passes on
Spark `.178` (208 build tasks, 588.1 MiB peak, zero build swap). Unmodified
source exports then compile exact 56/140-row AOT modules. Fifty-layer execution
on `.179` reproduces both experimental final tensors **bitwise**, confirming
the selected source policy actually reaches the runtime. The source-selected
corpus is retained in `/tmp/lunaflux-minimax-source-norm-20261011-v2`; the initial
`v1` failed before inference because the runner binary was not staged on the
execution host. That receipt is retained, the runner is corrected, and `v2`
stages the committed binary explicitly before execution.

The actual row/head CUDA renderer also passes a bounded independent F64 oracle
(zero/constant/outlier/mixed inputs, eight deterministic runs, at most one BF16
step at reciprocal-square-root publication). Memcheck reports zero errors and
zero leaked allocations; racecheck reports zero hazards; synccheck reports zero
errors. These are component correctness checks, not a whole-model tolerance.
Results are in `/tmp/lunaflux-text-norm-compensated-20261011-v1`.

Four alternating, unsanitized isolated row-normalization timing trials measure
283.1–284.7 microseconds ordered versus 311.4–312.0 microseconds compensated
(three 5,120-lane rows). Compensation adds about 10% in this scalar kernel;
it is an accuracy fix, not a performance claim. Instrumented sanitizer timing
is excluded. The original unchanged F64-projection reference also corroborates
the digits improvement: final global relative L2 is 0.1943% instead of 0.2901%,
and worst final row is 4.5235% instead of 14.9842%. Remaining error is not hidden
by global norms or declared a pass.

The complete source-selected twelve-case corpus is now finished, with all
native units exiting zero and releasing their GPU. Eight worst-row comparisons
improve and four worsen against the unchanged library reference; the complete
table, including regressions, is in
[the diverse-input report](MINIMAX_DIVERSE_TEXT_ENCODER_2026-10-11.md).
The more accurate component reduction is not evidence that every propagated
library differential must decrease. The completed English/digits/long
row-only/head-only interventions show nonlinear interaction, not one universal
bad kernel; the full matrix is in the diverse-input report. No prompt-specific
precision rule is introduced.

### Fresh complete MiniMax media execution

The committed `49b15d75` ARM binary and freshly exported, unmodified 31-module
request AOT bundle execute the corrected three-row `A red cat` conditioning on
Spark `.178`. The original FL2VA weights, geometry and seed 17 are unchanged.
The request executes five denoising evaluations through all fifty blocks,
then both actual video/audio VAEs. It exits zero, with empty execution stderr,
zero unit swap and no GPU owner at the terminal observation. Its cgroup memory
peak is 44,880,248,832 bytes, not complete CUDA unified-memory residency.
Planned aggregate memory remains 81,920,540,616 under the 96-GiB device budget.

Requested 120 frames resolve to **124 decoded frames** by the existing temporal
shape contract, not an accidental tail or a newly changed frame rule. Output
contains 380,928 finite video F32 values (32 x 32 x 124 x 3) and 331,200 finite
stereo audio values (165,600 samples/channel). Start/exit times are 09:17:28 /
09:46:52 CST. This single 29m24s correctness lifetime includes preparation and
is not a comparative throughput or speedup claim.

The new arrays are downloaded to
`/tmp/lunaflux-minimax-joint-compensated-download-20261011-v1`;
terminal logs/arguments are under
`/tmp/lunaflux-minimax-joint-compensated-20261011-v1/terminal`.
The offline F32 comparator passes its regression and an identity check on the
previous real audio output. It measures the following change from the prior
native media run, not independent media quality:

| Array | RMSE | Relative L2 to prior native output | Maximum absolute difference |
| --- | --- | --- | --- |
| Video | 0.256252 | 72.0499% | 0.963326 |
| Audio | 0.0377942 | 108.1699% | 0.277255 |

Neither array has bitwise-equal F32 elements. Such a large downstream change
must not be called an improvement just because encoder component arithmetic
is more accurate. The previous actual-caption encoder output matches the
unchanged twelve-case baseline bitwise. Corrected/old conditioning differs by
0.0337185% global relative L2 (5,815/15,360 BF16 elements exact); that small
global measure does not establish downstream stability. A fresh control uses **the new
binary and AOT but that old conditioning tensor**, with the same weights,
geometry, steps and CLI seed. It is retained separately at
`/tmp/lunaflux-minimax-joint-old-conditioning-control-20261011-v1`.
The control completes 09:49:32–10:19:08 CST with exit 0, empty stderr,
zero unit swap and released GPU ownership. Its video/audio still differ from
the old run by 72.0318% / 108.2589% relative L2. Changing conditioning within
the same new runtime changes video/audio by only 0.2263% / 5.9088% relative L2.
The prior large media change therefore cannot be assigned to normalization.
The later global GPU-owner snapshot in the first run's collector can contain
this control's PID; it is not a leak from the already completed first unit.

### Initial-noise identity coupling found and corrected

The old RNG mixed the entire model-plan label into the explicit seed. The
rounded audio-output publication fix changed that label even though video/audio
latent shapes and denoising kernels stayed unchanged. Comparing the two AOT
directories finds only the unused `text` module different; all thirty media
modules match. Both control runs consume identical old conditioning bytes.

A CPU-only original-config replay at declared label `d2dd0c60...81d0427`
reproduces different initial noise at CLI seed 17: old plan
`74401f8c...7251e648` resolves to effective key -5814885994241873734;
new plan `49ff4861...c9220e16` resolves to -4047618836202750322. None of the
3,552 video or 13,248 audio initial F32 cells match. This is a concrete
experiment-comparability/reproducibility defect, not an attention numerical
failure or quality result.

RNG v2 now depends only on explicit seed, modality stream and counter. Both
production and independent scalar reference drop the plan-label key mixing.
Typed model/storage association remains unchanged; this adds no hashing,
authentication, dtype change or token-step work. Regression tests exercise
different plan/content labels and audio publication laws, negative seeds,
disjoint streams and chunk-independent values. The new regression fails
against original `b07c0313` and passes after the fix. Supplying the captured
old effective key as the v2 explicit seed reproduces **every initial F32 cell
bitwise**. Historical v1 seed values alone are not compatible with v2.

`benchmarks/joint_noise_probe` reads only small configuration and records this
offline attribution; it never loads weights or opens CUDA. The same-new-runtime
conditioning contrast above is valid because both runs used the same v1
effective key.

### Matched-noise full-media replay completed

The fresh committed `d9e5c996` ARM binary now completes the full request using
the captured historical effective key as its explicit v2 seed and the exact
old conditioning bytes. This is not a seed search: the key was derived from the
original v1 plan before replay. The thirty unchanged media AOT modules and
original checkpoint remain unchanged. Five evaluations through all fifty
denoising blocks, then both actual VAEs, finish successfully on Spark `.178`.

| Decoded array against historical native run | F32 cells | Bitwise identical | RMSE / relative L2 / maximum error |
| --- | --- | --- | --- |
| Video | 380,928 | 380,928 | 0 / 0 / 0 |
| Stereo audio | 331,200 | 331,200 | 0 / 0 / 0 |

Both byte comparisons also return zero. This establishes that the earlier
72.03% / 108.26% old-conditioning output change came from the rerolled initial
noise in this fixture, not a changed denoiser/VAE arithmetic path. It is native
historical reproducibility, **not independent upstream media correctness**.
The same-new-runtime conditioning contrast remains 0.2263% video / 5.9088%
audio relative L2; the four encoder corpus regressions and media-quality
acceptance are still open.

The unit starts/exits at 10:27:55 / 10:56:47 CST (28m52s, including preparation;
not a throughput measurement). Exit status is zero, execution stderr is empty,
all outputs are finite, unit swap peak is zero, and the terminal GPU-owner
snapshot is empty. Cgroup memory peak is 20,477,472,768 bytes, not complete
CUDA unified-memory residency; planned aggregate remains 81,920,540,616 bytes
under the 96-GiB device budget. Terminal records are under
`/tmp/lunaflux-minimax-joint-stable-noise-old-conditioning-20261011-v1/terminal`.
Both decoded arrays and zero-error reports are downloaded without overwrite to
`/tmp/lunaflux-minimax-joint-stable-noise-old-conditioning-download-20261011-v1`.

At pushed `c77c724c`, the current working-tree native check and full suite pass
with the unchanged migration warning settings: **4,983/4,983** tests. This
includes both plan-label and actual audio-publication-law seed regressions.
This does not supersede the clean-archive TLS limitation below or convert an
independent full-model differential into a numerical acceptance result.
The clean `c77c724c` archive separately passes 11/11 production iteration tests
and 7/7 scalar reference tests. Formatting and the output/terminal/comparison
script regressions also pass. Later commits only format these golden tests and
update this ledger; GLM/DeepSeek execution sources are unchanged from the
corrected `c8c64721` physical capture.

The full native suite passes 4,981/4,981 tests, and the root native check passes
with the existing toolchain migration warning exclusions
`-20-25-29-35-79-92`. `moon info` completes with zero errors and the existing
3,646 migration warnings. The separate reference module passes its eight tests;
the production rotary planner's nine tests pass. No warning exclusion, runtime
authentication, TLS or cryptographic scan was added. The new explicit
normalization precision policy above is the measured numerical correction;
it is not a hardening gate or an independent whole-model acceptance claim.

A separate clean archive of pushed `31196b5f` passes **93/93** native tests for
precision IR, the text CUDA renderer, MiniMax text plan/integration and the
MiniMax checkpoint command. This removes reliance on unrelated local changes
for the affected-package claim. Whole-repository checking of that clean archive
is separately blocked at the unchanged `runtime/remote_tls/channel.mbt:174`:
the pinned async dependency marks `Tls::server` internal-only, promoted to an
error under the existing warning-denied configuration. The working-tree root
result above includes an existing uncommitted TLS migration, whereas these
93 tests do not. That unrelated migration is not staged into the numerical
commits, no alert exclusion is added, and these model executions do not use TLS.

### Compensated-reduction F32 overflow correction

A new boundary probe reproduces a concrete defect in the initial compensation:
finite BF16 values can have finite squares whose F32 sum overflows, or squares
that already overflow. The initial `(next - sum) - adjusted` correction then
becomes NaN and poisons the accumulator. On Spark `.179` the actual old renderer
fails with `compensated-row index=0 actual=nan expected=0`, whereas the ordered
F32 renderer returns zero after the reciprocal of an infinite sum.

The pure precision law now explicitly retains F32 overflow semantics. CUDA
lowering discards compensation when the updated sum is nonfinite; the sum
itself is preserved, so original NaN inputs still propagate. There is no
clamping threshold, dtype widening, model-specific rule or runtime validation
step. Row/head kernels consume the same numerical law as before.

The actual fixed lowering passes two overflow patterns (2^61 and 2^80 values),
eight deterministic F64 component cases, memcheck (zero errors/leaked bytes),
racecheck (zero hazards) and synccheck (zero errors). Native affected-package
tests pass 74/74. Old/fixed probe captures remain separately at
`/tmp/lunaflux-text-norm-overflow-{before,fixed}-20261011-v1`.
The small probe runs on `.179` while the isolated media control owns `.178`;
sanitizer timing is not performance evidence. Unsanitized row timing is
284.5–285.6 us ordered versus 323.4–324.6 us compensated. This is a correctness
correction, not a speedup or a fix for all twelve propagated corpus differences.
The two full-media runs above use the previously built `49b15d75` binary/AOT;
they must not be described as including this later overflow correction.

The extended actual-source probe also passes explicit NaN/infinity propagation
in row and Q/K-head normalization, with the same ordered-F32 classifications.
Memcheck/racecheck/synccheck remain clean and all allocations are released.
Those captures are under
`/tmp/lunaflux-text-norm-overflow-special-values-20261011-v1`. This tests
arithmetic semantics rather than adding production input validation.

Pushed `5ccd4147` also builds the ARM command from a fresh archive on Spark
(208 tasks). Its exporter generates unmodified exact-row AOT for **all twelve**
original inputs. Fifty-layer execution completes every case on `.179`, with
finite final tensors and released GPU owners. All twelve final tensors match
the previous compensated `49b15d75` corpus **bitwise**; the source fix therefore
corrects the overflow defect without changing the measured in-range corpus.
The eight-improved/four-regressed comparison remains unchanged, not closed by
this additional fix. Results are split between
`/tmp/lunaflux-minimax-source-norm-overflow-20261011-v1` (caption/digits/long)
and `/tmp/lunaflux-minimax-source-norm-overflow-remainder-20261011-v1`
(the remaining nine original inputs). Both `SOURCE.txt` files name the actual
committed build and exporter; the retest script now permits explicit selection
instead of always exporting from the old hardcoded build.

A second clean archive, now of `5ccd4147`, passes the same **93/93** affected
native tests. The current working-tree native check and complete **4,981/4,981**
suite also pass with unchanged migration warning settings. The existing clean
whole-repository TLS alert limitation above is not resolved or hidden by these
affected-package and working-tree results.
