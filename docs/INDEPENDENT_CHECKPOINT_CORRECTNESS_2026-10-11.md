# Independent GLM / DeepSeek checkpoint correctness — 2026-10-11

Internal ordinary/speculative agreement and successful resource cleanup are
not independent numerical correctness. This work keeps that distinction and
does not add production authentication, hashing, TLS or release prerequisites.

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

## Source validation

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

The full native suite passes 4,979/4,979 tests, and the root native check passes
with the existing toolchain migration warning exclusions
`-20-25-29-35-79-92`. `moon info` completes with zero errors and the existing
3,646 migration warnings. The separate reference module passes its eight tests;
the production rotary planner's nine tests pass. No warning exclusion, runtime
authentication, TLS, cryptographic scan or production numerical policy was added.
