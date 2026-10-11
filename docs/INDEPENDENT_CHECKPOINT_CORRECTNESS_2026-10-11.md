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

## GLM reference arithmetic caveat

The installed reference warns that gate/up global FP4 scales differ. Its
`ModelOptNvFp4FusedMoE.process_weights_after_loading` then selects
`w13_weight_scale_2[:, 0]` for both projections. LunaFlux retains the original
separate gate/up/down scale planes. Consequently an unmodified reference
completion is useful as a differential but cannot certify exact original-
checkpoint arithmetic. Its sparse MLA backend also selects FP8 KV storage.
These differences must be reported, not silently treated as native bugs or
fixed by altering the original checkpoint.

## Source validation

The full native suite passes 4,978/4,978 tests, and the root native check passes
with the existing toolchain migration warning exclusions
`-20-25-29-35-79-92`. `moon info` completes with zero errors and the existing
3,646 migration warnings. The separate reference module passes its eight tests;
the production rotary planner's nine tests pass. No warning exclusion, runtime
authentication, TLS, cryptographic scan or production numerical policy was added.
