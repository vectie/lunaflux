# Independent checkpoint arithmetic reference

This is a **separate test-only MoonBit module**. The production module does not
import it and acquires no ATen/PyTorch dependency. The driver owns original
checkpoint metadata, token IDs, layer order and deterministic tensor scopes.
The private dynamic ABI delegates independent arithmetic to the ATen library
already installed in the reference container. No Python automation, checkpoint
payload hash or production admission step is introduced.

The initial diagnostic covers text-only MiniMax-H3's original Qwen3-VL encoder:
separate Q/K/V and gate/up BF16 linear operations, staged BF16 RMSNorm/rotary
rounding, causal GQA SDPA, and the **unnormalized** state after decoder layer 49.
It follows the installed upstream `vllm_omni/.../minimax_h3/encoder.py` math,
but is not a claim of running the complete upstream H3 pipeline. Visual
conditioning, media quality and the joint denoiser are outside this diagnostic.

Only requested embedding rows and one projection weight region are uploaded
at a time. Retained reference tensor storage is capped at 2 GiB, with external
no-swap container limits still required to bound ATen workspace/caches and host
staging. Original BF16 tensors are read directly from safetensors offsets;
LunaFlux packed layouts, CUDA kernels and rotary tables are not reused.

Every layer's raw output is written without overwrite for differential
localization. Successful completion requires all retained tensor owners released.
These startup/offline diagnostics are never part of serving or token steps.

The separate `aten/nvfp4_reference.cpp` adapter covers GLM's original row-major
NVFP4 expert operands. It retains independent gate/up global scales, decodes
only selected experts, and preserves explicit BF16 operation boundaries. Its
CPU/GPU probe checks all E2M1 magnitudes/signs and demonstrably detects an
incorrectly collapsed up scale. The `.mbtx` preparer creates a test-only
read-only vLLM numerical-method overlay; it neither changes checkpoint files
nor introduces a production dependency. A corrected expert reference is not
yet a claim of identical attention/KV arithmetic or full-model equivalence.

Build the MoonBit driver with `moon build driver --target native --release`.
Build `aten/ops.cpp` as a shared library with the existing reference image's
ATen headers/libraries; the runner supplies that library path explicitly.
Run the bounded CPU ownership/arithmetic smoke first:

```
driver.exe self-test LIBRARY NEW_OUTPUT_DIRECTORY
driver.exe encode LIBRARY ORIGINAL_TEXT_ENCODER_ROOT 32,2518,8251 NEW_OUTPUT_DIRECTORY 50
```

The current two-Spark installation can reproduce the independent diagnostic
with `scripts/run-minimax-aten-reference.mbtx`: `stage` builds the ARM driver
and installed ATen library into a fresh evidence root; `self-test` exercises
the CPU ABI; `asan` compiles/runs the native ownership probe with Linux ASan and
LSan; `encode` executes the original caption; `collect` downloads all 51 layer
outputs without overwrite. Existing SSH control masters and the original
staged text checkpoint are explicit installation inputs to that runner.

Do not infer parity from a finite output. Compare against retained native
outputs, report measured errors and distinguish reduction-order differences
from missing semantic operations. No model-specific hardcoded tolerance is
used to turn a mismatch into a pass.

`replay-layer LIB MODEL ROWS NEXT_HIDDEN_INDEX INPUT_BF16 OUTPUT` executes one
complete layer on a named previous hidden-state frontier (1..1,024 rows,
next state 1..50). It captures the unchanged input, fifteen operator boundaries
and that layer's output. A reference replay on reference input must reproduce
the corresponding propagated reference layer before interpreting native-input
results. This separates local arithmetic differences from input sensitivity;
it is not a fifty-layer encoder result. The bounded `.mbtx` layer runner accepts
a new run label to keep native-input and reference-input outputs separate.

`encode-operators ... 1` saves sixteen first-layer operator outputs.
`replay-operators LIB MODEL IDS OUTPUT NATIVE_OPERATOR_DIRECTORY` substitutes
the corresponding native input at each independent operation; its result is an
attribution trace, **not an end-to-end hidden state**. Norm, rotary, attention,
residual and projection errors are separated from propagated input differences.
The fused native gate/up/SiLU operation is compared as one complete chain.

`projection-oracle LIB MODEL NATIVE_OPERATOR_DIRECTORY OUTPUT` compares
original first-layer Q/K/V, output, gate/up and down weights on identical three-row inputs
using library BF16, BF16 with reduced-precision reductions disabled, F32 and
F64 projections, then rounds each output to BF16. The test-only full-reduction
scope restores the library setting, including split-K, after success or failure;
it does not change the ordinary upstream arithmetic reference or serving.
Converted operands and temporary output are checked against the same explicit
2-GiB budget before allocation. Wider precision is an independent numerical
probe, not an upstream BF16 performance result or a proposed production backend.

`layer-projection-oracle LIB MODEL ROWS NEXT_HIDDEN_INDEX INPUT_OPERATORS OUTPUT`
uses the same seven projections with an explicit 1..1,024-row input and layer
1..50. `scripts/probe-minimax-layer-projections.mbtx` runs it sequentially within
the existing reference memory limits, with a new output label for each input.
`scripts/compare-minimax-projection-oracles.mbtx` compares precision policies on
identical operands separately from changing the input under the same policy.
Its `all`/`mlp` groups report global and worst-row errors, not parity tolerances.

`activation-factorization LIB ROWS WIDTH FIRST_PROJECTIONS SECOND_PROJECTIONS OUTPUT`
recombines captured gate/up operands using the unchanged reference SiLU/product:
both original pairs and the two cross-input pairs. It repeats those four cases
for BF16 and F64 **preceding projection** outputs; every operand remains BF16,
so this is not an F64 activation kernel. The existing projection runner accepts
`activation BUILT_REFERENCE_ROOT FIRST_PROJECTIONS SECOND_PROJECTIONS ROWS WIDTH NEW_LABEL`
and collects eight products under the same 2-GiB tensor / 8-GiB no-swap limits.
Original pairs must reproduce the earlier reference activations before interpreting
hybrids. `scripts/inspect-bf16-activation-error.mbtx` reports the largest-error
channel and its squared-error share for an explicitly selected observed row.
These are offline attribution tools, not serving dependencies or acceptance gates.

`replay-gate-input LIB MODEL LAYER ROW CHANNEL FIRST_OPERATORS SECOND_OPERATORS`
is CPU-only. It reads one original gate weight row and the corresponding two
captured normalized-input rows. Pure F64 dot-product attribution reports signed
and absolute operand contributions, cancellation ratios and the largest changed
term. It preserves BF16 input/weight boundaries rather than introducing a new
production precision policy. The existing projection runner's `gate` mode runs
this under a 1-GiB/no-swap limit without `--gpus`; a hidden-channel trajectory is
available through `inspect-bf16-activation-error.mbtx trace-hidden`. A trajectory
alone is not causal attribution: identical-input layer replay is still needed.

`encode-full-reduction ... 50` applies that test-only projection arithmetic to
all seven linears in each of the fifty encoder layers. Norm/rotary/SDPA/residual
semantics and original checkpoint inputs stay unchanged. All 51 layer outputs
are collected through `collect-full-reduction`; this separately labeled run
attributes accumulated rounding differences and does not replace the ordinary
upstream BF16 reference or define an acceptance tolerance.

`encode-wide-attention ... 50` changes **only attention** to an independent
two-pass F64 score/causal-softmax/value implementation, rounding its result to
BF16 at the original boundary. Projection, RMSNorm, rotary, residual and MLP
arithmetic remain the upstream BF16 reference. Collect all 51 states with
`collect-wide-attention`. This isolates propagation of attention-rounding
differences; it is neither a new production precision policy nor the upstream
BF16 reference. The CPU smoke checks both constant-value attention and causal
GQA with distinct value rows. The transient wider allocation estimate includes
score/probability copies and repeated KV heads before allocating them.

`encode-wide-projection ... 50` instead widens only the seven projection
accumulators to F64, then returns to BF16 at each original operator boundary.
`encode-wide-linear-attention ... 50` widens both projections and attention.
Collect with the corresponding `collect-wide-*` modes. These two interventions
separate projection error propagation from attention error propagation using
the same original weights and literal IDs; neither changes ordinary reference
arithmetic or production serving. They reuse the already tested wider linear
and causal-GQA primitives and their allocation budgets.

`replay-full-reduction` uses the same matched native operands at every first-layer
boundary while disabling reduced-precision projection reductions. Collect with
`collect-full-replay`. This isolates residual operator differences without
changing either production arithmetic or the ordinary reference. It is still
an attribution trace, not a complete encoder output.

`replay-down-reduction LIB MODEL NATIVE_INPUTS PRECISION_ORACLE` is CPU-only.
It reads only original first-layer down weight rows whose native output differs
from the wider probe, replays separately rounded ordered F32 and F64 sums, and
compares their BF16 results to the respective recorded outputs. It must not
acquire CUDA or become an encoder acceptance tolerance. Functional loop state
and explicit F64 intermediates preserve F32 rounding without host FMA contraction.
