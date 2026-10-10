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

`encode-operators ... 1` saves sixteen first-layer operator outputs.
`replay-operators LIB MODEL IDS OUTPUT NATIVE_OPERATOR_DIRECTORY` substitutes
the corresponding native input at each independent operation; its result is an
attribution trace, **not an end-to-end hidden state**. Norm, rotary, attention,
residual and projection errors are separated from propagated input differences.
The fused native gate/up/SiLU operation is compared as one complete chain.

`projection-oracle LIB MODEL NATIVE_OPERATOR_DIRECTORY OUTPUT` compares
original first-layer Q/K/V, output and down weights on identical three-row native inputs
using library BF16, BF16 with reduced-precision reductions disabled, F32 and
F64 projections, then rounds each output to BF16. The test-only full-reduction
scope restores the library setting, including split-K, after success or failure;
it does not change the ordinary upstream arithmetic reference or serving.
Converted operands and temporary output are checked against the same explicit
2-GiB budget before allocation. Wider precision is an independent numerical
probe, not an upstream BF16 performance result or a proposed production backend.

`encode-full-reduction ... 50` applies that test-only projection arithmetic to
all seven linears in each of the fifty encoder layers. Norm/rotary/SDPA/residual
semantics and original checkpoint inputs stay unchanged. All 51 layer outputs
are collected through `collect-full-reduction`; this separately labeled run
attributes accumulated rounding differences and does not replace the ordinary
upstream BF16 reference or define an acceptance tolerance.

`replay-full-reduction` uses the same matched native operands at every first-layer
boundary while disabling reduced-precision projection reductions. Collect with
`collect-full-replay`. This isolates residual operator differences without
changing either production arithmetic or the ordinary reference. It is still
an attribution trace, not a complete encoder output.
