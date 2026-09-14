# Fused ingress tail/barrier regression

This optional CUDA diagnostic exercises the production fused ingress source,
not a replacement kernel. Export the four fixtures with
`LUNA_TEST_EXPORT_INGRESS_BARRIERS=1 moon test kernels/luna_cuda_fused_parallel_aot/qwen_barrier_wbtest.mbt --target native --deny-warn`.
The marked sources cover head dimensions 16/32/64/128 and a 32-token envelope.
Save each marked source unchanged as its own `.cu` file using MoonBit tooling.

Compile this probe once per source with CUDA, passing
`-DKERNEL_SOURCE='"/absolute/path/to/exported.cu"'`, `--fmad=false`, and the
target architecture. Select the intended GPU through `CUDA_VISIBLE_DEVICES`.
Each executable checks 1/2/7/8/15/16/17/31/32 tokens, including wholly and
partially occupied matrix tiles, and releases all allocated buffers.
Run each under memcheck, racecheck and synccheck with a nonzero
`--error-exitcode` and an external timeout. Orchestration belongs in `.mbtx`.

The structured input/weight pattern has an independently calculated projection,
normalization and RoPE result; all output elements must be finite and within
0.02, and every written KV entry must exactly match the corresponding output.
This catches inactive-warp participation errors without relying on another GPU
kernel as oracle. It is not random-weight model qualification, a token-agreement
test, or a serving benchmark. Keep those broader gates separate.

The original warp-dependent column loop has block barriers inside it. At
dimension 32, half the warps skip that loop. Restoring it reproduces divergent
barriers under synccheck; masking only the independent matrix operations within
uniform rounds removes that error. Source-shape regression lives alongside the
export fixture, so no CUDA installation is required for normal native tests.
