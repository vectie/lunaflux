# Selected expert MLP CUDA source

This package renders a family-neutral serial correctness kernel for selected
SwiGLU experts. It validates distinct in-range indices, computes experts in
ascending expert-id order, rounds each weighted expert contribution to BF16,
and performs BF16 accumulation. It owns no expert parallelism, loader, launch,
or execution authority.

## Compact expert execution

packed_expert_mlp_source consumes the generic precision/expert plan and emits
three executable AOT stages by default: gate/up plus SwiGLU, down plus routing
weight, and canonical expert-order F32 combination. Compact weights are decoded at
consumption using the shared precision lowering; no whole-bank BF16 copy is
required. The backend owns 32-lane reductions and launch geometry.

The integration/packed_expert_mlp binder prepares these stages for the existing
ordered executor. The GLM adapter supplies checkpoint names, shapes and clamp;
the compute path has no model-family switch. Current coverage is BF16 activation
with NVFP4, MXFP4 or BF16 weights. An explicit activation quantization plan
adds BF16-to-FP8 producers before gate/up and down, yielding five ordered AOT
stages. These producers compute each scale block once, rather than repeating
amax for every output channel. Numeric IR selects staged BF16 or F32 SwiGLU
product rounding and routing-score placement before or after down.

The DeepSeek adapter selects block128 E4M3/UE8M0 activations, an explicit
1e-4 amax floor, F32 product arithmetic, and routing scores before down-input
BF16 rounding. FP4 experts remain packed; FP8 experts retain 128x128 scales.
This is executable expert compute, not a full GLM/DeepSeek/MiniMax serving
executor. Its SIMT accumulation has a declared tolerance, not bitwise parity
with the reference tensor-core schedule. Tensor-core expert GEMM and cross-rank
execution remain separate implementation work.
