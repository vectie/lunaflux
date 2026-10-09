# Selected expert MLP CUDA source

This package renders a family-neutral serial correctness kernel for selected
SwiGLU experts. It validates distinct in-range indices, computes experts in
ascending expert-id order, rounds each weighted expert contribution to BF16,
and performs BF16 accumulation. It owns no expert parallelism, loader, launch,
or execution authority.

## Compact expert execution

packed_expert_mlp_source consumes the generic precision/expert plan and emits
three executable AOT stages: gate/up plus SwiGLU, down plus routing weight,
and canonical expert-order F32 combination. Compact weights are decoded at
consumption using the shared precision lowering; no whole-bank BF16 copy is
required. The backend owns 32-lane reductions and launch geometry.

The integration/packed_expert_mlp binder prepares these stages for the existing
ordered executor. The GLM adapter supplies checkpoint names, shapes and clamp;
the compute path has no model-family switch. Current coverage is BF16 activation
with NVFP4, MXFP4 or BF16 weights. This is not DeepSeek's dynamic-FP8 activation
algorithm, nor a full GLM/MiniMax serving executor. Tensor-core expert GEMM and
cross-rank execution remain separate implementation work.
