# Selected expert MLP CUDA source

This package renders a family-neutral serial correctness kernel for selected
SwiGLU experts. It validates distinct in-range indices, computes experts in
ascending expert-id order, rounds each weighted expert contribution to BF16,
and performs BF16 accumulation. It owns no expert parallelism, loader, launch,
or execution authority.
