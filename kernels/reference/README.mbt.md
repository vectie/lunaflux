# LunaFlux reference kernels

This package is a deterministic numerical correctness oracle for the Phase 1
dense-Llama path. It is not registered as a production kernel, is not a device
fallback, and makes no steady-state allocation or performance claim.

The numerical contract is fixed:

- model bytes are decoded strictly as little-endian BF16;
- decoded BF16 values become IEEE-754 binary32 `Float` values;
- every multiply, accumulation, normalization, transcendental operation,
  attention score, softmax value, and returned activation uses `Float`;
- outputs are not rounded back to BF16;
- non-finite inputs and non-finite computed outputs are rejected;
- Llama RoPE uses split-half `rotate_half` pairing, not adjacent pairs;
- dense weights use `[output_width, input_width]` row-major layout;
- QKV activations concatenate Q, K, then V in head-major layout;
- GQA maps contiguous equal groups of query heads to one KV head.

All allocated matrix and scratch dimensions are checked against the caller's
`ReferenceLimits` before allocation. Because the implementation is only a
correctness oracle, production executors must select separately validated AOT
or vendor kernels rather than route live requests through this package.
