# Decoder projection CUDA source

This family-neutral package renders deterministic correctness-grade BF16
dense-projection CUDA source from exact input/output geometry and a bounded
maximum row count. Weights are contiguous row-major `[output,input]` values,
there is no bias operand, products and additions use explicit round-to-nearest
FP32 operations in declared input order, and the result is rounded once to
BF16. It also renders exact-row F32 input/weight/bias projections with either
F32 output or one final BF16 round for family adapters whose authenticated
checkpoint and activation boundary require those storage contracts.
The grouped renderer uses `[row,group,input]`, `[group,output,input]`, and
`[row,group,output]` layouts without a cross-group reduction.

The package owns source text only. It has no model identity, tensor binding,
compiler invocation, artifact bytes, loader, launch permission, or
qualification authority.
