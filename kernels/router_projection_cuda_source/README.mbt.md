# Family-neutral router projection CUDA source

This package renders a bounded correctness kernel for F32 activation rows,
row-major BF16 no-bias weights, ordered F32 accumulation, and F32 logits.
Score activation, correction bias, selection, and expert execution are outside
the operation.

The package owns no model, compiler, artifact, qualification, launch, loader,
or execution authority.
