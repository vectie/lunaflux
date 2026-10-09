# Affine modulation CUDA source

This family-neutral package renders deterministic fixed-geometry CUDA source
for row-aligned BF16 shift/scale modulation followed by an F32 output boundary.
It stages `hidden * (1 + scale) + shift` through three explicit BF16
round-to-nearest-even results before widening the final value to F32.

It does not project conditioning embeddings, select rows from a timestep
table, compile CUDA, admit artifacts, or grant execution authority.

The package also renders one independent final-AdaLN parameter-table row:
F32 conditioning is activated with SiLU, rounded to BF16, multiplied by a
row-major BF16 dense weight with BF16 bias under ordered F32 accumulation, and
split into BF16 shift and scale halves. Repeating rows and selecting them for
packed tokens remain orchestration concerns.

The package also renders family-neutral F32 channel-major inverse
normalization. Its four-operand ABI is normalized input, ordered channel
standard deviations, ordered channel means, and output; multiply and add use
separate round-to-nearest F32 instructions.
