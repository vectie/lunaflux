# Decoder hyper-connection CUDA source

This family-neutral package emits deterministic correctness-grade CUDA source
for versioned mHC block-control generation. One thread owns one bounded live
row. The ABI is counts, BF16 streams, separate F32 function/base/scale inputs,
then separate row-major F32 pre, post, and combination outputs.

The source fixes ordered F32 multiply/add/divide operations, normalized control
projection, sigmoid controls, stable per-destination softmax, and the exact
alternating normalization count. CUDA expf and sqrtf behavior remains bound to
an independently authenticated compiler and requires numeric qualification.
The package emits source only and grants no artifact or execution authority.

The pre-reduction renderer consumes row-major BF16 stream state and the F32
pre controls emitted by block control. One block owns each bounded live row;
threads stride hidden columns, reduce streams in ascending order with explicit
F32 multiply/add, and perform one BF16-RNE output round.

The post-combination renderer consumes a BF16 branch, BF16 residual stream
state, and F32 post/combination controls. Combination matrices are
destination-major then source-minor. Each output accumulates the post-scaled
branch followed by residual sources in ascending order and rounds once to BF16.

The head-reduction renderer consumes BF16 `[rows,width,hidden]` stream state,
F32 `[width,width*hidden]` function weights, F32 `[width]` base controls, and a
scalar F32 scale. It normalizes the flattened live row, derives width sigmoid
controls, reduces streams in ascending order, and rounds `[rows,hidden]` once
to BF16. CUDA `expf` and `sqrtf` remain compiler-bound qualification gaps.
