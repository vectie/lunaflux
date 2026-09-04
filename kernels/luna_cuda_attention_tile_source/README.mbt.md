# Luna CUDA attention tile source

This CUDA-only backend package emits a matrix-tiled long-prefill kernel from a
selected functional LunaTile attention schedule. One block handles a
sequence-aligned query tile, reuses each paged K/V tile across all query rows,
computes QK with matrix operations, and fuses causal masking, online softmax,
PV accumulation, and normalization without materializing the complete score
matrix.

The source ABI is backend-private. Model and scheduler packages see only the
generic attention problem, semantic IR, and schedule.

When the functional optimizer selects paged-row address hoisting, the CUDA
terminal lowering computes one page-table address per logical K/V row and
broadcasts it across that row's vector fragments. This removes repeated page
division and table loads without exposing CUDA subgroup vocabulary above the
backend boundary.
