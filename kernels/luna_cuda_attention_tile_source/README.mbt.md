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

When online-softmax storage reuse is selected, each subgroup retains its
running maximum and denominator in registers. Probability values reuse the
dead key-tile allocation and terminal fold state reuses that allocation after
PV. For head dimension 128, Q32/K64 shared memory shrinks from 70,032 to
65,536 bytes. These offsets now come from the portable schedule's explicit
storage plan. At Q64/K64/head64 the key overlay must include a 256-byte rescale
tail: total shared storage is 57,600 bytes, not the insufficient 57,344-byte
old bound. Lifetime boundaries include synchronization after query validation
and after tile-validation readers, before the next staging/QK writer.
