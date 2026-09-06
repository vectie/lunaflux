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

Grouped split-subgroup decode now realizes the schedule's contiguous column
vector map for both direct and partitioned execution. The existing 8-byte
schedule emits bit-preserving four-BF16 K/V transfers instead of scalar BF16
loads/stores. Width is selected above CUDA lowering; CUDA packed types remain
private here. Head boundaries and page strides must preserve alignment, with
compile-time scalar fallback for an unaligned stride. Key-tail masking, page
validation, synchronization, shared storage and floating-point fold order are
unchanged. This is synchronous vectorization, not an asynchronous pipeline.

The test-only `attention_tile_cuda_source_probe decode-serving` exporter emits
both production-shaped entry points through the compiler. The native CUDA
probe's `compare` mode compares two such modules bit-for-bit, checks a scalar
oracle, page/tile tails, empty partitions and mixed rows, verifies unchanged
K/V, and closes all resources. Its GPU fixture is the explicitly selected
RTX 5060 Ti, not an arbitrary visible device.
