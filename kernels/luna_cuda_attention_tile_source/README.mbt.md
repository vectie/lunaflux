# Luna CUDA attention tile source

This CUDA-only backend package emits a matrix-tiled long-prefill kernel from a
selected functional LunaTile attention schedule. One block handles a
sequence-aligned query tile, reuses each paged K/V tile across all query rows,
computes QK with matrix operations, and fuses causal masking, online softmax,
PV accumulation, and normalization without materializing the complete score
matrix.

The source ABI is backend-private. Model and scheduler packages see only the
generic attention problem, semantic IR, and schedule.

`FuseScoreTransform` is realized by consuming scale/mask directly inside
softmax, without another shared score write and block barrier. Explicit
float multiplication rounding preserves the former materialization boundary.
`FactorSharedKeyAcrossQueryMap` keeps independent matrix accumulators while
loading their shared K fragment once per reduction slice. Neither lowering
changes the per-output reduction order. See
`docs/PREFILL_SCORE_DEFORESTATION_2026-09-08.md` for paired GPU and Qwen results;
kernel-level gains must not be reported as whole-serving gains.

Matrix-prefill candidates 316 (Q32/K32) and 317 (Q64/K32) now realize a
two-slot lookahead over the ordered KV fold. The next tile's zero-filled
16-byte K/V copies are issued before current QK/softmax/PV, and waited on only
at the next first consumer. Their portable storage extraction retains future
keys and transfer validation separately from current probabilities and scores;
the synchronous overlay would race across iterations. Numerical reduction and
rounding order are unchanged. Async compiler capability is required, and the
static cost model does not assume a speedup; paired GPU measurements determine
selection. The existing source probe accepts `316` and `317`, paired against
`312` and `314` respectively.

When the functional optimizer selects paged-row address hoisting, the CUDA
terminal lowering computes one page-table address per logical K/V row and
broadcasts it across that row's vector fragments. This removes repeated page
division and table loads without exposing CUDA subgroup vocabulary above the
backend boundary.

When online-softmax storage reuse is selected, each subgroup owns its
running maximum and denominator. The portable finite-product transform now
expands statically bounded query slots into named scalar states, retaining a
runtime active-query guard on each component. Unlike the previous dynamically
indexed local arrays, these states have no addressable array representation.
Register allocation and actual local-memory traffic still require compiled
kernel counters; source scalarization alone is not a physical performance claim.
The ordered KV recurrence and all arithmetic expressions remain unchanged.
Probability values reuse the
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

The backend also realizes the generic two-stage asynchronous schedule as
alternating shared K/V buffers: produce the first tile, await its completion,
produce the next tile while consuming the current one, then retire consumers
before reusing their storage. CUDA `cp.async` and shared-address conversion
remain private here. Tile-tail transfers are zero-filled without out-of-range
global addresses. No arithmetic or softmax recurrence is replaced.

The portable schedule keeps logical decode partition grain independent of
the staging tile: a 64-token partition grain can be consumed by two ordered
32-token transfers. This prevents smaller buffers from silently changing the
floating-point reduction grouping. At head dimension 128, two 64-token buffers
need 66,844 bytes; two 32-token buffers need 33,820 bytes. Async support must be
explicitly enabled in compiler capabilities; existing synchronous production
capabilities remain unchanged. The test exporters `decode-pipeline` and
`decode-pipeline32` explicitly select schedules, not measured autotune records.

The test-only `attention_tile_cuda_source_probe decode-serving` exporter emits
both production-shaped entry points through the compiler. The native CUDA
probe's `compare` mode compares two such modules bit-for-bit, checks a scalar
oracle, page/tile tails, empty partitions and mixed rows, verifies unchanged
K/V, and closes all resources. Its GPU fixture is the explicitly selected
RTX 5060 Ti, not an arbitrary visible device.

## Compact matrix operand storage

Prefill Q/K/V and BF16 probability operands use compact 16-by-16 microtile
storage. The portable schedule validates a capacity-preserving bijection;
CUDA owns the WMMA extent and its concrete address expression. Cooperative
and asynchronous vector stores use the same permutation as matrix loads,
whose leading dimension is now 16 rather than the whole head/key width.
Scores and output accumulators retain their existing layout, and no fold,
rounding, synchronization, storage lifetime, or launch capacity is changed.
This targets shared-bank aliasing without padding the shared arena. The
selected candidate313 passes GPU differential and sanitizer checks on sm120;
fresh counters show 85.3% fewer shared-load conflicts and isolated latency
improves 1.40×. Producer shared-store conflicts increase. See
[the measured report](../../docs/OPERAND_REUSE_AND_LAYOUT_2026-09-09.md) for
geometry and scope; these are not all-shape or end-to-end claims.
