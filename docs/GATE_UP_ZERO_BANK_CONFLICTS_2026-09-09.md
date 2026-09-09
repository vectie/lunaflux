# Gate/up shared-memory conflicts: zero on the measured selected kernel

## Scope and result

Fresh paired runs on RTX 5060 Ti, sm120, CUDA 13.1.115 compare the preceding
compact-16 layout (`b98eede`, documented by `15b45f7`) against the eight-wide
matrix-load and register-plane lowering. The measured case is the existing
BF16 gate/up probe: input width 1024, intermediate width 3072, 1024 rows,
768 CTAs, 512 threads/CTA. This is not an end-to-end serving result and does
not claim that every kernel or smaller-row fallback has zero conflicts.

| Nsight Compute metric | Before | After |
|---|---:|---:|
| Shared load bank conflicts | 9,542,281 | 0 |
| Shared store bank conflicts | 196,608 | 0 |
| Total shared bank conflicts | 9,738,923 | 0 |
| Local load/store sectors | 0 / 0 | 0 / 0 |
| Registers/thread (kernel maximum) | 92 | 92 |
| Static shared memory | 24,576 bytes | 24,576 bytes |

Counters are `l1tex__data_bank_conflicts_pipe_lsu_mem_shared{,_op_ld,_op_st}.sum`.
The asynchronous-copy subcounter is also zero. Each revision was collected
using the same full NCU set, 40 replay passes, cache-control all, clock-control
none. Profiled durations are not used as the benchmark headline.

## Instruction-level correction

The old WMMA loads lowered to generic-address `LD.E` instructions with two-way
shared bank conflicts. Its accumulator `STS.64` stores had four-way conflicts;
the scalar epilogue loads were already conflict-free.

The immutable backend layout now partitions operands into 8x8 BF16 subtiles.
Producer vector ownership is its inverse permutation, so adjacent producer
vectors have adjacent shared destinations. Explicit shared `ldmatrix.x4`
loads feed two independent `mma.m16n8k16` column halves. The ordered K16
reduction is unchanged. Accumulators exchange through register-major planes,
one contiguous word per lane, instead of WMMA row-major shared stores.

Fragment ownership follows the documented PTX layout, not undocumented WMMA
fragment indexing. See NVIDIA's [PTX matrix fragment specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float).
NVIDIA instructions remain in the private CUDA source generator. The functional
product-fold plan, model, scheduler, public API, launch shape and scratch
capacity are unchanged. This is a hardware-specific lowering of the generic
ordered product fold, not a model-specific algorithm or a CUDA-free optimizer.

## Performance and correctness

Three interleaved trials, 30 repetitions/trial, CUDA-event timings (median):

| Token rows | Before µs | After µs | Change |
|---|---:|---:|---:|
| 256 | 149.56 | 147.47 | 1.4% faster |
| 257 | 158.06 | 159.65 | 1.0% slower |
| 504 | 272.35 | 268.07 | 1.6% faster |
| 1024 | 542.92 | 533.53 | 1.7% faster |

Conflict elimination is not a proportional throughput gain: remaining work
and synchronization still dominate. The tail-case regression is retained
explicitly; no claim of improvement for every shape is made.

- Gate/up and unchanged down output/workspace buffers compare bitwise over
  rows `1,7,17,63,64,65,255,256,257,504,1024`, including padded tails.
- Projection package: 32/32 tests; generic projection compiler: 35/35 tests.
- Native warning-denied check, `moon fmt`, `moon info`, `git diff --check` pass.
- Memcheck, racecheck, initcheck and synccheck: zero errors/hazards.
- Regression tests cover vector-preserving bijection, inverse producer
  mapping, matrix-load bank uniqueness and output register-plane coverage.
- No full-suite or fresh end-to-end serving benchmark is claimed in this
  scoped iteration. Unrelated working-tree changes were not included.

Raw before/after reports, binaries, deterministic probe source, timing logs
and sanitizer output are preserved in
`/run/user/1000/lunaflux-bank-zero-20260909-r1` on the test host.

The downloaded archive is
`/private/tmp/lunaflux-bank-zero-20260909-r1/results.tar.gz`, with matching
remote/local SHA-256
`b2184018201f7a36fd0c564d3ee2489245ee185dbfd9be4e85d91a377d4da105`.
