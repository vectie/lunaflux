# Compiler prefill transfer pipeline — 2026-09-08

## Result

A compiler-generated two-slot K/V transfer pipeline improves the paired
synchronous kernel by **1.23–1.26× for Q32 tiles at query lengths 16/64 and
contexts 512–4096**, and by **1.17–1.30× for Q64 tiles** on the tested RTX 5060 Ti.
It is not a universal replacement: the Q32 pipeline loses on query length 128
and short/ragged cases. No default or production capability was enabled.

The Q128 result illustrates why candidate-local improvement is not sufficient:
at context 4096, the Q64 pipeline improves its synchronous counterpart from
1083.940 to 843.598 μs, but synchronous Q32 is still faster at 823.442 μs.
Use measured shape/device selection, not a global async preference.

## Functional compiler boundary

Candidates 316/317 preserve the same ordered KV fold, QK/softmax/PV operation
sequence, reductions, and BF16 rounding as synchronous 312/314. The change is
one-element transfer lookahead with independent ring-slot ownership. Generic
candidate scheduling and storage lifetimes express that dependency; only the
CUDA source backend emits cp.async, commit, wait, and shared-address conversion.

The next K/V tile is issued before current-tile QK begins. Predicated tail
vectors use source-size-zero 16-byte copies. Current K/V remains live until PV
finishes, while probability/scale and validation storage cannot alias future
copies. This is actual generated overlap, not an unused planner option.

## Measurement

Compiler implementation commit: 9190fee. The physical source was regenerated
after the helper extraction and compared byte-for-byte with the uploaded input.
The follow-up probe-only LF_BOUNDED_SANITIZER switch reduces instrumented case
count; it does not alter timed cases or generated kernels.

- Device: RTX 5060 Ti, UUID GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6,
  PCI 00000000:17:00.0.
- CUDA 13.1.115, sm_120, O3, fmad=false, ftz=false, precise division/sqrt,
  maxrregcount=128.
- Head dimension 128, KV tile 32, block 256 threads.
- Each timing: 10 warmups, 9 samples of 40 iterations; CUDA-event median.
- Sequential isolated order: synchronous Q32, async Q32, synchronous Q64,
  async Q64. These are kernel microbenchmarks, not end-to-end Qwen or a fresh
  vLLM/SGLang comparison.
- Query/context lengths are independent vectors, including tails and ragged rows.

All times are μs; speedup is synchronous time divided by async time.

| Case | Sync Q32 | Async Q32 | Speedup | Sync Q64 | Async Q64 | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single-16 | 8.237 | 8.228 | 1.001× | 10.284 | 10.234 | 1.005× |
| single-65 | 18.478 | 20.511 | 0.901× | 25.301 | 22.555 | 1.122× |
| single-128 | 32.049 | 34.853 | 0.920× | 43.039 | 34.858 | 1.235× |
| ragged-17-65 | 18.496 | 22.549 | 0.820× | 34.102 | 28.919 | 1.179× |
| q16-context-512 | 64.245 | 51.690 | 1.243× | 87.940 | 74.179 | 1.186× |
| q16-context-1024 | 124.970 | 100.366 | 1.245× | 170.269 | 144.860 | 1.175× |
| q16-context-2048 | 243.954 | 195.337 | 1.249× | 335.463 | 284.454 | 1.179× |
| q16-context-4096 | 484.143 | 386.942 | 1.251× | 666.672 | 563.835 | 1.182× |
| q64-context-512 | 80.836 | 65.558 | 1.233× | 144.182 | 112.733 | 1.279× |
| q64-context-1024 | 155.662 | 124.922 | 1.246× | 279.422 | 217.102 | 1.287× |
| q64-context-2048 | 304.750 | 243.646 | 1.251× | 550.298 | 424.658 | 1.296× |
| q64-context-4096 | 604.322 | 480.309 | 1.258× | 1092.580 | 841.062 | 1.299× |
| q128-context-512 | 108.584 | 126.971 | 0.855× | 143.268 | 112.766 | 1.270× |
| q128-context-1024 | 210.228 | 243.888 | 0.862× | 276.708 | 217.154 | 1.274× |
| q128-context-2048 | 414.962 | 481.410 | 0.862× | 545.302 | 425.970 | 1.280× |
| q128-context-4096 | 823.442 | 954.831 | 0.862× | 1083.940 | 843.598 | 1.285× |

## Resource tradeoff

| Candidate | Tile | Shared bytes | Registers/thread | Spill loads/stores |
| --- | --- | ---: | ---: | --- |
| 312 synchronous | Q32/KV32 | 45,056 | 126 | 0 / 0 |
| 316 pipeline | Q32/KV32 | 63,632 | 128 | 0 / 0 |
| 314 synchronous | Q64/KV32 | 73,728 | 126 | 0 / 0 |
| 317 pipeline | Q64/KV32 | 94,480 | 126 | 0 / 0 |

The larger Q32 shared footprint reduces the number of CTAs that can fit within
the device's shared-memory budget. This is consistent with the Q128 regression;
no profiler-based attribution percentage is claimed. The next optimization
should reduce overlapping storage/retain fragments in registers or use an
epoch-aware slot alias, rather than increasing pipeline depth blindly.

## Correctness and checks

- All four candidates passed the 16-case deterministic scalar-oracle matrix.
  Small/tail/ragged cases are exhaustive; long cases use deterministic sampling.
- All 32 paired BF16 output files matched bit-for-bit (312 vs 316 and 314 vs 317).
- Both async candidates passed memcheck, racecheck, initcheck, and synccheck.
  Instrumented cases: single16, single65, single128, ragged17+65, and Q17 with
  context257 (repeated ring-slot reuse and both query/KV tails); no timing loops.
- ptxas reported zero spill loads/stores for all four candidates.
- Final GPU process inventory was empty; the exclusive test lease was released.
- Focused compiler/strategy/storage/lowering/source packages: 55/55 tests passed.

## Reproduction and artifacts

Existing generator entry:
`moon run --target native tests/attention_tile_cuda_source_probe -- CANDIDATE_ID`,
for 312, 314, 316, 317. Compile generated_attention_tile.cu through the existing
probe.cu harness. Sanitizer builds additionally define LF_SKIP_BENCHMARK and
LF_BOUNDED_SANITIZER.

The isolated orchestration script is a MoonBit .mbtx file, with no production
runtime changes or dependencies. Inputs and outputs were created in new paths.

- Remote directory:
  /dev/shm/lunaflux-prefill-pipeline-r2-bundle/lunaflux-prefill-pipeline-20260908-r2
- Local extracted results:
  /private/tmp/lunaflux-prefill-pipeline-r2-results/lunaflux-prefill-pipeline-20260908-r2
- Input archive SHA-256:
  9eee146d4b8168c77f61e71b5bb1e1f1a2aec7f7ca91fb9a860d5f4cc915949d
- Downloaded result archive:
  /private/tmp/lunaflux-prefill-pipeline-r2-results.tar.gz
- Matching local/remote result archive SHA-256:
  c6afbebe20c562033315109221ededaeb5b8a0afde407c22ec0e10a2a61b60d0

No end-to-end parity, production promotion, or performance claim outside this
shape/device matrix follows from these measurements.

