# Counter-guided compiler fixes: GPU validation

This validates the four work streams following
[selected-kernel diagnosis](SELECTED_KERNEL_COUNTERS_2026-09-09.md).
These are isolated kernel measurements, **not new end-to-end serving results**.

## Implemented transformations

- Projection: a pure product-of-ordered-folds partition distributes independent
  rows across consumers without reassociating the K reduction. QKV/output use
  all eight launched warps; gate/up uses all sixteen.
- CUDA operand layout: a private immutable physical layout maps logical copy
  and matrix-consumer addresses to padded BF16 rows. QKV/output pitch64 becomes
  80; down pitch32 becomes48. CUDA alignment/bank details stay in the backend.
- Down producers: specialize the input and weight copy domains independently,
  eliminating their per-vector operand tag/pointer selection.
- Attention: pure finite-product expansion of subgroup-strided query folds;
  CUDA lowers the carried maxima/denominators to named scalar variables instead
  of dynamically indexed arrays. Per-query arithmetic order is unchanged.

These are model-independent transformations, not Qwen branches. The physical
test geometry is Qwen3-0.6B BF16; this does not establish performance on every
shape or accelerator. The attention `selected-counters` exporter option is a
test fixture, not a production dispatch policy.

## Unprofiled isolated timings

RTX 5060 Ti, UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`.
Control: unchanged selected serving CUBINs from `d3d81ca`. New matrix modules
come from current compiler exports with diagnostic symbol `baseline`; attention
retains the production symbol and exact ABI. All runs are serialized.

Three interleaved old/new trials, three warmup launches and 20 measured launches
per kernel per trial, CUDA-event timing. Values below are rounded microseconds
representative of the three trials; full values are retained. No confidence
interval or end-to-end throughput claim is implied.

| Operation | 504 tokens: old → new µs | 1024 tokens: old → new µs | 1024 speedup |
| --- | ---: | ---: | ---: |
| QKV | 329 → 213 | 645 → 415 | 1.55x |
| Output projection | 177 → 109 | 330 → 203 | 1.63x |
| Gate/up | 328 → 323–325 | 656 → 647 | 1.01–1.02x |
| Down | 131 → 113 | 248 → 186 | 1.33x |
| Prefill attention | 733 → 672 | 1143 → 1049 | 1.09x |

For matrix tests, token counts are 1,7,17,63,64,65,504,1024. Paths below the
pipeline threshold are largely unchanged; noisy microsecond differences there
are not reported as wins. Attention additionally tests 31,32,33 query tokens,
all with a 1528-token context. These are synthetic deterministic BF16 inputs,
not captured model activations. Attention and output projection are measured
independently, not as an end-to-end chain.

## What the fresh counters actually changed

Both old and new were recaptured with Nsight Compute `--set full
--cache-control all --clock-control none`, 40 passes per kernel. Counts are
per 1024-token launch. Barrier values below are average stalled warp cycles
per issued instruction, **not kernel wall-time percentages**.

| Operation | Shared-load bank conflicts, old → new | Barrier cycles/issue, old → new | Tensor-pipe active, old → new |
| --- | ---: | ---: | ---: |
| QKV | 44.15M → 8.47M | 4.64 → 0.76 | 26.1% → 41.6% |
| Output | 22.06M → 4.22M | 5.33 → 0.92 | 25.4% → 43.4% |
| Gate/up | 14.16M → 28.54M | 8.22 → 3.46 | 36.9% → 38.1% |
| Down | 11.86M → 3.95M | 0.89 → 0.65 | 53.2% → 67.6% |
| Attention | 45.66M → 45.63M | 2.48 → 2.46 | 14.6% → 16.1% |

Attention local-load sectors fall **2,195,456 → 0**, and local-store sectors
**2,293,760 → 0**. Thus scalar expansion removes the measured 143.65MB of
local-sector requests in this workload. Registers remain128. Long-scoreboard
cycles/issue fall5.22→3.71. Shared-load conflicts remain essentially unchanged:
that separate attention bottleneck is **not fixed**.

Gate/up registers fall122→92, but the same shared-memory/launch envelope still
limits residency, and duplicated weight-fragment loads offset much of the
barrier improvement. Its bank conflicts double and L1/TEX throughput rises
32.3%→61.7%. Therefore the row-distribution transformation works mechanically,
but **does not solve gate/up throughput**. It is a small measured gain, not the
large gain register count alone might suggest.

QKV/output combine layout and fold distribution in this capture; their individual
contributions have not been separated. Down similarly combines layout and
producer specialization. Do not credit the complete gain to one sub-change.

## Correctness and regression coverage

- All tested matrix outputs/workspaces compare bit-for-bit with the old selected
  kernels, including sentinel-initialized untouched tails.
- Attention compares complete output buffers bit-for-bit across ten query sizes.
- Each of the five new kernels passes memcheck, racecheck, initcheck and
  synccheck at1024 tokens: zero errors; racecheck also reports zero warnings.
- Vocabulary-head collateral check uses the prior matrix-pipeline control
  (`51300d94a9ba9be158c316e42ca39a335804eef8970da909896803a3e634874b`)
  and the shared helper's new head export. Eight token/selected-row cases pass
  full-byte equality, untouched-row checks and a sampled numeric oracle. Its
  short timing run is regression coverage, not a new head speedup claim.
- The four affected MoonBit packages pass96/96 tests. Warning-denied native
  checking, formatting check and interface generation pass.
- The complete native warning-denied suite passes **3630/3630** tests.

No new runtime deployment, full-serving benchmark, or SGLang/vLLM rerun was
performed. The previously reported serving gap must not be divided by these
isolated speedups. Attention shared layout and gate/up operand replication
are the next measured problems, followed by exact-source serving validation.

## Reproduction

Remote run root: `/run/user/1000/lunaflux-counter-fixes-20260909-r1`.
The archive contains exported CUDA sources, CUBINs, diagnostic driver sources,
MoonBit runners, compilation logs, differential timings, twenty sanitizer
results, ten `.ncu-rep` reports and their raw CSV exports. Raw NCU duration
units differ by report (microseconds for matrix kernels, milliseconds for
attention); read each CSV's units row, not the unitless summary alone.

Compiled with CUDA13.1, `--cubin -std=c++17 -O3 -arch=sm_120 --fmad=false
--maxrregcount=128 --ptxas-options=-v`. Launch geometries and dynamic shared
sizes match the control. Static shared increases where the layout is padded.
Profiling uses existing administrator access, without changing driver policy.

Archive: `/run/user/1000/lunaflux-counter-fixes-archive-20260909-r1/results.tar.gz`.
SHA-256: `012db259203112dae2a025e142dcc860570a72c023655069aa504a96d318a867`.
Local destination: `/private/tmp/lunaflux-counter-fixes-20260909-r1/results.tar.gz`.
The downloaded archive hash matches the remote archive.
