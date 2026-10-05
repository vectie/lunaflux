# Earlier page readiness — 2026-10-05

## Decision

Do not promote this candidate. All six dual-Spark cells are inconclusive under
the fixed gate: every one of five paired reductions must reach 3%. Median
reductions are positive, 0.92–2.12%, but the additional instructions offset
most of the latency-hiding benefit. Production source and fixtures are restored
exactly to the accepted page-batch implementation. No serving bundle was rebound;
this is not an end-to-end or new vLLM/SGLang comparison.

One finite candidate, three workload cells per host, five alternating pairs per
cell, 30 GPU-event repeats per pair. .178 and .179 ran timings concurrently;
after timings .179 collected matched hardware counters while .178 ran short
sanitizer checks and long memcheck. One GPU job per host, no concurrent compiler
job during unprofiled measurements. Both devices are idle at completion.

## Hypothesis and compiler boundary

The existing split effect plan already stages next K before current PV. This
experiment does **not** move that copy or claim it was previously serialized.
It moves only immutable next-page metadata lookup to AfterKeyPublication,
before current QK arithmetic, reusing the existing pure PagedTileEpochs ownership
relation. Current K has consumed its identity; current V retains its already
resolved pointer. Validation stays at K consumption, not speculative production.

The CUDA terminal lowering uses a volatile `ld.global.u32` to prevent sinking
the metadata load back into its consumer. Only complete, page-aligned historical
tiles use this transformation. Partial/current/mixed/unaligned/overfull ownership
keeps the previous map. Numeric law, copy ordering, barriers, invalid-page
publication and retained V lifetime are unchanged. No model/device-name policy,
public API, request-path JIT, allocation or profiler dependency was introduced.
The experimental renderer is preserved outside production, not left as a dead
branch or permanent toggle.

## Unprofiled paired results

Q2048; runtime envelope rows32, block128, grid63×16×1 for both artifacts.
Times are median microseconds. Reduction is the median of individual paired
ratios, not the ratio of medians.

| Host | Rows / history | Page-batch µs | Readiness µs | Paired reduction | Worst pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| .178 | 1 / 28,672 | 6379.057 | 6316.776 | 1.14% | 0.62% |
| .178 | 2 / 28,672 | 6367.861 | 6245.298 | 2.12% | 1.11% |
| .178 | 2 / 8192 | 1912.231 | 1890.053 | 1.12% | 0.54% |
| .179 | 1 / 28,672 | 6611.412 | 6548.795 | 0.92% | −0.76% |
| .179 | 2 / 28,672 | 6605.139 | 6474.250 | 2.05% | 0.10% |
| .179 | 2 / 8192 | 1979.033 | 1956.225 | 1.15% | 1.04% |

## Matched executed counters

.179 Q2048/R2/H28672, baseline then candidate. Instrumented duration is
excluded from the decision. Warp-latency fractions are not additive elapsed-time
attributions and cannot be interpreted as percentages of wall time saved.

| Metric | Page-batch | Readiness |
| --- | ---: | ---: |
| Total warp instructions | 891,530,112 | 932,746,112 |
| `LOP3.LUT` | 33,782,784 | 42,182,656 |
| `IADD3` | 12,696,320 | 18,312,960 |
| `MOV` | 97,258,496 | 104,740,864 |
| `NOP` | 25,242,624 | 32,721,920 |
| `LDG.E` | 960,128 | 979,584 |
| `IMAD.WIDE.U32` | 7,342,080 | 7,342,080 |
| `SHFL.IDX` | 7,340,032 | 7,340,032 |
| Tensor `HMMA` | 119,668,736 | 119,668,736 |
| Async `LDGSTS` | 14,958,592 | 14,958,592 |
| CTA barriers | 2,808,832 | 2,808,832 |
| Registers/thread / allocated | 234 / 240 | 237 / 240 |
| Register/shared-limited resident blocks | 2 / 2 | 2 / 2 |
| Long-scoreboard / warp latency | 10.24% | 2.73% |
| Barrier / warp latency | 5.22% | 3.61% |
| Math-pipe throttle / warp latency | 14.89% | 19.31% |
| Wait / warp latency | 32.93% | 36.26% |
| Issue active | 28.73% | 34.03% |

Earlier page readiness substantially reduces the load-dependent wait, so this
hypothesis is supported. It also adds 4.62% executed instructions: guards,
address/control work and retained state are not free. Mathematical tensor work,
async-copy count and barriers are unchanged. Residency and spilling do not
explain the small gain. Do not pursue more page-address microchanges without
reducing their guard cost or changing a larger reuse boundary.

Next structural hypothesis, **not yet implemented or measured**: the pure
query-owned frontier and ownership classifier admit only Q32/Q64. A Q128
alternative could reuse each KV tile across twice as many queries; shared-memory
and residency costs must be measured, and tail/runtime-bucket work must remain
included. Earlier "retain query128" experiments referred to retained head
dimensions, not a 128-row query tile. This is a new experiment, not a claimed win.

## Validation and preservation

All 30 timing pairs are bitwise equivalent and pass the sampled BF16 oracle
ceiling 0.003. Q129/R2/H128 memcheck, racecheck and synccheck pass with zero
errors/hazards, oracle maxabs 0.000330008. Q2048/R2/H28672 memcheck passes,
oracle maxabs 0.000377474. Candidate reports 237 registers, two resident blocks,
zero local bytes. Experimental affected tests passed 121/121; restored source
and physical IR pass 120/120. Offline helper tests pass 3/3. Scoped native
checks use the existing warning exclusions 20/79/29/25, not a clean-whole-tree
claim. Public APIs are unchanged.

GPU178 UUID `GPU-9c3d3cf0-439a-5da2-67e9-20414255879f`;
GPU179 UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`. Both GB10/sm121.
GPU jobs are capped at 16GiB, MemorySwapMax=0, bounded tasks and 600 seconds.
Before/after paired MemAvailable readings exceed 122,087,744KiB against a
32GiB reserve. Existing host swap is not asserted to be zero, and outer helper
unit memory peaks are not serving-process peak-memory measurements.

Numerical law `approx-base2-f32-v1`; c30322/Q64/K64, stage1.
Baseline cubin `7b42b3531466f5fe22c59e5aac2883db68984907c49a5637ac5f1358ad2eab57`.
Experimental source `7b57beb265384418149dfb3ee49ec51c54435a337683f9dab1727f8752adca2a`.
Experimental cubin `e0a276f396790c00b1fae5929616d9831cb49edf471bb96dcbce550aded8e78d`.
Independent offline compilations match. CUDA13.0.88 nvcc stays pinned to
`fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.

Remote roots:

- .178 `/home/wlc003s/lunaflux-ako-page-readiness-20261005.28uwVSZP/experiment`
- .179 `/home/wlc004s/lunaflux-ako-page-readiness-20261005.hT9iETbs/experiment`

Local `/tmp/lunaflux-ako-page-readiness-20261005.yNNee1j3` preserves the source
patch, exact artifacts, raw paired timings, hardware capture and parsed report.
Downloaded archives and every manifest entry verified:

- .178 `88cca7135233a1a193fa2f41adb220294b303a0acf26f5e9e4e1904490e9aba0`
- .179 `a63fab436963a96ae653646f752ff82aecabae58f43169e1158673617c3fd352`
