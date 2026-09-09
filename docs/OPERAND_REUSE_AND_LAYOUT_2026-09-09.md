# Operand reuse and compact attention layout

Follow-up to [the first counter-guided fixes](COUNTER_GUIDED_COMPILER_FIXES_2026-09-09.md).
Both changes are implemented through the normal compiler path. These are
isolated kernel results, not a new serving or vLLM/SGLang comparison.

## Transformations

- Gate/up: a pure partition distributes the independent sibling axis before
  the row product. For the measured shape, sixteen consumer warps become
  two siblings × two row groups × four output groups. Each loaded weight
  fragment serves two row accumulators. The logical fragment-load count per
  K16 falls from 48 to 40: weight loads halve, input loads double. This is
  a static schedule count, not a measured memory-transaction reduction.
  CUDA lowers the partition and reuses expired operand storage for the gate
  epilogue, with uniform CTA fences around its lifetime transitions.
- Attention: a pure, capacity-preserving rectangular-to-microtile bijection
  replaces full-row operand storage. CUDA chooses 16×16 microtiles and WMMA
  leading dimension 16. Q/K/V and BF16 probability producers and consumers
  use the same map; F32 probabilities in scalar-PV schedules stay row-major.

Neither transformation reassociates the ordered reduction, changes numeric
rounding, introduces model-family branches, or adds runtime allocation.
CUDA instructions, warp mapping and physical shared-memory addressing remain
in the CUDA backend. Existing launch and shared-memory capacities are retained.

## Real GPU timings

RTX 5060 Ti, `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, PCI
`00000000:17:00.0`. Immediate control is the previous measured kernel set at
`407d9f2`, not the original slower kernels. Synthetic BF16 inputs; serialized
old/new measurements, three interleaved trials with three warmup launches.
Gate/up uses 30 measured launches per trial; attention uses 20. Values are
medians in microseconds, rounded to two decimals.

| Kernel | Tokens | Previous | New | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Gate/up | 256 | 178.14 | 166.01 | 1.07× |
| Gate/up | 257 | 200.79 | 174.15 | 1.15× |
| Gate/up | 504 | 325.53 | 303.23 | 1.07× |
| Gate/up | 1024 | 646.93 | 604.37 | 1.07× |
| Attention | 504 | 672.44 | 480.76 | 1.40× |
| Attention | 1024 | 1048.65 | 747.46 | 1.40× |

Gate/up checks the token vector 1,7,17,63,64,65,255,256,257,504,1024.
The below-256 path is unchanged. The down kernel in the combined module also
passes a full differential regression across that vector; its timings are
unchanged within run noise. Attention checks 1,7,17,31,32,33,63,65,504,1024
query tokens with a 1528-token context. All complete output/workspace buffers,
including sentinel-initialized untouched tails, match the previous kernels
bit-for-bit. These selected-shape comparisons are not an independent model
quality test.

## Before/after hardware counters

Fresh Nsight Compute full captures, cache control `all`, clock control `none`,
40 passes per kernel at 1024 query tokens. Profiled duration is not substituted
for the unprofiled timings above. Barrier values are average stalled warp
cycles per issue, not wall-time percentages.

| Metric | Gate/up previous → new | Attention previous → new |
| --- | ---: | ---: |
| Shared-load bank conflicts | 28,541,868 → 28,549,858 | 45,630,100 → 6,698,075 |
| Shared-store bank conflicts | 196,608 → 196,608 | 2,231,122 → 9,353,941 |
| Barrier cycles/issue | 3.459 → 2.138 | 2.453 → 1.461 |
| Tensor-pipe active | 38.14% → 40.99% | 15.76% → 22.21% |
| Registers/thread | 92 → 92 | 128 → 128 |
| Local load/store sectors | 0/0 → 0/0 | 0/0 → 0/0 |

Attention shared-load conflicts fall **85.3%**, validating the targeted layout
change. Producer shared-store conflicts increase, so this does not eliminate
all shared-memory contention. Gate/up improves weight reuse and latency, but
does not reduce aggregate bank conflicts; duplicated input reads and the
terminal handoff are explicit tradeoffs, not a claim of optimal reuse.

## Validation and reproduction

Both selected kernels pass memcheck, racecheck, initcheck and synccheck:
eight checks, zero errors, and zero race warnings. Focused tests additionally
cover ownership for all tail lengths and admitted consumer counts, epilogue
capacity, bijective compact layout, vector alignment, scalar-PV mapping and
unchanged decode emission.

The final affected-package suite passes **105/105** tests; the full native
warning-denied suite passes **3639/3639**. Native checking, formatting and
interface generation pass.

Five additional attention schedules (290,291,300,316,317) pass the independent
CPU numeric oracle: exhaustive short/ragged cases and deterministic sampled
queries over contexts 512,1024,2048,4096. The unchanged tolerance is
`0.0025 + 0.005 * abs(expected)`. Maximum normalized error is below 0.466
(pass requires at most 1). These use the oracle's four-KV-head/page16 ABI,
not the selected benchmark's eight-KV-head/page8 ABI. No performance claim
is made for these additional schedules.

Both async schedules additionally pass all four sanitizers over the bounded
short/ragged plus query17/context257 cases: eight further checks, zero errors
and zero race warnings. Total sanitizer checks for this follow-up: **16**.
The GPU returns to its initial 15 MiB usage with no compute process.

CUDA compile flags: `--cubin -std=c++17 -O3 -arch=sm_120 --fmad=false
--maxrregcount=128 --ptxas-options=-v` with CUDA 13.1.

Remote run directory:
`/run/user/1000/lunaflux-two-fixes-20260909-r1`.
It retains sources, binaries, driver, MoonBit orchestration, raw differential
timings, eight sanitizer logs, four Nsight reports and unit-bearing CSV data.

Selected source hashes:

- Gate/up: `4521b66104c385154578b285dbd95769aa3200928e396144ccec63618d862e7c`.
- Attention: `9a83ab26f2b50e841ff114b6892817259b84c2960d3ce55cb5a8455e1ef199e4`.

Implementation commits: `26f8032` (sibling reuse), `6779822` (compact layout).
Archive: `/run/user/1000/lunaflux-two-fixes-archive-20260909-r1/results.tar.gz`.
SHA-256: `b0abe76c6302c17c187e86aa1ac33be831f7f25095cec9702e2ff76a065ca1b5`.
Downloaded to `/private/tmp/lunaflux-two-fixes-results-20260909-r1/results.tar.gz`;
the local hash matches.

No production deployment, baseline-engine rerun or end-to-end speedup claim
is part of this measurement.
