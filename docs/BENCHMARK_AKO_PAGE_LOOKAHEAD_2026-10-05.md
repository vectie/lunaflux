# Historical page lookup lookahead — 2026-10-05

## Decision

Do not select this experimental renderer. Against the accepted page-batch
baseline, median paired kernel reductions are 1.15–2.98%, but none of the six
cells clears the predeclared 3% minimum across every pair. One .178/C1 pair
regresses 0.43%. Production source and its tests are restored exactly to the
accepted page-batch version; only offline experiment support and this report
remain. This is not an additional serving gain or a refreshed framework comparison.

Both Sparks ran concurrently, one GPU workload per device. The finite budget
was one variant, three cells per host, five alternating pairs per cell. The
experiment finished; no additional candidates were searched after seeing results.

## Hypothesis and implementation

The accepted kernel still waits on an immediate page-ID bounds check. The
experiment consumes the existing portable `PagedLookupLookahead` relation in
terminal CUDA lowering: subgroup owners retain immutable IDs for several tiles,
then consumers borrow the current tile's page subset. For the measured K64/page8
geometry this is eight tiles per epoch, derived from ownership, not a model rule.

Only complete page-aligned historical tiles use this map. Future loads are
bounded by the complete historical interval and partition limit. IDs are
validated only when their tile is consumed, preserving invalid-page publication
and retained V ownership. The subgroup shuffle executes before owner-lane
predication. Partial/mixed/current paths, numerical expressions, reduction order,
copy readiness and barriers remain unchanged. No runtime JIT, heap allocation,
model branch or additional IR/API was introduced.

The experimental source and AOT artifacts are preserved externally below rather
than leaving an unselected implementation or feature flag in production.

## Paired unprofiled results

Q2048, actual runtime envelope Q2048/rows32, grid63×16×1, block128. Each pair
contains 30 GPU-event repeats. Times are medians in microseconds; gains are
medians of individual `1-new/old`, not ratios of medians. All30 pairs match the
baseline output bitwise and pass the sampled BF16 oracle ceiling0.003.

| Host | Rows / history | Page-batch µs | Lookahead µs | Paired reduction | Worst pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| .178 | 1 / 28,672 | 6369.389 | 6296.840 | 1.15% | −0.43% |
| .178 | 2 / 28,672 | 6396.756 | 6218.514 | 2.40% | 1.09% |
| .178 | 2 / 8192 | 1915.994 | 1883.386 | 1.54% | 0.88% |
| .179 | 1 / 28,672 | 6623.776 | 6539.303 | 1.25% | 0.58% |
| .179 | 2 / 28,672 | 6629.166 | 6460.736 | 2.70% | 1.72% |
| .179 | 2 / 8192 | 2008.378 | 1948.469 | 2.98% | 0.98% |

All six decisions are **inconclusive** at the fixed conservative threshold.
Lowering that threshold after measurement would change the experiment's decision
rule. No serving bundle was rebound to this non-winner.

## Matched source/SASS counters

.179, Q2048/R2/H28672, baseline then candidate. NCU profiled duration is not used
as the unprofiled timing decision. Executed counts are warp instructions, not
DRAM byte counts; warp-latency fractions and PC samples are not additive elapsed
time allocations.

| Metric | Page-batch | Lookahead |
| --- | ---: | ---: |
| Total instructions | 891,530,112 | 914,909,056 |
| Scalar global `LDG.E` | 960,128 | 272,000 |
| `IADD3` | 12,696,320 | 15,005,440 |
| `LOP3.LUT` | 33,782,784 | 37,524,480 |
| Register `MOV` | 97,258,496 | 99,807,232 |
| `SHFL.IDX` | 7,340,032 | 8,257,536 |
| Branch `BRA` | 10,464,256 | 11,609,088 |
| `NOP` | 25,242,624 | 27,112,448 |
| Tensor `HMMA` | 119,668,736 | 119,668,736 |
| Async copies `LDGSTS` | 14,958,592 | 14,958,592 |
| CTA barriers | 2,808,832 | 2,808,832 |
| Registers/thread | 234 | 238 |
| Allocated registers/thread | 240 | 240 |
| Register/shared-limited resident CTAs | 2 / 2 | 2 / 2 |
| Long-scoreboard / warp latency | 10.18% | 3.99% |
| Barrier / warp latency | 5.37% | 6.57% |
| Math-pipe throttle / warp latency | 14.94% | 16.46% |
| Wait / warp latency | 33.00% | 32.61% |
| Issue active | 31.59% | 30.58% |

The desired dependency was reduced, but supporting instructions rise2.62%.
Residency and spill behavior do not worsen, so blaming occupancy alone would be
incorrect. The new epoch refresh test, bounded owner selection and extra shuffle
are visible in generated source and executed SASS. Their additional work is a
plausible counterweight to the removed load dependency; this is not a complete
causal allocation of each microsecond.

The baseline hottest metadata PC is `0x325b6dfd0`, a load-dependent
`ISETP.GE.U32.AND ...0x4000`, with46,619 not-issued long-scoreboard samples. New
top sites are `BAR.SYNC` at`0x325b88470` (30,035), the epoch-ID `SHFL.IDX`
at`0x325b81c40` (11,821), and `WARPSYNC.ALL` at`0x325b812b0` (11,070).
Total not-issued long-scoreboard samples are67,590→62,627. Moving the dependency
does not eliminate it; do not claim that every wait or synchronization is fixed.

The next metadata hypothesis should remove per-tile epoch-selection arithmetic
or place producers earlier in a proven effect interval, rather than blindly
increasing the prefetch horizon. A new experiment requires its own finite budget.

## Correctness, limits and preservation

.178 Q129/R2/H128 memcheck, racecheck and synccheck all pass, zero errors/hazards;
bitwise=true, oracle maxabs0.000330008. Candidate uses238 registers, two resident
blocks and zero local bytes. Native affected tests pass121/121 before restoration
and120/120 afterward, with existing repository warning exclusions20/79/29/25.
Offline helpers pass warning-denied checks and tests. Unrelated dirty work remains
untouched; no aggregate clean-repository validation claim is made.

GPU178 UUID`GPU-9c3d3cf0-439a-5da2-67e9-20414255879f`;
GPU179 UUID`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`. Both GB10/sm121.
CPU build caps8GiB, GPU jobs16GiB, zero new swap, bounded tasks and600-second
GPU deadlines. The paired helper checks32GiB MemAvailable reserve; observed
before/after readings exceed121,914,872KiB. No compilation runs during timing.
Both devices are idle after completion.

Explicit numerical law remains`approx-base2-f32-v1`, c30322/Q64/K64,
symbol`lunaflux_attention_prefill_tile_compiler_exp2_v1`.
Baseline cubin:`7b42b3531466f5fe22c59e5aac2883db68984907c49a5637ac5f1358ad2eab57`.
Experimental cubin:`44a64ae6078153a2d723c5c6e0085d41db2f9fbd4d74ff82f6ea7dd2469df09b`.
Experimental source:`2d06a57839de906cb81258c2943b995e2696fb11109fcfe97679d9b5cbe3e21a`.
Two independent offline compilations match. CUDA13.0.88 nvcc remains pinned to
`fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.

Remote roots:

- .178 `/home/wlc003s/lunaflux-ako-page-lookahead-20261005.ac5W09mL/experiment`
- .179 `/home/wlc004s/lunaflux-ako-page-lookahead-20261005.uuKCO0o9/experiment`

Local `/tmp/lunaflux-ako-page-lookahead-20261005.8SShbLEt` contains source patch,
exact artifacts, raw paired outputs, NCU report/source counters and parsed report.
Sealed archive hashes, verified after download and against every manifest entry:

- .178 `2828a0de80710361cfd1bb4eb84d543241094a730507a570eae3428288f7a31c`
- .179 `00005d7be45bf8e86ac47bdef697854f241c80ecd3409ead941f28b66f8718e4`

An initial helper invocation selected a copied old default-target binary and
rejected the experiment before creating its output. Explicit native rebuilding
resolved it; no failed invocation is reported as a completed GPU result.
