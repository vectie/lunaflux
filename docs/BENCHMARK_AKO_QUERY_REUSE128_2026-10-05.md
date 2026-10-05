# Wider query reuse — dual Spark, 2026-10-05

## Decision

Reject Q128/K64 for the tested family. All six cells regress by 8.08–19.10%
against the accepted Q64/K64 page-batch artifact. The fixed gate remains five
paired reductions each at least 3%; no threshold or workload was changed.
Production frontier, ownership and fixtures are restored exactly. No serving
bundle was rebound. This is a kernel experiment, not an end-to-end speedup or
a fresh vLLM/SGLang comparison.

One finite schedule family, three workload cells on each of .178 and .179,
five alternating pairs per cell, 30 event repeats per pair. Timings ran on both
hosts concurrently; subsequently .179 collected selected-kernel counters while
.178 ran short sanitizer checks and long-context memcheck. GPU jobs were
serialized per host; CPU compilation did not overlap unprofiled measurements.
Both GPUs were idle at completion.

## Hypothesis and functional compiler boundary

Extend the pure query-owned legal frontier from Q32/Q64 to Q128, retaining
16-row subgroup ownership, K64, D128, one pipeline stage and the existing
split-copy effect plan. Q128 reuses a KV tile across eight query subgroups
instead of four. Existing physical IR, fragment fold and CUDA terminal lowering
already express this geometry; no new CUDA renderer was required.

Experimental changes were limited to `query_owned_frontier.mbt`,
`compiler_ownership.mbt` and their fixtures in `kernels/luna_attention_strategy`,
plus a source-emission regression covering dense/paged and partitioned/non-
partitioned variants. Stable ID 1022 identifies the strict candidate; 31022
identifies the explicit `approx-base2-f32-v1` law. Resource tests verified
256-thread admission and the exact 65,552-byte shared-memory boundary. No
model-name policy, public API, request-path JIT or benchmark dependency was
introduced. Experimental source/tests are preserved outside production.

The hypothesis was reduced copy/address work, not reduced mathematical work.
The measured result supports that reduction but rejects the performance claim.

## Unprofiled paired results

Q2048, runtime rows envelope 32. Q64 uses block128/grid63×16×1; Q128 uses
block256/grid47×16×1, including inactive runtime-envelope CTAs. Reported gain is
the median paired ratio, not a ratio of medians. Negative reduction is slower.

| Host | Rows / history | Q64 µs | Q128 µs | Paired reduction | Worst pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| .178 | 1 / 28,672 | 6383.827 | 7452.798 | −16.74% | −18.30% |
| .178 | 2 / 28,672 | 6377.887 | 7595.880 | −19.10% | −20.35% |
| .178 | 2 / 8192 | 1943.518 | 2107.869 | −8.08% | −12.94% |
| .179 | 1 / 28,672 | 6605.128 | 7524.795 | −14.67% | −16.16% |
| .179 | 2 / 28,672 | 6612.778 | 7695.971 | −17.84% | −18.73% |
| .179 | 2 / 8192 | 1981.595 | 2164.932 | −9.74% | −11.44% |

## Matched executed counters

.179 Q2048/R2/H28672, baseline then candidate. Instrumented duration is excluded
from the decision. Warp-latency fractions are not elapsed-time attribution.

| Metric | Q64 | Q128 |
| --- | ---: | ---: |
| Total warp instructions | 891,530,112 | 792,951,552 |
| Async `LDGSTS.E.BYPASS.128` | 14,958,592 | 7,487,488 |
| `IMAD.WIDE.U32` | 7,342,080 | 3,672,064 |
| `IADD3` | 12,696,320 | 5,975,552 |
| `LOP3.LUT` | 33,782,784 | 8,562,688 |
| `MOV` | 97,258,496 | 97,359,872 |
| `NOP` | 25,242,624 | 36,501,504 |
| Tensor `HMMA` | 119,668,736 | 119,799,808 |
| CTA barriers | 2,808,832 | 2,811,904 |
| Registers/thread / allocated | 234 / 240 | 236 / 240 |
| Dynamic shared bytes | 49,168 | 65,552 |
| Register/shared-limited resident CTAs | 2 / 2 | 1 / 1 |
| Active warps, percent of peak | 16.03% | 16.67% |
| Average warp latency per issued instruction | 6.156 | 8.719 |
| Long-scoreboard / warp latency | 11.92% | 11.41% |
| Barrier / warp latency | 3.22% | 7.29% |
| Math-pipe throttle / warp latency | 16.01% | 23.83% |
| Wait / warp latency | 35.56% | 26.81% |
| Issue active | 31.24% | 25.09% |

Copy instructions nearly halve and total instructions fall 11.06%. Tensor work
changes only 0.11% (causal/tail schedule effects), not by a factor of two. The
extra query reuse does not remove fragment/softmax work: MOV counts remain
nearly identical. Wider CTA publication preserves essentially the same executed
warp barrier count, but barrier-latency contribution more than doubles. Math-
pipeline pressure also rises and effective issue rate falls.

Both register and shared-memory limits collapse to one resident CTA. Eight
warps in one CTA are not interchangeable with four warps in each of two CTAs:
independent workgroups can hide each other's collective waits. Active-warps
occupancy itself does not fall, so attributing the result simply to "lower
occupancy" would be wrong. These observations support a collective/dependency-
overlap problem, but do not independently prove that the resident-CTA change
alone causes the regression. This capture does not include DRAM-byte counters;
halving async instructions is not a claim that measured DRAM traffic halved.

The hottest sampled load-dependent PC remains the historical-page identity
comparison, `ISETP.GE.U32.AND ... 0x4000`, in both kernels. Moving/removing copy
instructions did not remove that dependent load chain. Do not interpret the
comparison opcode as an expensive arithmetic operation by itself.

Next hypothesis, not implemented or measured here: reduce per-subgroup
fragment/softmax dependency depth and collective publication cost while keeping
independent-CTA overlap. Merely increasing query reuse or changing page-address
arithmetic is not supported as the next production change.

## Correctness, limits and reproducibility

All 30 timing pairs are bitwise equivalent, maxabs zero against the accepted
artifact, and satisfy the 0.003 sampled BF16 oracle ceiling. Short Q129/R2/H128
memcheck, racecheck and synccheck pass with zero errors/hazards; oracle maxabs
0.000330008. Long Q2048/R2/H28672 memcheck passes with zero errors and oracle
maxabs 0.000377474. Candidate reports 236 registers, one resident CTA and zero
local bytes. This is not whole-model quality admission.

Experimental affected tests passed 191/191: strategy31, source89, schedule39,
physicalIR32. Restored production passes 189/189: strategy30, source88,
schedule39, physicalIR32. Offline helper tests passed 6/6.
Scoped warning-denied native checks use existing exclusions 20/79/29/25; this
is not a clean-whole-tree claim. `moon info` changed no owned public API.
An initial remote build used an invalid Moon working-directory option and was
corrected before GPU work. A new source fixture initially used byte
reinterpretation rather than UTF-8 decoding and was corrected before passing.

Both GB10/sm121 devices were checked:

- .178 `GPU-9c3d3cf0-439a-5da2-67e9-20414255879f`
- .179 `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`

GPU units: MemoryMax16GiB, MemorySwapMax0, bounded tasks and 600-second runtime.
Minimum paired before/after MemAvailable was 121,940,724KiB, above the 32GiB
reserve. Existing host swap is not asserted to be zero; outer helper unit
peaks are not whole-serving peak memory.

Baseline cubin `7b42b3531466f5fe22c59e5aac2883db68984907c49a5637ac5f1358ad2eab57`.
Candidate source `876dc7c5d24476e4a343e1c9f858da98597804226ca7587a6079c5e34bd31911`.
Candidate cubin `9d61300585ea40b17ea9b490f722f4a184a3bff3a3e1981163617c185444a60c`.
Two offline compilations match. CUDA13.0.88 nvcc SHA256 remains
`fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.

Remote experiment roots:

- .178 `/home/wlc003s/lunaflux-ako-query-reuse128-20261005.TErzRGA0/experiment`
- .179 `/home/wlc004s/lunaflux-ako-query-reuse128-20261005.HIjqI1mF/experiment`

Local `/tmp/lunaflux-ako-query-reuse128-20261005.Y9I7X71W` preserves the final
experimental source/tests, exact artifacts, raw paired results, capture and
parsed report. Downloaded archives and all manifest entries verified:

- .178 `691053b80c15f38ef87bdbd418ebe7d84640a73f1f54e2665410777f070191d5`
- .179 `c6034d285b4fc335576cb2e59b03055cf766f1a0699502a9c1cdcf3f875f6366`
