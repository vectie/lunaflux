# Dual-Spark value-readiness dependency cut, 2026-10-05

## Decision

Retain the existing c30322 implementation. Moving value-independent softmax
work before the V wait lost all six paired timing cells. The experimental
production-source changes and their temporary public API were removed; no
unused compiler pass, default-route change or slower serving path remains.
The diagnostic driver, source input archive and measurements are retained.

Both Sparks ran timing concurrently. Afterwards `.178` ran sanitizers while
`.179` captured matched counters. This is an attention-kernel experiment, not
a refreshed vLLM/SGLang comparison or new end-to-end token throughput.

## Hypothesis and bounded implementation

The typed online fold records reads/writes explicitly. Its maximal ordered
prefix before V readiness is ScoreProduct → PrepareScores → MergeMaximum →
RescaleOutput. Only the remaining AccumulateProbabilityValue →
UpdateDenominator needs V. An experimental pure dependency cut derived that
prefix without reassociation or changing the admitted approximate exponential
law. CUDA lowering carried `next`, `alpha` and `sum` across the uniform V wait;
all transfer/publication/release effects remained in their original order.
There were no model-name branches, request-time checks or JIT.

One alternative, Q2048 with R1/H28672, R2/H28672 and R2/H8192; five alternating
pairs per cell, 30 GPU-event repeats per member. Frozen baseline cubin reused,
experimental source AOT compiled twice with identical hashes. Both are c30322
schedule geometry; **their source/cubin hashes, not candidate number alone,
distinguish compiler implementations**. This diagnostic does not add a catalog
candidate or permit runtime mixing of two identities.

CPU build capped at 8 GiB; GPU jobs at 16 GiB, zero swap, runtime 600 seconds
and 32 GiB MemAvailable reserve. No CPU compilation during GPU timing. One
initial preparation invocation used a wrong helper executable path and exited
127 before creating experiment inputs or running GPU work; a fresh unit used
the actual `.mbtx` build path. Preserve the failed unit status.

## Unprofiled results

Times are median microseconds; paired change is median(candidate/baseline−1),
not the ratio of the two medians. Positive changes are regressions.

| Host | Requests/history | Baseline µs | Dependency-cut µs | Paired change |
| --- | --- | ---: | ---: | ---: |
| .178 | 1/28672 | 7171.59 | 7200.70 | +0.61% |
| .178 | 2/28672 | 7147.54 | 7230.47 | +1.05% |
| .178 | 2/8192 | 2098.12 | 2117.75 | +0.80% |
| .179 | 1/28672 | 7431.02 | 7491.13 | +1.63% |
| .179 | 2/28672 | 7386.66 | 7495.70 | +1.87% |
| .179 | 2/8192 | 2171.58 | 2187.91 | +0.75% |

Both timing jobs finished in about eight seconds. The earlier progress update's
3.4% upper bound referred to a losing individual pair, not the final median.

## Matched Q2048/R2/H28672 counters

| Metric | Frozen baseline | Dependency cut |
| --- | ---: | ---: |
| Registers/thread | 234 | 243 |
| Resident CTAs/SM | 2 | 2 |
| Dynamic shared bytes | 49,168 | 49,168 |
| Executed warp instructions | 936,583,040 | 936,520,384 |
| MOV | 100,893,696 | 109,352,960 |
| NOP | 25,242,624 | 29,917,184 |
| Async LDGSTS | 14,958,592 | 14,958,592 |
| Non-transposed LDSM | 37,396,480 | 37,396,480 |
| Transposed LDSM | 29,917,184 | 29,917,184 |
| HMMA | 119,668,736 | 119,668,736 |
| Workgroup barriers | 2,808,832 | 2,808,832 |

The new ordering does not remove physical work. It extends live state, adds
8.46M MOVs and 4.67M NOPs, and consumes nine more registers without changing
CTA residency. Average warp latency grows 6.44→7.08; barrier contribution
2.24%→4.60%, long-scoreboard contribution 14.06%→12.93%. These are one matched
instrumented capture, not a complete causal allocation of the timing loss.
NOP counts are not stall cycles, and percent reductions are not speedups.

The effect plan also places next-K issue after the V publication. Moving
softmax/rescaling before that boundary delays next-K issue as well: this is
not free overlap. The experiment rules out this phase cut as an improvement,
but does not prove that V waiting accounts for the main framework gap. A
future alternative must account for producer issue timing and carried-register
cost jointly, rather than just placing more arithmetic before a wait.

## Checks and retained artifacts

All 30 paired observations pass bitwise baseline equality and the fixed sampled
FP64 oracle ceiling. Irregular Q129/R2/H128 memcheck, racecheck and synccheck
pass with zero errors/hazards. This is not whole-model quality equivalence.
Minimum recorded MemAvailable: 122,398,892 KiB (`.178`), 121,736,112 KiB
(`.179`). Both GPUs idle at completion.

The experimental source passed 119 affected native tests after intended source
snapshot updates; after removing it, the original 117 tests pass and package
interfaces/source snapshots are unchanged. Existing warnings 20/79/29/25 were
excluded in affected-package tests. Both diagnostic `.mbtx` helpers pass
warning-denied tests without exclusions. Unrelated dirty work was untouched.

Local root: `/tmp/lunaflux-ako-value-overlap-20261005.cM9plZqT`, including
`report.json`, `source-patch.tar.gz`, `host178/` and `host179/`.

- `.178`: `/home/wlc003s/lunaflux-ako-value-overlap-20261005.Ft1AYP3V/experiment`.
- `.179`: `/home/wlc004s/lunaflux-ako-value-overlap-20261005.0pi1bN5M/experiment`.
- Baseline cubin: `66881bc0055ce4c4340e6001e7746d0abd64a8b7f9a02b741e6b87463f0e2922`.
- Experimental source: `9d91671865c31f613f69caaf0f83b946db9b73a2c8917f3e67b73726a8975f22`.
- Experimental cubin: `cfd0c1e97c667f1338671e65e78d791d22373d6b2a9978eb916faa30753349ac`.
- Source-input archive: `6b7f02cff492ad9dac6021d1b9950d92ccbab8aef9371708f2a5f5d94a844d80`.
- `.178` sealed archive: `80804e60a1a8b925ee64c74a142e7a9f9d7bf2598f7d3edd469890fe9e42aec3`.
- `.179` sealed archive: `55cfa4dc2f5bf6552410d3e54441e85b771719f7da47550383213cfe8b169c6d`.

Both downloaded archive hashes match and their complete manifests verify locally.
