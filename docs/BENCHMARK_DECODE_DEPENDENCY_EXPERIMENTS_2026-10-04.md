# Why the next decode rewrites do not improve C16

Five alternatives were tested against the final qualified owned8 c468 decode
from the [preceding serving campaign](BENCHMARK_SPARK_CLOSURE_2026-10-04.md).
There is **no demonstrated new C16 serving speedup**. Experimental compiler
changes were removed; sources and results remain preserved. The existing
optional FMA capability was neither removed nor promoted.

## Corrected diagnosis

A hot sampled `FMNMX` does not prove that max arithmetic itself dominates.
The old instruction immediately preceding it loads the shared score:

```text
@P6 LDS R75, [R36+0x8000]
@P6 FMNMX R141, R75, -INF, !PT
```

Register score transfer removed this exchange. Its hottest short-wait sample
then moved to the subtraction feeding softmax, following the maximum broadcast.
Some long-wait samples occur at constant `MOV` or control instructions. A
sampled PC alone establishes neither the producing load/register hazard nor
an additive wall-time contribution.

The offline SASS summarizer now preserves eight preceding and two following
instructions for each top sampled PC. These are static context windows, not
inferred dependency edges. The paired summary adds eligible warps, L2
throughput and requested DRAM counters. DRAM byte/throughput counters are
**unavailable in these captures**, not zero. This round does not prove saturated
DRAM by dividing nominal tensor sizes by time.

## Alternating unprofiled timings

GB10/sm121, CUDA 13.0.88, 48 SMs; the same frozen model shape, fragmented
page mapping and traced 32-row launch bucket. Five alternating before/after
samples per cell. C1/C8/C16 are real active rows, not the launch bucket.
Medians are microseconds; positive percentages mean **slower**.

| Experiment | C1, H4096 before → after | C16, H4096 before → after | C16, H8191 before → after |
| --- | ---: | ---: | ---: |
| Narrow per-vector K→V offsets | 169.865 → 164.658 (−3.07%) | 1174.420 → 1171.538 (−0.25%) | 2371.873 → 2372.168 (+0.01%) |
| Register QK-score exchange | 170.103 → 140.417 (−17.45%) | 1181.612 → 1198.108 (+1.40%) | 2373.348 → 2355.352 (−0.76%) |
| Existing explicit owned8 FMA law, c470 | 166.820 → 153.799 (−7.81%) | 1178.432 → 1179.586 (+0.10%) | 2391.128 → 2371.837 (−0.81%) |
| Independent QK-chain interleave | 162.660 → 150.849 (−7.26%) | 1177.983 → 1194.080 (+1.37%) | 2378.435 → 2371.263 (−0.30%) |
| Separate successor page-table epoch | 166.559 → 172.377 (+3.49%) | 1182.485 → 1175.519 (−0.59%) | 2370.996 → 2375.437 (+0.19%) |

C8/H4096 changes are respectively −0.10%, +1.46%, +0.06%, +1.21% and +3.11%.
Small C16 deltas do not establish statistical significance. Ordinary C1 gains
are not serving gains: the selected small-batch long-history route can use a
faster partitioned entry. None justified replacing the C16 path.

## Exact selected-kernel counters

Cold-cache paired Nsight Compute replay of C16/H4096; not ordinary timing,
service latency or a cross-framework run.

| Experiment | Warp instructions before → after | Registers/thread | Replay µs before → after |
| --- | ---: | ---: | ---: |
| Narrow vector offsets | 41,646,848 → 40,937,472 | 148 → 153 | 1248.736 → 1261.088 |
| Register score exchange | 41,646,848 → 41,776,640 | 148 → 148 | 1224.224 → 1251.616 |
| Explicit FMA | 41,646,848 → 33,084,416 | 148 → 152 | 1238.912 → 1238.528 |
| Independent chain interleave | 41,646,848 → 41,646,848 | 148 → 148 | 1236.160 → 1235.296 |
| Successor metadata prefetch | 41,646,848 → 41,685,248 | 148 → 154 | 1237.120 → 1237.536 |

All counter pairs report zero local spilling requests and zero source-correlated
excessive shared wavefronts. Shared allocation remains 34,064 bytes. This does
not prove that every hardware bank counter is zero.

- Offset reuse saves 1.70% of instructions but retains more live state. It
  establishes neither a C16 gain nor a register-pressure causal attribution.
- Register score exchange lowers long-scoreboard/active-issue ratio
  2.79 → 2.13, but adds shuffle/selection work. Short-scoreboard is approximately
  unchanged, 3.42 → 3.45; time regresses.
- FMA saves **20.56%** of instructions with essentially unchanged replay time.
  Eligible warps/cycle fall 0.15 → 0.12; issue activity falls 12.72% → 10.28%.
  Waiting ratios are normalized per active issue: their increases are not
  equivalent increases in absolute waiting time. Arithmetic compression did
  not shorten the overall critical path.
- Explicit interleave gives identical total instruction count/resource use.
  This is consistent with the device compiler already exploiting much of that
  freedom, not proof of byte-identical SASS.
- Existing epoch reuse loads an epoch when first needed. The successor cache
  moves that load earlier; long-scoreboard ratio falls 2.74 → 2.45. Eligible
  warps stay 0.15, registers rise by six, and C16 time is neutral. The exposed
  lookup is real, but moving it alone does not eliminate the serving gap.

## Functional boundaries and disposition

Experiments used immutable ownership/domain plans and explicit terminal CUDA
realization. The four same-law transformations preserve tested bitwise outputs.
FMA retains a separate explicit numerical law and independent FP64 oracle;
these fixtures do not make contraction universally bitwise-equivalent.

No model/scheduler CUDA branching, request-path JIT/authentication, global
mutable state or production dependency was introduced. Rejected plans,
renderers and tests were removed rather than kept as unused compiler branches.
Only the offline diagnostic improvements and report are retained. Further
changes need a demonstrated critical-path/whole-chain benefit, not merely fewer
instructions or a relocation of sampled stalls.

Latest verified serving rates remain **220.93 tok/s** at 4096/64 C16 and
**319.53 tok/s** at 4096/256 C16. These are the preceding three-round results,
not fresh throughput measurements. Their completion gaps are 10.33%/9.86% and
8.31%/6.66% versus vLLM/SGLang respectively. No production deployment or new
cross-framework serving campaign was run.

## Validation and preserved results

Each qualified experiment passed deterministic double compilation, the
21-cell C1/C8/C16 history matrix through 8191, the independent oracle,
memcheck/leak, racecheck and synccheck. GPU work was serialized; counter
containers were limited to 8 GiB without swap, retaining 32 GiB host memory.
An initial vector-offset compile failed unused-variable Werror; it was corrected
and qualified in a new directory. Failed sources/logs remain preserved.

The complete experimental native suite passed **4,355/4,355** using existing
toolchain-migration exemptions `-79-20-29-25-92-14`. An earlier run hit the
unrelated zero-wait TCP timing test; its focused rerun passed 54/54 and the full
rerun passed. This is not an unsuppressed warning-denied claim.
After cleanup, retained-source native check and formatting check pass; the
full suite passes **4,353/4,353**, affected packages pass **115/115**, and both
offline diagnostic helpers pass their regression test. The same warning
exemptions apply. Rejected compiler implementations leave no local source diff.

Twelve explicit roots were snapshotted without replacing originals: 10,595
readable files and 79,151 declared exclusions (build/toolchain/source trees).
The archive preserves overlays, rejected kernels, recipes, qualifications,
raw counters and instruction context. Its downloaded bytes match SHA-256:
`87d9d472da892b1ec22e8f3afc7a78aa5ac5dd830708b96b16ba179622102f25`.

Local archive/inventory:
[`benchmarks/results/next-decode-20261004.SdsG9n7e`](../benchmarks/results/next-decode-20261004.SdsG9n7e/).
It was not extracted into the MoonBit module, where source snapshots would
become accidental build inputs. Remote archive:
`/home/wlc004s/lunaflux-next-round-archive-20261004.A7PsYRgO/archive`.

Counter roots under `/home/wlc004s/`:
`lunaflux-vector-origin-counter-20261004.Xc4VmYRB`,
`lunaflux-register-score-counter-20261004.gIYE0EPG`,
`lunaflux-owned8-fma-counter-20261004.74jPKPdI`,
`lunaflux-interleave-counter-20261004.s3sLYyEq`,
`lunaflux-metadata-prefetch-counter-20261004.DQTv6tXV`.
Original evidence remains intact; no serving artifacts were overwritten.
