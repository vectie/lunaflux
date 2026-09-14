# Fence placement experiment

Query-owned, single-slot split K/V attention can publish next K at the current
V consumer-release barrier. A prologue publishes the first K. This changes
three per-iteration CTA barriers to two without changing QK/softmax/PV order.
The final outer-tile barrier remains for empty partitions and tile reuse.

Five actual-row shapes passed independent sampled scalar checks, unchanged KV
and bitwise output comparison. Memcheck passed; racecheck and synccheck passed
on four target launches of the long-history case. An initial unbounded
racecheck timing-loop run was intentionally terminated and is not a pass.

The first timing comparison used the previously published binary. A second
control rebuilt the original source with the exact same nvcc flags as the
experiment (`-arch=sm_120 --cubin -O3 -Xptxas=-v`). Both use 225 registers.
Use this matched control for attribution, not the old 228-register artifact.

Five alternating timing trials, median microseconds:

| Shape | Separate fences | Combined fences |
| --- | ---: | ---: |
| 2048 fresh queries | 532.32 | 531.47 |
| 2048 queries, 2048 history | 1416.86 | 1419.85 |
| C8 short prefill tail | 242.20 | 245.36 |
| C16 mixed step | 670.19 | 664.82 |
| C16 final prefill tail | 264.76 | 265.21 |

There is no consistent win. The compiler source experiment was removed from
the production generator; the generic dependency-plan description remains for
future measured schedule search. Fewer barriers alone do not establish lower
latency. This is not a full-model benchmark, a new source-export campaign, or
a measured instruction-counter comparison.

Local/remote archive: `/tmp/lunaflux-fence-experiment-20260914.tar.gz`.
Raw results are under `/tmp/lunaflux-barrier-control-20260914` and
`/tmp/lunaflux-barrier-trial-20260914-r2`; the archive includes diagnostic
transformation drivers and compiled modules.

The complete local native suite passed 3762/3762 with the experimental
generator before reverting it. The retained production generator is unchanged;
affected-package checks are rerun after that revert. Cost decomposition is
behavior-preserving, not a newly calibrated cost model.
