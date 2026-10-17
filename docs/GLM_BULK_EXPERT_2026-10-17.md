# Bulk expert transport and persistent consumers

Continuation of `GLM_EXECUTION_REPAIR_2026-10-17.md`. Gate/up remains about
74 ms in the selected warmed prefill; the previous buffering/read-ahead
campaign did not materially change it. All other open families remain in that
report's ledger; this document does not close them by implication.

First hypothesis: issue granularity and work reuse must change together.
Test at most three coupled schedules against the frozen actual source:
packed byte-preserving rows, byte-counted bulk row transfers, explicit
producer/consumer epochs, wider independent output ownership and bounded
persistent workers. Start with 64/128 columns and one/two consumer teams.
This is bulk **row** transfer, not descriptor-based multidimensional TMA.
Its unswizzled consumer layout is a measurable tradeoff, not presumed faster.
If this loses, inspect transaction/instruction/dependency evidence before
adding another buffering variant. Descriptor-based tensor transfer remains a
distinct alternative with startup-owned descriptor lifetime requirements.

The numerical program is unchanged: existing quantized bytes/scales, ascending
K64 matrix folds, and declared BF16 publication boundaries. Physical plans own
the worker/consumer cardinalities; CUDA lowering owns instructions and barriers.
Comparison, counters and selection evidence stay outside the compiler.

Require exact complete-chain outputs, tails, invalid/padded rows, memory/race/
synchronization/leak checks, five alternating timing pairs, and actual serving
selection before claiming a model gain. Include packing and quantization time.
Keep all failures and losing shapes. The retained vLLM component uses different
physical padding; a component comparison alone does not explain the full-model
completion gap.

Original GLM remains running during source work. GPU tests temporarily stop
only its head/worker, leave NFS intact, and restore the original containers.
Every task owner stays within 100 GiB, no swap, independently observed host
MemAvailable at least 16 GiB. No payload hashing or integrity scans.

## First three candidates: rejected for selection

All retain exact full-chain output and packed payload/scales on 1/3/22/64 rows,
both capacity and diagnostic trimmed grids. Roots are
`/tmp/lunaflux-glm-fp4-performance-bulk-20261017-v{1,2,3}`. At 22 rows the
capacity chain changes approximately 6.76→10.38 ms (64 columns/one consumer
team), 6.78→10.25 ms (128/two teams), and 6.76→10.01 ms (padded rows).
These are regressions, not production winners. No full-model gain is claimed.

V3 instruction-level SourceCounters demonstrate the mechanism: varying row
addresses within the producer warp lower to repeated ELECT, R2UR.BROADCAST,
UBLKCP and BRA.U.ANY. Issuing a bulk operation per lane is serialized. Consumer
readiness polling dominates its extra executed instructions (three polling
instructions each execute 120,147,483 times in that instrumented gate/up
capture). SourceCounters counts differ from the shorter metrics capture;
polling depends on profiling perturbation. Do not combine those captures.
In the metrics capture, gate/up is 6.253 versus 4.194 ms, 251.0 versus 96.5M
instructions; L2 read sectors fall 36.49→27.85M and shared-load conflicts
1.270M→0.317M. Layout alone is not the answer.

New hypothesis, bounded to two executable schedules: represent each operand
as contiguous physical tiles, preserving every encoded byte and scale, so one
elected producer issues a whole-tile transfer. Transform immutable weight
storage once at startup using one-expert scratch, never each token; activation
packing remains an explicit effect included in timing. This is an alternative
to tensor-descriptor TMA, not a claim to have implemented that API. Its added
startup cost, scratch ceiling, exact permutation, launch propagation and whole
chain must all be checked. Persistent workers and numerical K order remain
independent immutable plans. Retain the failed row-transfer experiments.

## Contiguous tiles: demonstrated component benefit, serving test pending

The first whole-tile schedule succeeds. Weight payload and scales are permuted
bijectively once at startup through one-expert scratch; activation packing uses
the same physical-domain plan. One elected producer issues six whole-plane
copies per paired gate/up K epoch. Persistent 48 workers, two consumer teams,
two slots and the existing ordered K64 arithmetic are explicit schedule choices,
not new model semantics. Ordinary source/default selection is unchanged.

Root `/tmp/lunaflux-glm-fp4-performance-tiles-20261017-v2` compares the frozen
pre-lookahead source with this candidate, using all 288 distinct expert banks,
five alternating pairs of 20 iterations, and including activation packing and
quantization in the chain:

| Capacity workload | Retained median | Whole-tile median | Time reduction |
| --- | ---: | ---: | ---: |
| 1 row | 361.33 us | 301.43 us | 16.58% |
| 22 rows | 6.7586 ms | 5.2136 ms | 22.86% |
| 64 rows | 13.1398 ms | 10.1793 ms | 22.53% |

All eight 1/3/22/64-row capacity/trimmed cases match complete outputs and the
independent packed payload/scale oracle bitwise, with no nonfinite values.
V2 passes memcheck/full leak, racecheck and synccheck. V3 repeats those gates
using the actual one-expert scratch plus copy-back startup helper, rather than
a separate whole-bank destination; all three pass without swap. These probes
exercise three live rows with all distinct banks, not every production shape.

Separate matched metrics capture at 22 rows:

| Metric | Retained gate/up | Tile gate/up | Retained down | Tile down |
| --- | ---: | ---: | ---: | ---: |
| GPU time (ms) | 4.188 | 3.162 | 2.161 | 1.632 |
| Warp instructions (M) | 96.530 | 113.701 | 49.418 | 66.889 |
| L2 read sectors (M) | 36.488 | 24.384 | 18.230 | 12.948 |
| Active warps (%) | 45.89 | 18.75 | 81.10 | 18.75 |
| Long-scoreboard contribution (%) | 77.74 | 37.18 | 73.34 | 55.73 |
| Barrier contribution (%) | 19.67 | 1.62 | 13.06 | 6.14 |

The faster candidate issues more total instructions and has lower occupancy.
The causal change is coarser operand delivery with reusable physical tiles and
separate producer/consumer ownership, reducing traffic and dependency waits.
Neither occupancy nor instruction count alone explains completion time.
Shared-load bank counters remain nonzero; zero conflicts is not this objective.

This does not yet explain the complete 42.12% model gap or establish a model
speedup. The retained production trace puts gate/up, down and shared experts
ahead of the sparse-attention leaf; collective duration also includes peer
waiting. The installed CUTLASS component uses different physical padding and
a separate capture. Its 3.247/1.745 ms cannot be used to claim a vLLM model win.
Next gate: commit-pinned startup selection, bounded full-model run, matched
output lengths and original-service restoration. Other ledger items remain open.
