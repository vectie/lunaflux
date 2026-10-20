# Row-reusing projection pipeline and chunk handoff

## Contract and experiment budget

This follows the `09bd656d` sparse/HC candidate, not the earlier M16 projection
baseline. The remaining cold rank-1 projection excess is 15.963 ms in one
64-token / 45-layer chunk. The retained vLLM trace is from October 19; it is
not a new reference run.

The measured sparse output shape is M64/N4096/K8192. Its current M32/N32/K64
schedule reduces instructions, but still stages operands through registers
and waits before shared-memory publication. The existing two-slot transport
supports only M16/N64: it cannot currently compose with row reuse.

Hypothesis: composing row reuse with the existing two-slot transport can
remove synchronous operand publication waits without changing the ascending
K16 numerical fold. The alternative hypothesis is that address/copy
instructions and the doubled shared-memory footprint outweigh that overlap.
The budget is two candidates: M32/N32 with buffered K64 and K128. Their
baseline uses the latest row-reusing synchronous K32/K64 choice. The older
losing M16 buffered schedule is not being retried.

The change belongs to the immutable matrix plan's legal schedules and the
private CUDA ownership lowering. Defaults and serving selection stay unchanged
until exact-source correctness, bounded sanitizer/leak checks, alternating
paired measurements, and selected-path traces justify a change.

A separate handoff investigation concerns the 9.013 ms warm rank-1 interval
without recorded GPU work. It must identify CPU/control dependencies before
removing waits. Cancellation, KV retirement and two-rank commit ordering are
not optional. One handoff change may be tested after causal tracing.

## Validation and operating boundaries

- No overall benchmark, payload hashes, or integrity scans.
- Serialize GPU work; at most 100 GiB per Spark, at least 16 GiB host reserve,
  and zero owned-workload swap. Component probes use 2 GiB limits.
- Preserve the actual capacity-reserved launch grid and test live-row tails,
  decode, inactive rows and nonaligned K.
- Keep the old qualified GLM service available for restoration.
- Record nonwins rather than promoting an experiment to claim completion.

## Two-slot result: rejected for the selected GLM shapes

Actual source export:
`/tmp/lunaflux-glm-projection-staging-buffered-rows-export-20261020-v1`.
GPU trial root:
`/tmp/lunaflux-glm-projection-staging-buffered-rows-20261020-v2`.
The empty v1 preparation is retained: expired SSH masters prevented any GPU
work. Connection masters were re-established before v2.

The immutable legal schedule now composes M32/N32 ownership with buffered
transport. CUDA lowering assigns both row and column fragments accordingly.
No numerical policy or default selection changed.

Five alternating pairs, 64 live rows, capacity-reserved launch grid:

| N / K | Sync / buffered K64 (µs) | Sync / buffered K128 (µs) |
| --- | ---: | ---: |
| 4096 / 4096 | 212.288 / 237.816 | 200.306 / 225.035 |
| 4096 / 8192 | 410.648 / 523.434 | 429.614 / 612.552 |
| 8192 / 1536 | 119.678 / 128.914 | 115.682 / 135.562 |
| 4096 / 1536 | 51.430 / 69.640 | 51.523 / 71.597 |

These are separate paired runs, not a common-baseline speedup calculation.
Every tested row count (0, 1, 3, 22, 36, 64, invalid 65) and both odd-K
fixtures preserved all output/inactive bits. The two tiny tail fixtures are
correctness coverage, not a reason to promote a losing large-shape path.
Both candidates passed memcheck/full-leak, racecheck and synccheck on the
K33/K136 fixtures at 1/3/64 rows under 2 GiB/no swap. This is not a claim
that the large full-grid sanitizer campaign was run.

Selected N4096/K8192 counters, actual exported generated sources:

| Metric | Sync K64 | Buffered K64 | Buffered K128 |
| --- | ---: | ---: | ---: |
| Warp instructions | 16,872,448 | 22,737,920 | 18,011,136 |
| L2 read sectors | 6,408,590 | 8,391,726 | 8,392,242 |
| Registers/thread | 40 | 40 | 40 |
| Allocated shared bytes/block | 13,312 | 21,504 | 37,888 |
| Shared-memory block limit | 7 | 4 | 2 |
| Active warps | 44.28% | 28.04% | 16.05% |
| Barrier stall share | 13.52% | 18.82% | 17.08% |
| Long-scoreboard share | 57.43% | 57.84% | 59.05% |
| Tensor activity | 7.23% | 6.70% | 5.54% |

The next-copy overlap is insufficient: the K64 path increases instructions
34.8%, both alternatives increase L2 traffic about 31%, and shared-memory
residency falls. The private transport uses `.cg` copies, whereas synchronous
loads can use a different cache path; these counters alone do not isolate
cache policy from traffic caused by scheduling. The vLLM TMA implementation
has larger work ownership and a different producer pipeline, so copying just
its use of asynchronous movement is not an equivalent schedule.

Decision: keep both alternatives explicit, nondefault and unselected in GLM.
The two-candidate budget is exhausted; no unchanged full-model run is needed.
This does **not** fix the remaining projection gap.

## Handoff candidate (not yet qualified)

The existing GLM child unconditionally sleeps one millisecond whenever its
nonblocking inherited-pipe read is pending. This is after the previous
two-rank commit and before the next plan or request-retirement frame. A frame
arriving during that interval cannot wake the sleep.

The candidate uses the existing bounded inherited-channel readability wait
instead, followed by a cooperative cancellation point. It retains the same
one-millisecond maximum idle blocking bound, pending-frame parser, EOF path,
absolute deadlines, request-retirement protocol and two-rank commit ordering.
It does not move HTTP, model or CUDA decisions into the scheduler or add an IR
layer: the logical plan is unchanged; this changes one runtime effect's wake
condition. No GPU work is outstanding while this wait is used.

This source-level timer explains at most one portion of the handoff. It is
not yet proof of the whole 9 ms interval, nor permission to skip commit.
The first CPU-enriched capture
`/tmp/lunaflux-glm-http-handoff-osrt-20261020-v1` stopped at the 17 GiB
diagnostic headroom threshold (lowest observed 17,465,084 KiB, above the
16 GiB floor). Its failed result is retained. A second unchanged-baseline
capture filters OS-runtime events shorter than 100 µs and backtraces shorter
than 1 ms to bound profiler overhead; no model capacity is reduced.

## Revised projection hypothesis

The two vector-copy nonwins motivate one further **different** experiment:
publish contiguous operand row ranges using the existing expert path's
byte-counted transfer mechanism. Keep M32/N32/K64, two slots, four consumer
subgroups and ascending K16 arithmetic fixed. One producer subgroup issues
row copies; publication completion and reader retirement remain explicit.
This isolates transfer granularity, not a new numerical program or a larger
tile. The new pure `BulkBuffered` schedule has an aligned-row domain; private
CUDA lowering owns byte barriers and instructions. Its budget is one K64
candidate. It is not selected by serving configuration.
