# Row-reusing projection pipeline and chunk handoff

Outcome: two runtime ordering/wakeup defects are corrected. Their combined
candidate shortens the four observed chunk cadences by 0.37–0.97%; this is not
a whole-request benchmark or a solution to the main projection gap. Three
projection transport candidates lose and remain unselected or were removed.
The final ablation uses matched rank-1 graph tracing; it does not replace the
earlier two-rank, per-kernel allocation. Full results and limitations follow.

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
not optional. The original handoff budget was one change. The subsequently
observed local-before-peer submission dependency adds one separately measured
candidate; no other handoff changes are bundled into these ablations.

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

More specifically, `progress_read_frame_or_eof` intentionally returns `None`
after consuming the four-byte header, even if the entire payload is already
queued. The old caller confuses "no complete frame yet" with "no readable
data": it forces a timer sleep at **every header-to-payload transition**.
The new inherited-pipe fixture proves `None`, followed by immediate readable
payload, unchanged absolute deadline, exact bytes and clean EOF. This is a
source-level defect independent of any attribution of the whole GPU gap.

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
tile. The trial used an aligned-row domain; private CUDA lowering owned byte
barriers and instructions. Its budget was one K64 candidate.

It failed: N4096/K8192 takes 1741.741 µs versus paired baseline 412.085 µs.
The other three large shapes also regress about 3–4x. Output and inactive
bytes remain bitwise across all tested row counts. Therefore the temporary
`BulkBuffered` enum and source dispatch were removed before committing any
production-facing support. The experiment's MoonBit emitter/export and exact
CUDA source are preserved outside the compiler at
`/tmp/lunaflux-glm-projection-staging-bulk-rows-export-20261020-v1`;
raw measurements remain in
`/tmp/lunaflux-glm-projection-staging-bulk-rows-20261020-v1`.

Small independent 128-byte row transfers plus byte-barrier accounting are not
vLLM's two-dimensional TMA tensor-map pipeline. A common instruction family
does not imply an equivalent implementation. The unsuccessful candidate is
not a serving path or a performance fix.

## Peer dispatch frontier (second independent handoff candidate)

The retained latest rank-0 warm trace has six successive `cuGraphLaunch`
CPU calls at 152625936--158970400 ns. The first NCCL node runs at
153948768--160739072 ns. Thus 5.022 ms of that node's interval occurs while
the same host is still enqueuing the remaining graph windows. These are
profiled CPU intervals, not estimates of uninstrumented driver cost.

The causal source ordering is unambiguous:

```
both inputs ready
  -> local.submit() [enqueue all graph windows]
  -> remote.run() [only prepares the framed command]
  -> later remote.poll() [actually writes the socket]
```

The peer cannot begin its enqueue until after that local submission interval.
This creates avoidable early-arrival waiting; it does not establish that all
6.790 ms of the first NCCL span is attributable to this cause.
The local vLLM executor (`vllm/v1/executor/multiproc_executor.py`,
`collective_rpc`) broadcasts the work before collecting responses. It does
not execute one rank's whole host enqueue before dispatching its peers.

The candidate changes the runtime effect boundary, not mathematical IR:

```
both inputs ready -> prepare peer RUN -> bounded send retirement
                                      |-> local graph enqueue
                                      `-> peer receives RUN -> peer graph enqueue
both executions retired -> remote commit receipt -> local commit -> publication
```

An explicit `Dispatching` state handles a partial socket write without
spinning, extending deadlines, or starting local GPU work too early. The
existing remote channel performs at most one zero-wait write per progress.
Dispatch is not execution acknowledgement and cannot publish tokens.
Cancellation still drains both ranks before KV reuse. The compiler's ordered
device effects and captured graph artifacts remain unchanged. No comparison
or transport implementation enters the pure model/compiler layers.

Four CPU-enriched attempts stopped at the unchanged 17 GiB diagnostic guard,
before the 16 GiB user floor. They are not valid complete handoff captures.
The final mixed OSRT-only/CUDA attempt is
`/tmp/lunaflux-glm-http-handoff-cpu-ingress-20261020-v2`, with its last rank-1
observation 17,797,692 KiB available. The next ablation uses graph-level CUDA
tracing on both ranks; it deliberately cannot provide per-kernel allocation
or CPU syscall attribution.

**Trace setup correction:** those four attempts and the subsequent
`handoff-graph-control-20261020-v1` used the sparse/HC artifact-export
configuration, which omitted `collective_replay` and
`collective_window_sites`. Their AOT kernels were current but their host
submission policy was not the latest 16-site graph path. They must not be
used as performance controls, or as evidence that the latest graph path
cannot fit instrumentation. The graph-only attempt also stopped at the
17 GiB guard. All failed roots are retained.

The corrected control
`/tmp/lunaflux-glm-http-handoff-exact-control-20261020-v1`
copies the actual previously traced
`/tmp/lunaflux-glm-http-sparse-hc-candidate-trace-20261020-v3` configuration.
Both rank manifests explicitly select exact capacity / 16 sites, 64 rows,
eight slots and context8192. The offline trace runner now supports a required
expected replay policy and checks both launch and artifact manifests before
starting any GPU owner. Its regression rejects an omitted or different window
policy. This is bounded offline experiment metadata, not request-path checks
or payload hashing. No resident-slot reduction was made.

The corrected simultaneous graph control also reached the 17 GiB early guard
(last rank-0 observation 17,814,200 KiB; rank1 remained above that guard).
The next control therefore traces rank1 only while both ranks execute.
Its first root, `handoff-exact-egress-control-20261020-v1`, loaded and warmed
successfully but the collector still addressed rank0 because it used a second,
independent rank-selection variable. Collection failed before the measured
request. The collector now inherits the native launch scope unless an explicit
collection subset is supplied; a regression covers egress/both selections.
The corrected v2 root retains the same eight-slot manifests and kernels.
These failed harness attempts are not numerical failures or performance data.

## Completed runtime ablations

All three completed variants use the latest sparse/HC AOT artifacts, exact
64-row replay, 16 collective sites/window, eight resident slots and context 8192.
Both GPUs execute; **only rank 1 is instrumented**, at graph granularity.
The literal cold and warm requests each complete with 7973 prompt tokens,
zero cached tokens, one output token and `[DONE]`. Each variant then answers
`323`, finish reason `stop`, to `17 * 19`. This exercises the exact 64-row and
partial 37-row paths and request retirement; it is not broad answer-quality
qualification or a throughput benchmark.

| Variant | Executable | Evidence root |
| --- | --- | --- |
| Unchanged latest control | retained sparse/HC candidate | `/tmp/lunaflux-glm-http-handoff-exact-egress-control-20261020-v2` |
| Pipe readiness only | `d8c900ca` | `/tmp/lunaflux-glm-http-handoff-readiness-20261020-v1` |
| Readiness + early peer dispatch | `b573c1f4` | `/tmp/lunaflux-glm-http-handoff-peer-dispatch-20261020-v1` |

The new external `analyze-glm-graph-cadence.mbtx` reads the actual graph spans,
verifies the six-window instance cycle, matches CPU graph-launch correlations
and excludes the incomplete tail from complete-frame comparisons. It does not
infer kernel times from graph durations or assume synchronized host clocks.
Reports are `cadence-{cold,warm}.json` beside each SQLite.

Observed handoff, end of a complete effect frame to next frame start:

| Request / chunk ordinal | Control ms | Readiness ms | Both fixes ms |
| --- | ---: | ---: | ---: |
| Cold / 0 | 3.675488 | 2.177248 | 1.040608 |
| Cold / 1 | 3.489376 | 1.392224 | 1.210624 |
| Warm / 0 | 3.423008 | 3.739872 | 2.377568 |
| Warm / 1 | 3.973120 | 1.055040 | 1.634944 |

Full start-to-next-start cadence, including that handoff:

| Request / chunk ordinal | Control ms | Readiness ms | Both fixes ms |
| --- | ---: | ---: | ---: |
| Cold / 0 | 278.611200 | 279.275520 | 276.776064 |
| Cold / 1 | 222.666496 | 229.817952 | 220.508160 |
| Warm / 0 | 279.058560 | 280.038528 | 278.022336 |
| Warm / 1 | 217.982176 | 215.773056 | 216.914720 |

The combined candidate improves all four observed cadences by 1.036–2.158 ms,
or 0.37–0.97%. This is a small positive boundary result, **not elimination of
the projection gap**. Readiness alone improves three of four handoffs but
does not consistently improve cadence. Adding early dispatch also does not
uniformly beat readiness in every row: warm chunk 1 worsens 1.142 ms. In cold
chunk 1 the readiness-only GPU frame is 228.426 ms versus 219.177 ms for control
and 219.298 ms for the combined variant. No causal explanation for that
unchanged-kernel variation is established; it must not be counted as 9 ms
saved by socket dispatch.

The combined first effect frame is 275.735 ms cold / 275.645 ms warm. Its unchanged
control is 274.936/275.636 ms: **the GPU effect frame itself did not get faster**.
The demonstrated cadence benefit comes from shortening the boundary. These
graph spans include TP and differ in scope from the earlier kernel-classified
decoder allocation. They are not replacement per-stage numbers.

### Re-evaluation of the quoted earlier graph

- The prior latest node-level sparse/HC trace had already moved the cold
  decoder from 290.09 ms to 277.364 ms, compared with retained vLLM 256.071 ms;
  its warm result was 282.440 ms and was not a consistent whole-chunk win.
- The remaining projection excess in that cold allocation was 15.963 ms.
  None of the three projection transports tried here earns serving selection.
- The old 9.352 ms warm rank-1 handoff is **not a stable application-only cost**.
  With engine source unchanged, the lighter single-rank graph control observes
  3.423 ms at the same first boundary. Rank coverage and trace granularity both
  changed; this does not isolate either one's overhead individually.
- The retained vLLM handoff is 4.011 ms. It remains a historical observation,
  not a fresh matched graph-granularity ablation. Do not subtract it from the
  new measurements and advertise a new cross-engine speedup or gap.
- This capture has no internal NCCL nodes or rank-0 GPU timeline. The source
  dispatch dependency is removed, but a fresh two-rank NCCL wait decomposition
  has **not** been obtained. Earlier arrival/wire-time bounds remain historical.
- HC small-row compiled resource envelopes, KDA non-projection work, dense
  FFNs and shared/routed contention are not declared fixed by this change.

### Gates and deployment boundary

All completed variants peak at 96,539 MiB observed device allocation (94.28 GiB).
Sampled host minima are 18,125,756 / 18,136,332 / 18,334,160 KiB, respectively;
owned process swap peaks are zero. Fast headroom monitoring and the 100 GiB
device / 16 GiB host limits remain active. These are sampled observations,
not a claim of continuous peak visibility.

Local affected gates pass: 137 packed-execution tests, 19 GLM command tests,
19 grouped-matrix tests, 14 tiled-source tests, inherited-pipe deadline/wakeup
fixtures, and the offline runner/analyzer regressions. The final command
check passes with the repository's existing migration-warning exclusions;
none were added. Source-pinned ARM builds pass under 2 GiB/no swap (the
peer-dispatch build reports 1.5 GiB memory peak).

Full `moon check --target native --deny-warn` still stops on 21 pre-existing
migration diagnostics, including `try?` and implicit derived methods.
Full warning-denied `moon test` also fails during checking on existing
migration diagnostics, before a full test campaign can execute. The unrelated
dirty toolchain migration was preserved, not staged or waived.

The prior qualified GLM deployment is restored unchanged, not replaced by the
private candidate. Both user services are active with `Result=success`,
`ExecMainStatus=0`, zero owned-process swap, healthy backend and a real `323`
answer (`finish_reason=stop`). Restoration observations have retired.
Evidence: `/tmp/lunaflux-glm-handoff-restoration-20261020-v1`.
Its sampled minimum host availability is 17.362/19.222 GiB and maximum GPU
allocation is 94.190/94.219 GiB on ranks 0/1.

The new host runtime is committed and physically tested, but this limited
four-interval result is not a broad production-performance qualification.
No losing projection path or unproven numerical policy is promoted.

### Remaining work, not claimed complete

The projection experiment rules out these small-tile copy substitutions for
the measured shapes. A genuine rectangular tensor-transfer pipeline with
larger useful work per publication is distinct from the rejected 64 small
row-copy variant; its descriptor/ownership implementation and qualification
are still missing. HC needs independently compiled small-row/full-row resource
envelopes, not merely a runtime branch in one wide-register kernel. The other
previously measured KDA/dense-FFN costs and shared-branch resource contention
also remain open. Neither faster pipe wakeup nor moving peer dispatch solves
those mathematical-work/kernel-efficiency gaps.
