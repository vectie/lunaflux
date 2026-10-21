# Low-memory two-rank tracing and idle-rank abort

Follow-up: the filtered collector has now completed a lossless native two-rank
cold/warm full-model capture. See [the October21 filtered retrace](GLM_FILTERED_TWO_RANK_TRACE_2026-10-21.md)
for complete-chunk allocation, overlap and paired TP bounds. Its fresh vLLM
attempt was stopped by loading headroom protection; the reference comparison
remains explicitly historical. The failures and budget below describe the
earlier campaign and are retained, not overwritten.

## Scope and budget

Latest b1788d1d accepted projection artifacts, scales+bulk experts, local MoE
combine, BF16 projection TP, 16-site graph windows, 64-token chunks. This is a
selected execution-graph retrace, not an overall benchmark. The initial budget
was two full-model collector configurations. Both failed before a capture window;
one final window-only configuration was then explicitly announced. Keep GPU usage below 100 GiB per Spark and
host availability above 16 GiB; use earlier stops during loading. No payload
hashing. Restore the accepted GLM service after owned diagnostic workers retire.

## Idle abort: evidence before implementation

The historical rank-1 failures were SIGABRT after approximately 2h22m, not
demonstrated memory-limit kills. The live b1788d1d rank (PID35148) spends almost
one CPU in an idle epoll/zero-timeout polling loop. A read-only, symbol-resolved
probe at 09:53 CST observed coroutine ID2363640517 (unsigned) and timer
ID1181820263. No live process counters were modified.

`RemoteChannel::poll_receive` creates a zero-timeout async task group per empty
poll; GLM repeatedly calls it and `pause`. The async0.22.4 coroutine identity is
a 32-bit incrementing integer used for equality in the live-coroutine set.
Two task IDs are created per idle polling iteration. A complete wrap can reuse
the identity of the still-live main task, invalidating the scheduler's live-set
accounting. The observed rate is consistent with the historical failure period.

An isolated CPU-only ARM reproducer uses the same pinned toolchain/async source
and idle socket pattern. Its debugger changes ONLY that disposable process's
counter immediately before polling. Ordinary counter100 and crossing the signed
midpoint from2147483640 complete normally. Starting at-8 causes SIGABRT at
counter4 in `Coroutine::unwrap` called by `with_event_loop.inner`. Evidence:
`/tmp/lunaflux-idle-repro-20261021-v1/{ordinary,signed,collision}.txt`.
This demonstrates the collision mechanism; a live natural-abort stack has not
been captured. The live debugger was deliberately interrupted and detached
before stopping the old pair; its saved SIGINT snapshot is not an abort stack.
The historical missing core is not presented as evidence.

The runtime repair suspends an indefinitely idle stage on its socket, retaining
at most one prefix byte in already-owned storage. A fresh-clock poll starts the
absolute transfer deadline after wakeup. GPU progress/prepared transactions stay
polled; there is no fixed sleep added to the request path. The external async
library's general 32-bit identity limitation is not claimed to be repaired.

Local transport tests pass12/12, including idle wakeup, fragmented prefix
deadline, cancellation and EOF. Targeted stage-control execution and retirement
tests pass2/2. The service check passes with the already-existing migration
warning exclusions; an unrestricted warning-denied check still fails on
unrelated migration warnings. No warning policy was changed. Implementation
`5540ad40` and deployed-toolchain spelling correction `ed6be466` are pushed.
The clean, commit-pinned ARM build in
`/tmp/lunaflux-glm-service-arm-idlewait-20261021-v2` completed within its 2 GiB,
no-swap CPU-only unit. The earlier build failure from local formatter `Int32`
spelling is retained; it was not a CUDA or runtime failure.

## Collector design and qualification status

The external CUPTI13.0.1 injection fixture has eight fixed1MiB host buffers,
a four-buffer1MiB device activity pool, no stack sampling, and no module-image
snapshots. Actual internal CUPTI overhead still requires measurement. It loads
at CUDA initialization in the real approved-spawn GPU child, not in the HTTP
parent. The final collector registers buffers/control at initialization but
enables activity collection only at the first post-readiness `start`. Four
timestamped capture windows are supported; completed buffers flush every 10 ms
and raw events stream to disk. Conversion and symbol demangling occur after
model retirement. `seal`, issued only after HTTP work drains, forces the final
activity flush without synchronizing CUDA, then writes completeness evidence.
This does not rely on Python workers running `atexit` rather than `_exit`.

Kernel/graph/stream/correlation IDs, CUDA API intervals, copies and memsets are
retained. Cross-machine clocks are not assumed aligned. Buffer refusals, dropped
records, malformed timestamps and collector failures disqualify a capture.
SQLite output records provenance and leaves uncollected resource fields NULL.
The compiler neither imports the collector nor treats reference comparisons as
correctness evidence.

CPU ASan/UBSan buffer ownership, exhaustion/reuse and serialization tests passed
in a256MiB/no-swap user unit. Both-rank CPU builds pass. The final library is
`/tmp/lunaflux-activity-20261021-v7`; its actual approved-child GPU fixture
`/tmp/lunaflux-activity-fixture-20261021-v8` creates its graph before tracing is
enabled, executes one million pre-window API calls, captures two windows, seals,
and deliberately calls `_exit(0)`. All16 kernels/eight graph nodes are present,
with50 total records and zero dropped/refused/invalid/error records. This proves
fixture coverage, not full-model numerical correctness or trace completeness.

## Failed captures retained, not used as timing evidence

- Native full-model roots `.../lunaflux-glm-http-activity-20261021-v1` and `v2`
  loaded and passed warmup, but the initial and periodic-flush collectors refused
  96,975 and105,375 rank-1 buffer requests during startup, respectively. Capture
  control rejected them before the long request. Neither is a valid decoder
  trace. A tiny API-burst fixture did not reproduce this full startup behavior.
- Window-only root `.../lunaflux-glm-http-activity-20261021-v3` failed before GPU
  model loading: the harness launched before asynchronous artifact preparation
  finished, causing `ApprovedFsError.Unavailable`. This is an orchestration
  failure, not a projection kernel or memory failure. The final attempt waits
  for both remote manifests/executables and preparation completion.
- The former non-streaming acceptance payload was inappropriate for the
  cold/warm trace harness's SSE completion assertion. No earlier failed attempt
  reached that request. The final trace retains its literal messages, requests
  one output token and SSE, and requires7,973 uncached prompt tokens in both
  responses. It is not an overall throughput or latency benchmark.

Offline conversion preserves integer nanosecond timestamps, including values
above2^53; it never routes them through a floating-point JSON number. Updated
execution/TP/handoff analyzers preserve that precision. The trace owner selects
the actual GPU worker by joining its owned control endpoints to GPU process IDs,
not a guessed Python PID or arbitrary memory threshold. Missing or ambiguous
ownership fails the capture. Comparison remains outside the compiler.

## Actual full-window failure and bounded correction

The final window-only attempt, `/tmp/lunaflux-glm-http-activity-20261021-v4`,
starts both collectors with zero drops/errors. During its first cold window,
rank1 emits1,719,959 retained records and ultimately reports859,722 dropped and
refused records. Stop-time counts were1,708,459 retained and701,397 dropped;
the larger terminal counts include later buffer draining. Both ranks retire
cleanly; the incomplete cold request is not a correctness result.

The retained records identify the pressure directly:

| Rank1 record | Count |
| --- | ---: |
| `cuCtxSetCurrent` | 857,563 |
| `cuEventQuery` | 857,257 |
| All other driver records | 170 |
| GPU kernels | 4,902 |
| GPU copies | 67 |

These are counts in an incomplete capture, **not an unbiased execution-frequency
estimate or valid timing allocation**. Context binding and event polling account
for over99% of retained records. The relevant native polling owner is
`internal/cuda/ordered_executor.c`; this active-GPU polling is distinct from the
idle socket-coroutine churn repaired above. The tracer must not confuse the two.

Library `/tmp/lunaflux-activity-20261021-v8` now disables context-bind/event-query/
stream-query activity through CUPTI's per-API filter, **before record allocation**.
Filtering text after buffering would not resolve this pressure. GPU events,
launches, copies, explicit waits and synchronization remain enabled. The schema
declares the omitted API scope, so individual polling CPU time cannot be inferred
from this filtered trace. No production polling policy or compiler pass changed.

Fixture `/tmp/lunaflux-activity-fixture-20261021-v9` adds one million context
bind/query pairs **inside each of two windows**, four million calls total, while
preserving the graph-before-enable and `_exit` conditions. All16 expected kernels
and eight graph nodes are present, with zero dropped/refused/invalid/error
records. Latest collector CPU ASan/UBSan ownership/schema checks also pass in
the256MiB/no-swap unit. This is a qualified fixture, not a completed full-model
retrace. The full-model capture budget is exhausted; no new vLLM timing or
two-rank overlap/TP comparison is claimed. The retained October19 vLLM trace
remains historical, and is not relabeled as a fresh measurement.

The failed v4 attempt stayed within its sampled memory envelope:
rank0/1 GPU maxima94.241/94.270 GiB; host minima17.276/19.079 GiB; owned swap0.
Source: `memory-envelope.json` plus raw fast/slow watcher logs in that root.
These are sampled bounds, not a claim of continuous physical peak measurement.

## Unprofiled runtime qualification and release boundary

`/tmp/lunaflux-glm-http-idlewait-validation-20261021-v2` uses the commit-pinned
`ed6be466` runtime and unchanged accepted projection AOT/geometry, without any
tracer. The7,973-token SSE request completes, followed by a ten-second idle
interval and the complete arithmetic answer `323` with `finish_reason=stop`.
Both owned workers retire successfully. This validates request completion and
idle wakeup, not model quality across a new corpus or a multi-hour abort soak.
Sampled rank0/1 host minima17.159/18.980 GiB, GPU maxima94.239/94.268 GiB,
owned swap0. The earlier validation-v1 harness failed before launch because it
used obsolete default SSH socket paths; the v2 invocation supplies the live
sockets explicitly. That failure is retained and is not a runtime defect.

Targeted `moon info` succeeds. The final full warning-denied native check and
test commands were attempted and remain blocked by pre-existing migration
warnings promoted to errors (including `implicit_impl_as_method`); no full-suite
pass is claimed. Transport tests pass12/12 with the same pre-existing warning
exclusions. The offline owner/converter/execution/TP/handoff regression tests
pass26/26. No warning policy or unrelated migration file was changed.
Raw phase results: `/tmp/lunaflux-low-memory-phase-checks-20261021-v1`.

The new runtime is installed into separate `glm53-lunaflux-ed6be466` release
directories. Existing accepted cubins/configuration remain unchanged; the
previous `b1788d1d` releases and unit backups remain available. No observer is
linked or injected into the public service. Installation reads existing declared
labels only; there is no payload hashing. Deployment evidence:
`/tmp/lunanexa-glm-idlewait-install-20261021-v1`.

### Restored service

Both public USER units are active on the `ed6be466` release. The actual rank1
GPU PID1011947 resolves to that release's executable. The private backend is
healthy and returns exactly `323`, `finish_reason=stop`, after restoration.
LunaNexa's opaque runtime revision metadata now identifies `ed6be466`; its
controller rollout completed. The existing customer API key was not retested.

Restoration watcher evidence at
`/tmp/lunaflux-glm-idlewait-restoration-20261021-v1` records rank0/1 sampled host
minima17.171/18.993 GiB, GPU maxima94.239/94.268 GiB, and zero owned swap. The
watcher is retired; it is not a continuing multi-hour monitor. The service remains
running without diagnostic injection.

In a31.285-second post-answer observation, rank1 accumulated four CPU ticks at
100 ticks/second:0.128% of one core, versus the old approximately97% idle polling
observation. This supports elimination of the busy-idle trigger. It is not a
multi-hour soak or proof that the external async library has no other ID-wrap
path. Raw `/proc` samples and the calculation are in `idle-cpu-stat.txt` and
`idle-cpu.json` in the restoration root.

Remaining work is explicit: run a bounded full-model capture with the now
fixture-qualified, pre-allocation-filtered v8 collector; require clean final
records and a complete45-layer chunk on both ranks; then use the identical
observer scope for vLLM within the memory reserve. No latest two-rank timing
table, speedup, TP-wait attribution or fresh vLLM result exists for this turn.
The accepted projection numerical/sanitizer evidence remains in
`GLM_FRAGMENT_PIPELINE_2026-10-21.md`; no kernel numerical policy changed here.
