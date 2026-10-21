# Low-memory two-rank tracing and idle-rank abort

## Scope and budget

Latest b1788d1d accepted projection artifacts, scales+bulk experts, local MoE
combine, BF16 projection TP, 16-site graph windows, 64-token chunks. This is a
selected execution-graph retrace, not an overall benchmark. Two new full-model
capture configurations maximum. Keep GPU usage below 100 GiB per Spark and
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
yet been captured. The historical missing core is not presented as evidence.

The runtime repair suspends an indefinitely idle stage on its socket, retaining
at most one prefix byte in already-owned storage. A fresh-clock poll starts the
absolute transfer deadline after wakeup. GPU progress/prepared transactions stay
polled; there is no fixed sleep added to the request path. The external async
library's general 32-bit identity limitation is not claimed to be repaired.

Local transport tests pass12/12, including idle wakeup, fragmented prefix
deadline, cancellation and EOF. Targeted stage-control execution and retirement
tests pass2/2. The service check passes with the already-existing migration
warning exclusions; an unrestricted warning-denied check still fails on
unrelated migration warnings. No warning policy was changed.

## Collector design and qualification status

The external CUPTI13.0.1 injection fixture has eight fixed1MiB host buffers,
a four-buffer1MiB device activity pool, no stack sampling, and no module-image
snapshots. Actual internal CUPTI overhead still requires measurement. It loads
at CUDA initialization in the real approved-spawn GPU child, not in the HTTP
parent. Four timestamped capture windows are supported; raw events stream to
disk. Conversion and symbol demangling occur after model retirement.

Kernel/graph/stream/correlation IDs, CUDA API intervals, copies and memsets are
retained. Cross-machine clocks are not assumed aligned. Buffer refusals, dropped
records, malformed timestamps and collector failures disqualify a capture.
SQLite output records provenance and leaves uncollected resource fields NULL.
The compiler neither imports the collector nor treats reference comparisons as
correctness evidence.

CPU ASan/UBSan buffer ownership, exhaustion/reuse and serialization tests passed
in a256MiB/no-swap user unit. Both-rank CPU builds pass. GPU graph/approved-child
coverage and full-model memory/retrace results are pending; these CPU checks do
not establish trace completeness or service qualification.
