# Bounded instance telemetry

This package owns one fixed-capacity, thread-confined telemetry accumulator for
one LunaFlux engine instance. Scheduler prefix telemetry uses the same finite
vocabulary, while the service bridge that copies scheduler scalars into this
owner remains a separate serving integration step.

The vocabulary is closed at compile time. Twenty-two counters cover admissions,
completions, cancellations, deadlines, failures, prompt/generated tokens,
worker restarts/failures, network accepts/disconnects/rejections, and
backpressure plus prefix lookups, hits, misses, evictions, reused/computed
tokens, and publications. Six gauges add live prefix entries/pages to queue,
active-request, and KV-page state. Six histograms retain request, first-token,
inter-token, cold-start, batch-row, and batch-token distributions.
Label cardinality is exactly zero: record APIs accept no strings, label maps,
request/model identity, paths, vendor text, or raw payloads. Adding a semantic
dimension therefore requires a reviewed vocabulary/API change rather than
runtime cardinality growth.

All histograms have exactly sixteen non-cumulative storage buckets. Latency
bounds are `1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000,
30000, 60000, Int::max`. Batch-row bounds are powers of two through 16384 plus
`Int::max`; batch-token bounds are powers of two through 8192, then 32768 and
`Int::max`. Values equal to a bound enter that bucket. Negative observations,
gauge values, counter increments, and invalid bucket indices fail with a typed
payload-free error before mutation.

Construction allocates exactly 22 counter cells, six gauge cells, and 96
histogram cells for live state, plus the same fixed 122-cell snapshot slot.
Counter and histogram cells saturate at `UInt64::max`; they never wrap. Gauges
are nonnegative `Int` scalars and accept `Int::max`. Record, scalar read, reset,
and snapshot operations allocate no managed storage after construction.

A snapshot copies live scalars into the owner's preallocated slot and returns
an opaque owner/generation capability. Live recording and reset cannot mutate
that snapshot slot. Reusing the slot increments its generation and makes every
older snapshot stale before it can observe substituted values. Generation
exhaustion is typed and checked before copying, so it cannot wrap or partially
publish. Snapshot APIs expose scalar values only—never arrays, storage,
generation numbers, or mutation authority. The owner and snapshots require
serialized access; this package does not claim cross-thread synchronization.

`scripts/validate-instance-metrics-allocations.sh` pins the exact public
vocabulary and opaque interfaces, bans string/label/raw-storage escape, and
checks release-generated C for allocations or bulk-copy helpers in record,
read, reset, snapshot, and snapshot-read paths. Constructor allocations and a
separate 19-byte white-box positive control keep the evidence honest.
