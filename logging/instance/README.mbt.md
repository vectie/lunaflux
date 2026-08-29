# Bounded instance event log

`logging/instance` owns one fixed-capacity structured event ring for one
LunaFlux instance. It is an observability foundation, not a global logger,
registry, formatter, sink, exporter, tracing system, audit log, or service
integration. The reusable service will own and feed this object in a later
slice.

The public vocabulary is compile-time finite. Events cover instance lifecycle,
admission and request terminal outcomes, worker restart/failure, network
accept/disconnect/rejection, backpressure, drain, and close. Reasons are a
second finite enum. There is no arbitrary metric name, label, message, request
identifier, model or filesystem path, vendor text, pointer, `String`, or
`Bytes` field.

Every record contains only:

- a finite event and reason;
- a caller-supplied monotonic `UInt64` millisecond timestamp;
- a nonnegative `count` no greater than 1,000,000; and
- a nonnegative `duration_millis` no greater than 86,400,000.

Timestamp validation precedes count then duration validation, and every
rejection is transactional. A lower timestamp is accepted only after `reset`,
which starts a new monotonic sequence. These two scalar fields are semantically
counts and durations; they must not be repurposed as request IDs, handles,
addresses, hashes, or application payload.

Construction accepts a capacity from 1 through 65,536 and preallocates the
current ring plus one equally sized snapshot slot as structure-of-arrays
storage. A full ring overwrites its oldest logical entry and increments
`dropped_count`, saturating at `UInt64` maximum rather than wrapping. Logical
indexed reads retain oldest-to-newest order across wraparound. `reset` is
constant work and does not mutate an already-copied snapshot.

`snapshot` copies the current logical ring into the startup-owned snapshot
slot without allocation. It returns one opaque generation capability. A later
snapshot invalidates every prior capability; every snapshot getter checks this
generation before its index. Snapshot generation exhaustion is typed and
transactional. The owner and snapshot expose only scalar reads and have no
`Debug` implementation or raw storage projection.

Record, direct read, reset, and snapshot operations allocate no warmed managed
storage. Snapshot work is bounded by the configured capacity and is intended
for an exclusive owner/off-reactor observability handoff; the package provides
no synchronization and makes no cross-thread safety claim.

Focused evidence is run with:

```sh
logging/instance/validate-luna-instance-log.sh
```
