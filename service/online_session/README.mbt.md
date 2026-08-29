# Luna online instance

This package owns one persistent, alias-free `LunaOnlineInstance` for a
startup-bounded set of live online requests and a reusable transport-neutral
`LunaOnlineFramedService` with exact-generation `LunaOnlineFramedStream`s.
`LunaOnlineFramedCoordinator` remains the max-one compatibility facade over
that same service/stream implementation.
Instance preparation binds an expected tokenizer digest, model identity, and
exact inference envelope, then starts and authenticates one scheduler, online
worker lease, and rooted worker process. The instance itself owns no tokenizer,
request receipt, listener, or transport adapter.

Tokenization, deadline validation, and incremental-output construction happen
outside this owner in `service/request_admission`, which publishes an opaque
`LunaPreparedRequest`. Its live fixed-lane path is cooperatively budgeted; only
the compatibility facade remains synchronous/off-reactor. `begin` returns an
opaque allocation-free admission result whose `kind()` is Admitted, Busy, or
Draining and whose `ticket()` is valid only for Admitted. Busy, Draining, and
exhausted request-epoch outcomes precede destructive claim transfer. Streaming
mode, tokenizer digest, model, and the exact inference envelope are
authenticated before lower mutation.

Before Accepted publication, `begin` preflights semantic-event epoch headroom
for `max_new_tokens + 3`: Accepted, at most the maximum Token count, Usage, and
one Completed or Failed event. Accepted is published before the prepared shell
is destructively claimed and before lower admission. If claim or lower
admission then fails, Accepted is discarded. A lower rejection first releases
the exact prepared claim after the lower lease proves it retained no scheduler
authority. The scheduler request passed to that lower lease is a borrow from
the claim and is never retained by this package beyond the claim lifecycle. A
committed request is authorized only by its opaque
`LunaOnlineRequestTicket`.

The persistent `LunaEventOwner` publishes typed Accepted, Token, Usage,
Completed, and Failed state without encoding a protocol frame. `take_event` is
the sole acquisition path and issues one opaque `LunaOnlineEventCredit` for the
exact request and event epochs. `credit.view()` grants semantic read authority;
`credit.ack()` alone advances or retires the event. Usage ACK atomically
publishes a distinct Completed or Failed epoch, so a delayed Usage-credit alias
cannot read or acknowledge the terminal event. Abort discards the pinned event
and invalidates outstanding credit.

Transport adapters receive only `credit.view()`. Adapter capacity and other
fallible transport setup must complete before `begin` can consume a prepared
request. Compatibility callers may still stage `LunaFramedEventAdapter`; the
coordinator instead owns cooperative framed Work/View, releases confirmed frame
authority, and only then acknowledges the Luna event credit. Neither path gives
the framed encoder semantic ACK authority. SSE and OpenAI-compatible adapters
can consume the same typed boundary without parsing canonical framed bytes.

Generated-token decode and semantic publication use scalar transactional
status after exact scheduler reservation/dequeue. Physical stop tokens count
toward usage but are suppressed from Token output. Incremental string-stop
matching withholds matched/post-stop bytes. Natural completion, cancellation,
deadline, output rejection, and worker loss all publish Usage before a distinct
terminal event. Valid unmatched UTF-8 decoder tail appears only on Completed.

Multi-request publication routing is a startup-fixed direct table from the
opaque lower worker route lane to the owning session lane plus exact route
generation. One oldest-publication peek performs one table probe and then
authenticates the mapped live request before dequeue; lane scans and stale-route
substitution are excluded. The exact mapping survives worker recovery until
its retained failure terminal is consumed, then clears before either lane can
be reused.

A healthy Completed or Failed ACK moves that request to ReleaseReady.
`progress_request_retirement(ticket)` retires exact lower request authority,
then releases the exact prepared claim, clears its fixed lane, and leaves other
live lanes untouched. Failed retirement retains the claim for retry. The
worker, scheduler history, plan predecessor, lease epoch, semantic event epoch,
and fixed storage persist across later requests. Worker, protocol, or device
failure uses cooperative child cleanup, exact physical and pending scheduler
retirement, device invalidation, bounded restart delay, replacement, and
failed-replacement cleanup. Readiness remains false during the delay and
`maintenance_wait_remaining_millis` exposes its cached host wake without
sleeping or advancing replacement. Drain during that interval abandons spawn
and reuses deterministic instance-loss terminalization; cleanup and retained
request authority are never delayed.
The claim remains retained until exact lower retirement succeeds. Explicit
instance drain is the only healthy worker-shutdown path and uses the same
cooperative owner; no instance progress method runs a blocking recovery, reap,
or close loop.

## Reusable framed service

`LunaOnlineFramedService` owns the sole preparation pool, online instance,
framed-output workspace, monotonic clock, FIFO storage, and one preallocated
stream-authority slot. `readiness()`, `open_stream()`, and
`open_semantic_stream()` return scalar dispositions. `take_open_stream()`
transfers an opaque stream exactly once. Native mode exposes event-v2 copy
Offers; semantic mode exposes an exact-generation
`LunaOnlineFramedSemanticEvent` and no framed Offer. Neither mode exposes pool
Work, Prepared, ticket, lower event credit, framed View, storage, or raw epochs.

Semantic progress reports `LunaOnlineFramedCoordinatorSemanticEventReady`.
Taking the semantic event returns a read-only authenticated `LunaEventView`.
The service retains the sole lower ACK and output-stall authority. `delivered()`
invalidates the semantic delivery capability exactly once and only arms ACK;
the caller must invoke it only after its complete protocol response has been
confirmed. A retained View becomes stale when later progress ACKs, disconnects,
or advances the stream epoch. Usage ACK may copy at most the configured
`max_decoded_delta_bytes` while replacing Usage with Completed or Failed; this
is a fixed bounded terminal transition, not a strict one-work-unit operation.

Disconnect retires only that stream. It revokes output and rejection credit,
discards partial and queued preparation authority, cooperatively aborts and
retires an active ticket, and returns the same service to Ready. It does not
drain the preparation pool or close a healthy worker. Stream generations never
wrap; retained Stream, Offer, and rejection aliases authenticate both stream
and result epochs and reject after the next open. Request and worker plan
sequence history remains monotonic across sequential streams and pipelined
requests within one stream.

Each stream also owns one preallocated, finite observation pulse. Exact lower
event evidence publishes Admission, Token, Usage, Completion, Cancellation,
Deadline, RequestFailure, or WorkerFailure. WorkerFailure is the early durable
worker/request-failure accounting pulse. A successful lower replacement
publishes WorkerRestart before the recovered request's FailureUsage and a
distinct payload-free WorkerRequestTerminal event. That terminal carries only
the request-latency disposition and must not recount the earlier failure.
The restarted healthy worker lease remains owned while the exact failed
request terminal is consumed and its claim is released; disconnect during
either recovered terminal event retires the request without closing that
healthy lease. `progress` reports `ObservationReady` and performs no later
service transition until the exact observation is taken and ACKed. Usage alone
exposes authenticated input, cached-input, output, and total token counts; no
observation exposes a request ID, payload, timestamp, raw epoch, or mutable
telemetry owner. A startup-preallocated generation-keyed timing slot begins
only after transactional lower admission commits. Token pulses carry either a
relative first-token or inter-token interval, and per-request terminal pulses
carry one relative request interval. Clock failure, rollback, or interval
overflow degrades only that request's timing and emits no fabricated sample.
Disconnect preserves a pending pulse as non-protocol
bookkeeping and stream retirement cannot finish before its ACK. The max-one
compatibility coordinator explicitly consumes these pulses to preserve its
historical progress surface; the reusable TCP server records them first.

Only `LunaOnlineFramedService::begin_drain` cuts admission. It first retires an
active stream, then drains the pool and cooperatively closes the empty online
instance. `LunaOnlineFramedService::progress` drives ordinary work and
cooperative maintenance on repeated calls, with no separate off-reactor
executor prerequisite. The legacy coordinator keeps its historical
`MaintenanceRequired` plus `progress_off_reactor_maintenance` handshake for
existing endpoint callers, but both methods operate on the same service owner.

Ready service preparation exposes authenticated immutable model identity,
inference limits, framed limits, and maximum transport wait before ownership is
consumed, allowing an outer server to reject substitutable configuration.
It also exposes the same initial scalar Luna telemetry projections before
consumption. A live Service projection adds its framed preparation FIFO and
retained Prepared authority to scheduler waiting depth while leaving active
request and KV/last-plan scalars exact. No scheduler, worker lease, request,
plan, page, or raw epoch crosses this boundary.
For an outer protocol that observed receipt before canonical frame encoding,
`Stream::offer_luna_framed_with_receipt` transfers the opaque trusted receipt
only with the first admitted byte range; later chunks use the ordinary offer.
No receipt timestamp or absolute deadline is exposed or rebased here.

## Transport-neutral framed coordinator

`LunaOnlineFramedCoordinator` is the max-one compatibility view over one
fixed-lane `LunaRequestPreparationPool`, one persistent `LunaOnlineInstance`,
and one cooperative `LunaFramedEventWorkspace`. Its preparation factory creates
all three internally, so callers cannot retain a pool, preparation Work,
Prepared request, online ticket, semantic event credit, or framed Work/View
alias. It owns no listener, socket, TLS state, async runtime, or TCP retry
policy.

Before transferring the ready coordinator, its preparation exposes one
authenticated scalar `maximum_transport_wait_millis`. A transport must bind
its blocking read/write interval at or below that owned inference deadline;
this prevents a separately supplied timeout from postponing receipt, request,
or output-stall polling.
Immediately before each body read or event write, the live coordinator also
reports the shorter current remaining interval across its incomplete receipt
and output-stall obligations. The scalar is recomputed from authenticated
owners and exposes no absolute deadline.

Ingress accepts only caller-owned canonical framed byte ranges. A short
accepted count leaves the unconsumed tail with the caller. Once the exact frame
body is received, its Work remains in a fixed preallocated FIFO while the
receipt position can accept the next frame into another free lane. Later-ready
preparations never bypass the FIFO head. One preallocated Prepared slot retains
Busy admission without consulting the now-stale preparation Work. Saturation,
draining, a pinned rejection, and a live receipt that cannot currently consume
bytes return `Backpressured`/`Draining` with zero consumption; `Accepted` always
means at least one byte was consumed.
While an active request owns the ordered stream, a later Ready or Failed FIFO
head remains pinned and reports progress rather than transport idleness; it is
inspected only after the active request retires.
Coordinator progress reports `LunaOnlineFramedCoordinatorAwaitingInput` only
when the selected authenticated direct receipt remains byte-starved after the
preparation pool has applied deadline, cancellation, and drain precedence.
Scanner work and object-form preparation never use that result.

Every admitted frame receives a nonwrapping one-based sequence. A preparation
failure publishes one opaque, payload-safe `LunaOnlineFramedRejectionCredit`;
later FIFO progress remains pinned until its exact ACK. Rejections expose only
sequence plus a bounded rule. They share the output-stall deadline, so an
untaken or unacknowledged rejection cannot pin drain forever.
This typed pre-admission disposition is not a semantic Luna event and is not
encoded as a canonical event-v2 frame.

The coordinator takes semantic event credit privately and cooperatively builds
event-v2 bytes. `copy_framed_event_chunk_to` returns an exact-generation
`LunaOnlineFramedEventOffer`; only that Offer can confirm a positive prefix of
the copied bytes. Partial confirmation consumes the Offer, advances the
confirmed cursor, and requires the transport to copy the remaining bytes again.
An old Offer cannot confirm an equal-sized range from a later event. Final
confirmation releases framed View authority before semantic ACK. Usage ACK can
copy the bounded terminal tail while replacing Usage with Completed or Failed,
so that step runs only in `progress_off_reactor_maintenance`; other ACKs remain
scalar transitions.

One system monotonic clock is shared by receipt, online admission, and output
stall enforcement. Taking a semantic credit or publishing a rejection derives
a checked hard deadline using `inference.max_deadline_millis`. Reactor
`progress`, chunk copy, Offer confirmation, and rejection access must be polled;
expiry or clock failure revokes the pinned result, cuts the active request when
present, and enters drain/maintenance. This is cooperative enforcement, not a
background timer.

Compatibility `progress` advances at most one preparation, online, or framed-output quantum.
It reports `MaintenanceRequired` without running worker recovery, terminal
retirement, process close, or clean shutdown. Those transitions are owned only
by `progress_off_reactor_maintenance`. Scheduler admission is constant with
respect to prompt length and stop count because token maxima are cached; it
remains bounded by the configured scheduler-slot scan.
The coordinator itself owns no TCP listener, connection, socket writer, or
network retry adapter. `service/online_tcp` composes the service into one-shot,
reusable max-one, fixed-lane pipeline, and serialized OpenAI HTTP owners.
Concurrent-client arbitration, TLS, and a listener fleet remain outside this
slice.
