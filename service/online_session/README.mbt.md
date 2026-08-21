# Luna online instance

This package owns one persistent, alias-free `LunaOnlineInstance` for bounded,
one-active-request-at-a-time online inference and the optional transport-neutral
`LunaOnlineFramedCoordinator` that composes it with framed preparation/output.
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

A healthy Completed or Failed ACK moves the request to ReleaseReady.
`progress_request_retirement(ticket)` retires exact lower request authority,
then releases the exact prepared claim, clears request-local state, and returns
the instance to Idle. Failed retirement retains the claim for retry. The
worker, scheduler history, plan predecessor, lease epoch, semantic event epoch,
and fixed storage persist across the next request. Worker, protocol, or device
failure remains close-only; its claim is retained across failed close attempts
and released only after terminal close succeeds. Explicit instance drain is
the only healthy worker-shutdown path; blocking recovery, retirement, and
shutdown stay off the network reactor.

## Transport-neutral framed coordinator

`LunaOnlineFramedCoordinator` is the single-stream owner that composes one
fixed-lane `LunaRequestPreparationPool`, one persistent `LunaOnlineInstance`,
and one cooperative `LunaFramedEventWorkspace`. Its preparation factory creates
all three internally, so callers cannot retain a pool, preparation Work,
Prepared request, online ticket, semantic event credit, or framed Work/View
alias. It owns no listener, socket, TLS state, async runtime, or TCP retry
policy.

Ingress accepts only caller-owned canonical framed byte ranges. A short
accepted count leaves the unconsumed tail with the caller. Once the exact frame
body is received, its Work remains in a fixed preallocated FIFO while the
receipt position can accept the next frame into another free lane. Later-ready
preparations never bypass the FIFO head. One preallocated Prepared slot retains
Busy admission without consulting the now-stale preparation Work. Saturation,
draining, a pinned rejection, and a live receipt that cannot currently consume
bytes return `Backpressured`/`Draining` with zero consumption; `Accepted` always
means at least one byte was consumed.

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

`progress` advances at most one preparation, online, or framed-output quantum.
It reports `MaintenanceRequired` without running worker recovery, terminal
retirement, process close, or clean shutdown. Those transitions are owned only
by `progress_off_reactor_maintenance`. Scheduler admission is constant with
respect to prompt length and stop count because token maxima are cached; it
remains bounded by the configured scheduler-slot scan.
No TCP listener, connection owner, socket writer, or network retry adapter is
implemented by this slice.
