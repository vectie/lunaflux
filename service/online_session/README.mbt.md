# Luna online instance

This package owns one persistent, alias-free `LunaOnlineInstance` for bounded,
one-active-request-at-a-time online inference. Instance preparation binds an
expected tokenizer digest, model identity, and exact inference envelope, then
starts and authenticates one scheduler, online worker lease, and rooted worker
process. It preallocates semantic-event, decoded-output, and failure-code
storage. It owns no tokenizer, request receipt, listener, or transport adapter.

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
request. An outer owner then stages a `LunaFramedEventAdapter`, copies and
releases its frame credit, and only then acknowledges the Luna event credit.
The adapter cannot ACK semantic state. SSE and OpenAI-compatible adapters can
consume the same typed boundary without parsing canonical framed bytes.

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
