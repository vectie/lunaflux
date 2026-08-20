# Luna online instance

This package owns one persistent, alias-free `LunaOnlineInstance` for the
bounded one-active-request-at-a-time online mode. Instance preparation binds
tokenizer and model identities, inference and wire envelopes, reusable
event/output storage, one scheduler, and one rooted worker process. It does not
accept a request.
`prepare_owned_luna_online_instance` starts and authenticates the worker once,
then transfers its opaque online lease into the instance.

Tokenization, deadline validation, and incremental-output construction happen
off-reactor in `service/request_admission`, which publishes one opaque
`LunaPreparedRequest`. `LunaOnlineInstance::begin` decides Busy, Draining, or
request-epoch exhaustion before consuming that owner, validates its streaming,
tokenizer, model, and exact inference-envelope evidence, prepares Accepted
credit, and only then claims its scheduler request exactly once and commits
lower admission. A lower admission rejection consumes the prepared request,
retires the writer credit, and leaves the instance idle. After admission
commits, only nonraising assignments publish a nonzero
`LunaOnlineRequestTicket`. Every request operation authenticates that ticket
before touching writer, decoder, scheduler, or worker state.

Only one request is active at a time. The first outbound credit is the
canonical event-v2 Accepted frame. Normal progress publishes exact Token
credits, followed by Usage and Completed-v2 for natural maximum-output or
physical stop-token termination. Stop tokens count toward usage but are
suppressed from Token output; their exact adjacent terminal is authenticated
before Usage. Any final valid UTF-8 decoder tail is carried only by Completed.
Every frame remains pinned in the same reusable one-credit storage until copied
and acknowledged.

Ordinary generated-token decode and writer publication use scalar
transactional status after exact scheduler reservation/dequeue. String-stop
matching remains incremental and transactional. Caller cancellation is
deferred behind pinned Accepted or Token credit and commits one exact cut after
acknowledgement. Every credit-free progress call enforces the owner-bound
deadline before worker/publication mutation. Deadline, output, and worker
failures publish the same payload-safe fixed codes established during instance
preparation.

A healthy final Completed or Failed acknowledgement moves the request to
ReleaseReady. `progress_request_retirement(ticket)` invokes the lower
`retire_terminal_request` transaction, clears only request-local decoder,
event, counter, and cut state, and returns the instance to Idle. The retained
lease epoch, scheduler publication cursor, worker plan predecessor, rooted
process, fixed event storage, and failure-code storage are not reset. The next
request receives a fresh Luna ticket and begins token positions at
zero without spawning or handshaking another worker.

Worker, protocol, or device failure remains close-only. Normal `progress`
latches the first product-safe failure cause and reports
`LunaOnlineInstanceTerminalizationRequired`; explicit off-reactor
`progress_terminalization(ticket)` recovers and drains through the exact
terminal. Its final acknowledgement leads to terminal close, never back to
Idle. Retry paths retain the same ticket and cleanup authority.

`begin_drain` prevents new request admission without invalidating an active
ticket. Once the active request retires, `progress_shutdown` closes the empty
lease and rooted worker. This explicit instance drain is the only healthy
worker-shutdown path. Child shutdown and reap may block, so terminal recovery,
request retirement, and instance shutdown do not belong on an async network
reactor.
