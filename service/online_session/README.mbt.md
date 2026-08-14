# Owned online session

This package owns one alias-free, single-request Streaming session. Its factory
consumes `ReceivedRequest`, performs tokenization/admission internally, creates
the scheduler and rooted worker through `worker_service.prepare_owned`, and
immediately transfers that preparation into its preallocated online lease.
Neither `AdmittedRequest`, `WorkerService`, `OnlineWorkerLease`, scheduler
handles, decoder owners, nor raw publications appear in this package's public
interface.

`prepare_owned_session` is synchronous and off-reactor: it may tokenize, spawn
and handshake with the worker, and acquire/release rooted authority. The
ordinary approved roots are borrowed; callers retain and close their original
root capabilities after preparation returns.

The first outbound credit is the canonical event-v2 Accepted frame. Normal
progress then publishes exact Token credits, followed by Usage and
Completed-v2 for natural maximum-output or physical stop-token termination.
Stop tokens count toward usage but are suppressed from Token output; their
exact adjacent terminal is authenticated before Usage. Any final valid UTF-8
decoder tail is carried only by Completed. Every frame remains pinned in the
same owner-resident one-credit storage until copied and acknowledged; pinned
credit rejects before worker, scheduler, or decoder mutation.

Ordinary generated-token decode and writer publication use scalar transactional
status after exact scheduler reservation/dequeue. No typed error is created
until cancellation/recovery cleanup state is secured. Natural terminal tail
flush is likewise scalar. String-stop matching is incremental and transactional:
the trigger is counted, stop and post-stop bytes are withheld, and an exact
reserved cancellation terminal is privately translated to canonical `Usage`
then `Completed(StopSequence)`. If the same final token already has an adjacent
natural Maximum/StopToken terminal, that precedence is authenticated and
translated without cancelling an already-terminal request. Caller cancellation
is deferred behind pinned Accepted/Token credit, commits one exact cut after
acknowledgement, and publishes `Usage` then `Completed(Cancelled)`. Every
credit-free `progress` enforces the owner-bound deadline after deferred caller
intent and before worker/publication mutation; expiration publishes `Usage`
then payload-safe nonretryable `Failed(deadline_exceeded)`. `check_deadline` is
a non-latching poll while credit is pinned. Decoder/output/protocol rejection
publishes `Usage` then nonretryable `Failed(output_invalid)`; physical
worker/device loss publishes `Usage` then retryable
`Failed(worker_unavailable)`. Both codes are fixed and preallocated before
rooted startup, and lower errors or payloads never enter the frame. Caller,
deadline, abort, output, and worker-failure cuts fix output at the last retired
public event and suppress decoder-pending bytes; only an authenticated natural
terminal flushes a final decoder tail.

Normal `progress` only latches owned failure state and returns
`OnlineSessionTerminalizationRequired`. Explicit `progress_terminalization`
then runs off-reactor to recover/reap, retire an existing flight, suppress
through the exact terminal, and publish the failure bundle. Recovery retries
retain both cleanup authority and the first product failure cause. Healthy
shutdown, terminal close, and abort cleanup are explicit as well. Child
shutdown/reap may block, so terminalization and cleanup progression do not
belong on an async network reactor.
