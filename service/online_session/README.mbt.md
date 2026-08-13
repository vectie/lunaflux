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
flush is likewise scalar. This slice deliberately rejects string-stop requests
before rooted startup; caller cancellation, deadline translation, and public
Failed events remain later work.

The complete off-reactor cleanup path prevents stranded authority: existing
flight retirement, suppression through exact terminal, worker-failure
recovery, healthy shutdown, and terminal close. Child shutdown/reap may block,
so cleanup progression does not belong on an async network reactor.
