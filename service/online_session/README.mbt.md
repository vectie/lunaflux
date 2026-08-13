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

The first outbound credit is the canonical event-v2 Accepted frame. The frame
is pinned in owner-resident one-credit storage until copied and acknowledged;
no transferable frame owner or scalar authentication token is exposed.

This initial foundation intentionally stops before normal token stepping. It
does include the complete off-reactor abort path required to publish a session
without stranding authority: exact cancellation, existing-flight retirement,
suppression through the exact terminal, worker-failure recovery, and clean or
terminal close. Child shutdown/reap may block, so cleanup progression does not
belong on an async network reactor. Natural token/string-stop/deadline output
and the final Usage/Completed/Failed bundle land in the next coherent slice.
