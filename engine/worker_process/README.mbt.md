# Isolated-worker supervisor

The legacy transport supervisor owns one exact child channel plus physically
distinct A/B plan and completion frame owners. It admits at most two plans for
deterministic echo/transport fixtures. The production root-bound facade retains
the same preallocated storage but admits exactly one outstanding plan because
the real device child reads, executes, and completes plans serially.

Construction preflights the full bootstrap-source receiver capacity and exact
source digest before spawn, then performs `Configure -> BootstrapSource ->
Ready`. The child canonically decodes the bounded source bytes, compares their
digest with Configure, and returns the exact model identity, bootstrap/source
identities, model generation, predecessor, and runtime limits. Startup framed
I/O has its own validated per-prefix/per-payload timeout; steady plan traffic
continues to use the separate I/O timeout.
Incompatible children are closed before publication. If both handshake and cleanup fail, preparation
returns opaque retained cleanup authority so the child is never abandoned.
Submission also rechecks the loaded model generation before any frame write.

A response side remains pinned after wire validation. Callers may inspect its
exact frame repeatedly while scheduler publication is backpressured, and only
`retire_received` permits that side to be reused. Foreign, stale, out-of-order,
malformed, partial, timed-out, or closed-channel traffic fails closed. Native
process handles and transport buffers never escape.

The compatibility-only `prepare` entry remains for deterministic echo fixtures.
Production service construction uses `prepare_with_approved_roots`, which
privately duplicates caller-owned model and kernel roots. Its root-bound owner
retains that exact opaque pair with the immutable executable bytes, process
limits, startup sequence domain, and encoded source. Replacement is
zero-argument and therefore cannot substitute another executable, limit set,
source, or root pair.

The supervisor retains the immutable encoded source, and replacements receive
the same canonical bytes and pinned root capabilities. The legacy
`worker_echo` child proves two-slot protocol agreement only. The device worker
child reconstructs admitted inputs and readiness, then runs the serialized
steady-state plan/completion loop. Recovery is explicit and ordered: the old
child must first be closed and reaped, validated completions
remain retryable, and each unreturned `WorkerSubmission` must be committed as a
scheduler worker failure before `abandon_submission` retires its exact sequence.
Only after all obligations are retired can `recovery_startup_contract` derive
the non-reusing predecessor for a replacement child. The supervisor does not
silently infer scheduler mutation or discard in-flight work.

Recovery closes only the child; the root pair remains live across replacement.
Instance/service retirement attempts child and root cleanup independently.
Busy root close preserves retry authority, while a consumed close failure
remains bounded evidence without claiming a live pair. Failed replacement-child
cleanup stays inside the root-bound owner and must be retried before restart;
terminal service close is likewise retryable.
