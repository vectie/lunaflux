# A/B isolated-worker supervisor

This package owns one exact child channel plus physically distinct A/B plan and
completion frame owners. Submission is strictly monotonic and allocation-free
after startup storage. At most two plans may be in flight; the oldest response
is received first.

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

The supervisor retains the immutable encoded source, and replacements receive
the same canonical bytes. The included deterministic child proves protocol
agreement only: it does not read model/module files, reconstruct a device plan,
or establish CUDA readiness. Recovery is explicit and
ordered: the old child must first be closed and reaped, validated completions
remain retryable, and each unreturned `WorkerSubmission` must be committed as a
scheduler worker failure before `abandon_submission` retires its exact sequence.
Only after all obligations are retired can `recovery_startup_contract` derive
the non-reusing predecessor for a replacement child. The supervisor does not
silently infer scheduler mutation or discard in-flight work.
