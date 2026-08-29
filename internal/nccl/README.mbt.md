# Private NCCL lifecycle boundary

This native-only package dynamically admits the NCCL v2 runtime SONAME, its
minimal communicator-lifecycle symbols, and exact all-reduce/all-gather entry
points. The immutable dispatch table and DSO handle remain bounded
process-lifetime loader state so communicator function pointers cannot be
invalidated by unload.

MoonBit sees only payload-safe runtime availability/version evidence, opaque
128-byte rendezvous identities, and one explicit communicator owner per rank in
the supported 2..16 local-rank world.
No raw handle, vendor status/text, topology, model, or scheduler type crosses
the package boundary. The execution method borrows vendor-neutral opaque
context, region, and ordered-queue tokens. Only the CUDA-owned interop seam can
resolve those tokens or retain their resources. Group generations are nonzero. Each admitted
launch carries its nonnegative operation ID, collective kind, nonzero plan
sequence, and exact one-based collective sequence. This package authenticates
generation, global sequence, and increasing operation order within a plan; it
does not match a site operation, kind, rank, or world size to the immutable rank
plan. That match belongs to the device execution bridge. A scalar substitution
here latches failure and requires abort.

Runtime admission requires NCCL 2.14.3 or newer and the nonblocking
communicator-init/async-error ABI. The device interop seam provides a
nonblocking ordered-queue completion query. Communicator startup and each
collective are polled; no thread remains blocked inside native synchronization.
The production submit holds one native lock across scalar authentication,
opaque resource-lease acquisition, and NCCL enqueue. One fixed inline lease
retains every device operation guard and Moon reference until successful poll completion
or consuming abort. A failed async-error or stream query retains those guards,
latches failure, and forbids continuation. Only a completed poll releases the
guards and commits sequence progress. Admission, invalidation, close, and abort
serialize through the thread-confined native owner state.

NCCL defines destroy and abort as consuming operations: after either returns,
the handle is never accessed again even if the call reported failure. Explicit
close/abort is the production ownership rule. The finalizer is only defensive;
it aborts a still-live communicator and terminates if native authority cannot be
consumed.

Any non-success destroy/abort result poisons the owning worker process. Native
authority has still been consumed and released exactly once, but service
continuation is forbidden. The sanitizer probes prove end-to-end reference
balance, substitution and alias rejection, and hostile pending/failure state
transitions through the exact production translation units.

The fake seam and exact-translation-unit sanitizer require neither NCCL nor
CUDA. They do not prove physical multi-GPU initialization, cross-rank order,
timeout behavior, rank-loss recovery, topology support, BF16 numerical
correctness, resource balance under a real NCCL runtime, or performance.
