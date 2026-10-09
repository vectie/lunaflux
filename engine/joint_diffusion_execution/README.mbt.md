# Prepared joint diffusion execution

Owns one prebound frame per paired diffusion step. Frames contain joint
prediction followed by both RF updates and wait for completion before returning.
The array is snapshotted once; advancing performs no preparation or tensor
allocation. Cancellation is observed between frames, never claimed to preempt
a kernel. A partially executed frame poisons the underlying loop permanently.

Call `close` on success, cancellation and failure. It stops submissions, drains
then releases each frame, and retains a retry cursor if cleanup fails. Do not
drop the owner after a failed close. Resources referenced by multiple frames
must use leases; only the frame owns its queue, not the shared allocations.

This is not a denoiser implementation. Startup binding is responsible for exact
step coefficients, graph ordering, non-duplicated handles and model identity.
