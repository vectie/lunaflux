# Batched tensor-parallel publication

The runtime owner executes one canonical batch on a local rank and a remote
token-producing rank. The model plan binds all collectives before startup;
this package knows no model, GPU backend, checkpoint, or reference comparison.
Inputs, execution, and commit receipts are separately retired. The caller
publishes completions only after both ranks commit. A rank failure poisons the
whole step; it is never retried against partially mutated state.

After both input-readiness receipts, the coordinator flushes the bounded peer
run command before enqueuing its local device frame. A partial socket write
yields in an explicit dispatch state. This allows host graph submission on
both ranks to overlap; it does not wait for remote execution before submitting
locally, change the compiler's device effect order, or weaken final retirement.

Request retirement carries the original identity and generation, not a batch
ordinal. Cancellation remains scheduler-owned: drain the active step, suppress
that request's response, and retire its state before cache pages are reused.
Other live requests retain their state. This package does not itself implement
HTTP, token sampling, prefix reuse, or admission limits.
