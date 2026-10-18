# Batched tensor-parallel publication

The runtime owner executes one canonical batch on a local rank and a remote
token-producing rank. The model plan binds all collectives before startup;
this package knows no model, GPU backend, checkpoint, or reference comparison.
Inputs, execution, and commit receipts are separately retired. The caller
publishes completions only after both ranks commit. A rank failure poisons the
whole step; it is never retried against partially mutated state.

Request retirement carries the original identity and generation, not a batch
ordinal. Cancellation remains scheduler-owned: drain the active step, suppress
that request's response, and retire its state before cache pages are reused.
Other live requests retain their state. This package does not itself implement
HTTP, token sampling, prefix reuse, or admission limits.
