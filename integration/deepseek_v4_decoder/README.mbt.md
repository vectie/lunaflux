# Complete DeepSeek base decoder block

The model adapter composes both checkpoint-backed F32 mHC envelopes, retained
window/compressed attention with learned output, and automatic routed/shared
compact MoE. All launches belong to the containing stage's single queue. One
intermediate residual connects attention to FFN; learned output and expert
combine write directly into their mHC branch results, with no extra copy.

Expert banks, router operands and startup-narrowed token tables belong to the
stage loader. `bank_bytes()` describes the two compact expert capacities;
`required_device_bytes()` describes private frames and streamed non-expert
weights. A whole-stage budget must sum these disjoint allocations, other
borrowed operands and external residual ports before any upload. Close the
borrowing queue, this block, then stage-owned banks and ports.

This is complete **single-rank base-block** composition, not a tensor-parallel
head/expert implementation, a full-model worker or DSpark draft/verify execution.
DSpark's separate prediction blocks must not be replaced by ordinary causal
base blocks. Numerical components alone do not prove whole-model accuracy or speed.

## Contiguous checkpoint stages

`DeepSeekDecoderSlice` uploads complete routed/shared compact banks and router
weights only for its base-layer interval. It owns at most two reusable
inter-layer residual frames and one ordered queue. Request positions, token
IDs, startup-narrowed token-hash tables and learned-index offsets are borrowed;
the caller budgets those once. No checkpoint file access occurs in submit/poll.

`DeepSeekDecoderStage` composes embedding only at ingress, learned output only
at egress, and two distinct boundary residuals. Its aggregate pre-upload budget
includes text weights, compact banks, private state/workspace and residuals.
It requires the worker's explicit absolute-position port. Text prefix/suffix
join the decoder queue; only a prepared complete range can bind whole-model
worker delivery. A partial stage is not an independently runnable model.

Stage sources and negative/preparation tests are not full-checkpoint numerical
execution. Owned startup token-table narrowing, stage-bound activation leases,
two-host runner integration, DSpark prediction execution and complete generation
remain required before claiming the full DeepSeek/DSpark model is runnable.
