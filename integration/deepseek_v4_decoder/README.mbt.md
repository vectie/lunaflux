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
base blocks. Whole-model placement/loading, text ingress/egress and generation
remain integration work; numerical components alone do not prove whole-model
accuracy or speed.
