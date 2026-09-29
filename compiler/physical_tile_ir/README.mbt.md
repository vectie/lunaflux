# Physical tile IR

This package owns device-neutral descriptions of shared operand layouts and
register-fragment dataflow. It is a first shared physical dialect, not a
replacement for domain-specific projection or online-softmax semantics.

`FragmentProgram` describes a product of ordered reduction folds. Operand
values have stable IDs, products reference shared values, and accumulators
remain independent and ordered. A finite one/two-slot register schedule makes
lookahead and tail reads explicit without changing arithmetic association.
`OperandLayout` describes a row-preserving vector permutation. Backend policy
selects its parameters; producers and consumers use the same layout value.

Both descriptors are immutable, constant-space, and checked at construction.
They contain no model names, CUDA instructions, warp width, or runtime state.
The CUDA backend binds values to buffers and instructions only at emission.

`OperandStorage` packs disjoint row strips in a finite ring allocation.
`VectorOwnership` assigns each copy vector to one owner/round pair. Producer
and consumer bindings therefore derive their offsets from one plan rather than
reconstructing strip boundaries independently.

Current consumers are the matrix, intermediate/down, sibling gate/up,
sibling-reuse and fused-ingress pipelines. Scalar and selected-row reduction
plans live in the projection compiler. Online-softmax state lives in the
separate `compiler/attention_physical_ir` domain dialect. These packages do not
claim that every kernel family has migrated to the new physical layers.
