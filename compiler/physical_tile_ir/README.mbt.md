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

`FragmentRowDomain` describes the live prefix of instruction row groups. A
partially live final group retains its padded arithmetic; wholly inactive
groups have no producer reads, fragment products or publications. Projection
physical planning constructs this immutable domain once, and the fused-ingress
CUDA backend consumes the same value at all three boundaries. Its finite
specialization dispatch is outside the reduction loop, not a per-MMA runtime
predicate. Full groups retain their original joint product association.

`FragmentProgram::forward_right` performs single-use register forwarding:
one-row, one-slot products may bind each packed producer word directly to its
ordered consumer operand. A bounded bijection rejects omitted, repeated or
foreign words. Multi-row reuse and lookahead lifetimes are rejected rather than
silently reloaded. This transform removes an intermediate representation;
actual register moves are still decided by the device compiler.

`FragmentReadOwnership` composes a forwarding bijection with read-provider
groups. It preserves each source word while placing consumer pairs contiguously;
device lowering supplies the provider width and instruction interpretation.
`OrderedOperandReads` describes read-ahead within one already-published operand
epoch. Slots are primed, consumed in ascending item order and rebound only after
retirement. Both values are immutable; enumerating the bounded actions is
compiler work. Neither transform reassociates arithmetic, changes shared storage
or introduces a runtime publication barrier.

`OperandStorage` packs disjoint row strips in a finite ring allocation.
`VectorOwnership` assigns each copy vector to one owner/round pair. Producer
and consumer bindings therefore derive their offsets from one plan rather than
reconstructing strip boundaries independently.

`TiledPlane` composes a complete logical plane with an inner `OperandLayout`
and contiguous outer tiles. It is a byte-preserving address permutation, not a
quantizer or a new numerical law. The grouped low-bit backend uses it for both
startup weight permutation and step-local activation packing, enabling a single
producer to transfer a whole tile to the consumer's physical layout. Padding
belongs to `OperandStorage`; a complete tiled plane rejects partial tiles and
row-stride holes. Device policy still chooses tile geometry and copy mechanism.

`ProducerConsumerPipeline` describes disjoint producer/consumer owners and
slot reuse by monotonically ordered epochs. Publication follows every producer;
reuse follows every consumer's retirement. The finite descriptor is pure;
CUDA alone chooses phase-counted barriers and copy/MMA thread assignments.
The experimental grouped low-bit expert lowering consumes it with packed-row
materialization from `compiler/grouped_matrix_ir`. Neither descriptor changes
quantization, reduction association or original job-order publication. These
transports are explicit opt-in alternatives, not an assumed performance win.

Current consumers are the matrix, intermediate/down, sibling gate/up,
sibling-reuse and fused-ingress pipelines. Scalar and selected-row reduction
plans live in the projection compiler. Online-softmax state lives in the
separate `compiler/attention_physical_ir` domain dialect. These packages do not
claim that every kernel family has migrated to the new physical layers.
