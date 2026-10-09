# Precision / representation refinement

Pure, immutable plans separate storage precision from activation/accumulator/
output precision. Conversion placement and accuracy bounds are semantic inputs.
Startup writes, ordinary step writes and transactional cache commits are explicit
effects. Quantization publication stages are planned before device lowering.

`PhysicalOperand` refines an existing physical operand region; it does not invent
an unrelated shared-memory map. Local/global F32 scaling and BF16 rounding are
retained boundaries. A caller-provided supported-placement set is a capability
input, not an assertion that arbitrary hardware implements low-bit MMA.

`WeightSchema` refines memory over the whole graph's ordered tensor bindings.
Alignment and sequential/concurrent scratch lifetimes are explicit; per-tensor
budgets cannot substitute for the aggregate device memory ceiling.

ExpertMlpPrecision additionally plans three compact expert projections,
noncontiguous rank placement, a shared packed bank, and bounded BF16
intermediate/weighted workspaces. PackedBufferLayout is the single source of
payload, block-scale and scalar-scale offsets for both upload and lowering.
Rank-local contributions are summed in F32 before any cross-device reduction.
The compute implementation uses pairwise F32 dot reductions and explicit BF16
stage rounding; it does not promise bitwise equivalence to a serial dot product.

See [implementation, tests and remaining serving work](../../docs/PRECISION_IR_2026-10-04.md).
