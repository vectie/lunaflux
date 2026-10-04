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

See [implementation, tests and remaining serving work](../../docs/PRECISION_IR_2026-10-04.md).
