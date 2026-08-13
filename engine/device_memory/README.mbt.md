# Device activation memory

`engine/device_memory` turns an immutable `StaticDevicePlan` into one bounded,
deterministic BF16 activation/workspace layout. It computes value liveness,
reuses aligned physical slots only after their final consumer, retains the
final output through completion, and appends one shared aligned kernel
workspace.

The startup allocator creates one device allocation for the complete layout.
Token-step execution only consumes immutable regions and operation bindings;
this package contains no per-token allocation, model-family dispatch, CUDA
import, global context, kernel launch, or performance claim.

The token-row envelope is checked from the static plan's maximum batch and
sequence dimensions. The caller supplies alignment and the exact total byte
ceiling. Unsupported shapes, arithmetic overflow, and capacity excess fail
before device allocation. Native resource ownership is explicit through
`DeviceMemoryArena::close`.

The existing stateless reference executor still consumes the arena's frozen
Phase-1 allocation-alias exception. New paged execution must use an
owner-mediated synchronous launch boundary instead: public `Allocation` or
`KernelArgument` returns are retainable values and therefore cannot enforce
activation liveness or poison semantics. `engine/device_executor` owns removal
of this exception after its launch and output-copy paths become owner-mediated,
and no Phase-3 paged executor may depend on it.
