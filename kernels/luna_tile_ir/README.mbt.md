# LunaTile IR

`kernels/luna_tile_ir` owns the smallest typed, deterministic offline kernel
program used by LunaFlux Phase 5. Tensor views bind dtype, address space,
checked 64-bit byte offsets, shape, byte strides, and alignment. Affine tile
origins are bounded by explicit loop extents. Instructions cover synchronous
and asynchronous copies, staged barriers, MMA, reductions, and elementwise
operations. Generic programs have exact instruction semantics and publish an
opaque operation inventory, but cannot assert a model-operation family. The
separate residual-add constructor remains the only family-level semantic
contract: admission proves its ordered roles, BF16 register views, full width,
and single Add algorithm before publishing it. Embedding, RMSNorm, projection,
RoPE, attention, and MLP family claims still fail closed until their complete
algorithms receive separately typed semantic contracts.

Admission validates identifiers, tensor bounds, affine arithmetic, copy and
MMA contracts, compute address spaces, the shared-memory envelope,
nonoverlapping shared regions, stage/barrier order, and compile-time
constraints before it publishes an immutable `LunaTileProgram`. Canonical
bytes and their SHA-256
identity are deterministic. Constraint order is canonicalized while execution
order is preserved.

`plan_luna_tile_cuda_aot_input` produces a versioned canonical AOT planning
input and explicit shared-memory/vector/pipeline plan. Async copies must retain
at least four-byte aligned vectorization; synchronous copies deterministically
narrow to their admitted alignment. `lower_luna_tile_cuda_translation_unit`
then lowers the complete generic instruction sequence to deterministic CUDA
source with a stable numeric entry-point identity. Its fixed one-block,
one-thread launch guard makes it a semantic reference translation rather than
a performance claim. The package-private bounded decoder continues to feed the
host residual-add plan canary from the planned representation.

The translation unit is inert bytes. This package does not invoke a compiler,
open a filesystem, acquire a device, load a module, modify a manifest, or grant
promotion/runtime-JIT authority.

`specialize_luna_tile_parallel_candidate` builds a second, separately identified
offline artifact above that serial oracle. Its policy fixes the target, block,
warp, tile, vector, and pipeline geometry. Every affine tile is mapped by
round, block, warp, and lane; global writes require a conservative proof that
tiles are disjoint, and any tensor that is both globally read and written must
use one exact region so a cross-worker alias cannot be hidden. Shared tensors
receive deterministic interval-lifetime placement and reuse, then each thread
receives a stage-ring slice for uniform async-copy/compute overlap. These
choices, the exact compute capability, all placements, the emitted source, and
the serial-oracle digest are bound by the candidate canonical identity.

Tensor-core eligibility is exact but descriptive: an eligible program is
recorded as `eligible_unqualified` and still emits the parallel SIMT candidate.
The package rejects a caller request to require tensor cores because it owns no
physical qualification authority. The serial translation is qualification-only
and is never selected as the candidate execution route. A separate test-only
exporter and campaign can compile and exercise the source while preserving
those identities, but no NVIDIA run has occurred. Physical CUDA correctness,
device numerics, profiler-led tile selection, sanitizer/race/resource evidence,
benchmarking, manifest admission, and promotion remain external Phase-5 gates.
