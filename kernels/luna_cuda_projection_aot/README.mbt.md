# BF16 projection-family CUDA AOT lowering

This native, offline-only package lowers the paged-graph BF16 QKV projection,
dense output projection, gated MLP, and language-model head into closed,
deterministic reference CUDA sources and recipes. It does not open files,
invoke a compiler, load CUDA, or participate in the request path.

Every candidate lowering authenticates the semantic operation shape, canonical
`StepCounts`/activation/row-major-weight/output operand order, exact BF16 byte
counts, fixed alignments, device target, compiler policy, and launch geometry.
The source/recipe API consumes the typed operation, profile, stable entry-point
ID, operands, target, and compiler policy but deliberately cannot see a module
digest or final catalog family. After two deterministic offline compilations,
`bind_manifest_projection_cuda_aot` joins that candidate to the exact admitted
full-graph contract and emits a separate binding record containing the final
module, family, workspace, dimensions, and operands.
Generated math uses explicit ordered FP32 multiply/add operations and one BF16
round at the output; gated MLP additionally binds the `expf` SiLU contract.

The existing narrow cuBLASLt seam remains the preferred optimized execution
primitive for an isolated dense row-major projection. It is currently
synchronous and cannot express QKV concatenation or gated MLP as one operation,
so it is not substituted into the ordered paged AOT manifest by this package.
The sources here are correctness-grade release artifacts, not performance
claims, and physical CUDA evidence remains a separate gate.

`fixtures/physical_sm120` records the exact generated CUDA and recipe bytes for
small QKV, dense-projection, gated-MLP, and language-model-head numerical
shapes. `physical_fixture_wbtest.mbt` binds both SHA-256 values to fresh typed
candidates; the shared standalone CUDA Driver probe compares the checked source
against an independent host referee. The recipe fixtures are explicitly
non-bindable and contain no module digest; only the post-compile binding record
may carry that identity. The CUDA source fixtures did not change during this
ownership correction. No checked fixture is a physical-CUDA pass.
