# Selected projection integration audit

Scope: the current projection compiler and its production CUDA realizations,
including scalar/selected-row fallback. This is an integration audit, not a
claim that every schedule is optimal or that numerical acceptance is complete.

| Decision | Common owner | Production consumer |
| --- | --- | --- |
| Concatenated operand row selection | `ProjectionRowSelection` | Scalar, map and matrix QKV lowering |
| Striped vector ownership | `ProjectionVectorTransferMap` / `ProjectionOperandTransferPlan` | Matrix and intermediate operand staging |
| Weight identity composed with vector ownership | `ProjectionWeightTransferPlan` | Matrix-pipeline weight transfer |
| Sibling operand/worker intersection | `partition_projection_transfer_segments` | Phased gate/up transfer |
| Consumer/product distribution | `ProjectionProductFoldDistribution` | Matrix and sibling fold lowering |
| Operand issue/completion/publication order | `ProjectionFoldEffectPlan` | Matrix, sibling and intermediate pipelines |
| Materialization publication and reuse | `ProjectionMaterializationEffects` | Shared map and sibling epilogues |

The finite ring tests cover serial transfer, already-completed prefetch and
pending asynchronous prefetch. Early issue does not imply a pending completion:
the narrow synchronous transfer path now omits empty async waits without moving
its issue point. CUDA vector width, register coordinates, shared permutation,
shuffle instructions and async instruction spelling remain device lowering.
They do not require model-family branches or another identity-only wrapper.

## Actual elimination, not source preservation

The direct sibling register epilogue has no shared-reader lifetime and no longer
emits a trailing warp fence. Down output now exchanges packed registers instead
of materializing a shared result tile. Its source, resource accounting and
companion launch all remove dynamic result scratch. Ownership/tail regressions
and the actual GPU experiments are recorded in
[the register-epilogue report](BENCHMARK_REGISTER_EPILOGUE_2026-09-15.md).

Full runtime integration found an additional obsolete assumption:
`validate_projection_companion` inferred dynamic scratch as `block_x * 32`.
It rejected the new zero-scratch companion during release publication. Commit
`3b2f25d4` removes that layout inference, while retaining the companion's operation,
workspace and launch-geometry checks. The executor preserves the explicit
bounded launch dimensions from the admitted artifact through bucket dispatch.
Regressions exercise both zero-scratch and prior scratch-bearing companions.
This is startup validation only; no token-step check was added.

Clean Linux source `3b2f25d4` passes 3108/3108 native tests. Its kernel tree is
unchanged from `a73c7bb1`; current release binding can therefore reuse those
compiled modules while independently regenerating their expected identities.
This does not reuse an old runtime executable or relabel an old benchmark.

## Separate conclusions

Selected-graph repeatability and the prior uniform/mixed measurements are
complete for their documented commits. They do not establish cross-batch
reference accuracy. The observed propagation differences remain a numerical
diagnostic question, not permission to weaken tolerances or force all shapes
onto a slower bit-identical reduction. Current integrated performance must be
reported from the rebuilt runtime, separately from the isolated down speedup.
