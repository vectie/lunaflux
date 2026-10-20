# Rejected physical projection trials

`glm_tensor_transfer_20261020.patch` is an **offline experiment**, not a
selected compiler/backend implementation. The first transport is slower;
the partitioned variant fails its predeclared arithmetic comparison gate.
Do not apply it to a serving build or treat it as a qualified optimization.

It preserves the exact immutable plan/lowering, export tests, startup tensor
descriptor harness and trial-runner changes used in the October 20 campaign,
against main `2801ab8e`. Apply only in a disposable diagnostic checkout.
The compiler keeps no dependency on vLLM, profiling, or comparison.

The `.mbtx` exporters are:

- `scripts/probe-glm-projection-staging.mbtx NEW_ROOT export-rectangular`
- `scripts/probe-glm-projection-staging.mbtx NEW_ROOT export-partitioned`

Preparation/run/profile instructions and all successful and failed evidence
roots are in
[the campaign report](../../docs/GLM_TENSOR_TRANSFER_PROJECTION_2026-10-20.md).
No model payload hashing is part of this workflow.
