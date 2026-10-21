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

## Wider operand epochs, October 21

`glm_wide_operand_epochs_20261021.patch` retains the rejected K128 fragment
pipeline against main `ad4cc85b`. Apply only in a disposable diagnostic checkout.
It preserves the immutable physical-plan parameter, private lowering, source
export tests and explicit service-policy propagation used in the experiment.
It is **not** part of the compiled production compiler or serving selection.

Export with `LUNA_FRAGMENT_BF16=1 LUNA_FRAGMENT_WIDE=1` and
`scripts/probe-glm-projection-staging.mbtx NEW_ROOT export-fragments`.
Variant0 is the accepted M32/N64/K64 leaf; variant1 changes only the K epoch
to128. Both use the existing runtime launch domain and ascending MMA fold.
The 64-row matched component regressed from314.517 to444.914us despite fewer
instructions. The shared-memory allocation doubles and its residency limit
falls from four to two CTAs/SM. Do not promote the smaller-row result as a
long-prefill improvement. The complete experiment and loading failures are in
[the October21 follow-up](../../docs/GLM_LOADING_AND_PROJECTION_FOLLOWUP_2026-10-21.md).
