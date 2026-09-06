# Functional compiler implementation workstreams

This is the active implementation checklist, not a completion or performance
claim. Existing model-family work in the checkout is outside this change.

1. Matrix schedules: immutable output/reduction tiling, work distribution,
   resource estimates and device/shape-scoped offline selection. Wire selected
   schedules through source and launch geometry together. Preserve the numeric
   contract and the single-row path.
2. Fusion profitability: separate semantic sharing from storage placement and
   fusion choice. Compare complete alternatives, including intermediate traffic,
   barriers, occupancy and launch costs. No model-name tuning rules.
3. Attention dependencies: keep the query outside the KV fold; reuse page
   addresses; separate K-score and V-update readiness; retain ordered folds and
   the logical partition grain independently of transfer tiles.
4. Work planning: derive query/context/KV-write ranges and output demand from
   request progress. Keep scheduling backend-neutral and execution selection
   allocation-free using startup-owned tables.
5. Output demand: eliminate unused output-head and sampling work without
   eliminating KV effects. Preserve row identity, mixed-batch semantics and
   counter-addressed RNG. Capture variants must include their memory cost.

For each workstream, completion requires an actual compiler/executor consumer,
focused regression tests, and physical correctness/performance checks where
generated kernels or execution ordering change. New IR declarations alone do
not count as integration. Measurements select serving defaults; losing
alternatives remain disabled.

## Progress

- Attention: query residency, reused transfer addresses and separate K/V
  readiness implemented and physically measured. See
  [the paired report](ATTENTION_DEPENDENCY_PIPELINE_2026-09-06.md). The serving
  default and end-to-end comparison are not yet updated.
- Matrix: output-map distribution is selected through the generic strategy,
  compiler, CUDA source, local-storage calculation and graph launch geometry.
  The artifact producer accepts typed offline records. See the
  [paired matrix/storage report](MATRIX_STORAGE_SCHEDULES_2026-09-06.md).
- Fusion/storage: selected-input residency is a separately measured schedule,
  not mandatory materialization after CSE. Full/partial/unfused ingress
  profitability selection is still unfinished.
- Work planning: pure bounded progress-to-query calculation is consumed by
  scheduler selection; typed query/context/KV-write ranges are consumed by both
  descriptor paths. Existing fairness and KV allocation policy are preserved.
- Output demand: represented in row work, but GPU head/sampling omission is
  **not implemented**. No effect-only capture or mixed-row output compaction
  is claimed. Production default selection and end-to-end validation remain.
