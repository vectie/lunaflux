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
  exporter now enables asynchronous decode under an explicit CUDA-backend
  48-KiB per-block occupancy budget. The generic cost model distinguishes
  per-key subgroup softmax updates from transfer-tile synchronization; it no
  longer charges extra semantic folds just because staging tiles are smaller.
  Current-source end-to-end comparison of this new selection is pending.
- Matrix: output-map distribution is selected through the generic strategy,
  compiler, CUDA source, local-storage calculation and graph launch geometry.
  The artifact producer accepts typed offline records. See the
  [paired matrix/storage report](MATRIX_STORAGE_SCHEDULES_2026-09-06.md).
  Device/toolchain-scoped persistence and matching exporter/release-binder
  consumption are now implemented; see [offline tuning](PROJECTION_TUNING_2026-09-06.md).
  A measured whole-profile head choice is now installed and benchmarked.
  Per-bucket artifact selection remains distinct from this input plumbing.
- Fusion/storage: selected-input residency is a separately measured schedule,
  not mandatory materialization after CSE. The pure complete-span cost selector
  now has an offline exporter consumer for full, producer-separated and unfused
  ingress. See [fusion selection](FUSION_PROFITABILITY_2026-09-06.md).
  Per-bucket alternatives and a measured installed ingress comparison remain
  separate from the completed whole-profile selection mechanism.
- Work planning: pure bounded progress-to-query calculation is consumed by
  scheduler selection; typed query/context/KV-write ranges are consumed by both
  descriptor paths. Existing fairness and KV allocation policy are preserved.
- Output demand: model-level dead-suffix analysis retains the ordered prefix
  through the last persistent-state effect. Eager executors materialize a
  shorter reusable queue at startup; both descriptor paths select it only when
  every row has no output demand. They also omit unused sampling readback.
  The BF16 matrix head and sampler now consume a compact terminal-row view in
  mixed frames and captured execution, with stable pointers and original-row
  result mapping. Upstream KV effects retain the original descriptor. No second
  graph is allocated; captured upstream pure suffix operations are not pruned.
  FP8/I8 envelopes and diagnostic canaries retain their original contract.
  See [output-row projection](OUTPUT_ROW_PROJECTION_2026-09-06.md).
  The exact-commit queue regression and four GPU sanitizer checks passed;
  see [the execution report](OUTPUT_DEMAND_EXECUTION_2026-09-06.md).

Current-source installation, compact-head physical checks and the eight-cell
Qwen result vector are recorded in
[the implementation report](COMPILER_REMAINING_RESULTS_2026-09-06.md), including
the existing concurrent-output variation and remaining work. No all-workstreams
completion or batch-invariant generation claim is implied.
