# Does the new compiler outperform the old selector with equal information?

## Answer

**Not in this controlled test.** Old and new selectors chose the same kernel
for all eight GPU-measured workloads and agreed on all 64 resource-budget /
candidate-order queries. Choices were frozen before a separate validation run;
both chose its lower median in all eight cases.

The latest schedule improvement is real, but it does not establish an advantage
of the execution-frontier redesign. A four-line manual backport into an isolated
old snapshot emits exactly the same CUDA source and launch metadata as the new
KV64 choice. That source also matches the previously measured service winner.
This is a causal attribution control, not a recommendation to restore magic
constants or remove the typed schedule contract.

The test concerns the latest execution-economics upgrade, **not every historical
compiler refactor**. It does not disprove the value of preserving alternatives
for different consumers. It shows that this value has not been demonstrated by
the current prefill speedup or the equal-information selection experiment.

## Fixed inputs and protocol

The predeclared protocol and executable diagnostics are in
[`benchmarks/compiler_selection_ablation`](../benchmarks/compiler_selection_ablation/README.md).
AKO was used to hold executable identity, workload, numerical permission and
measurement budget fixed; it did not introduce another production optimization.

| Component | Control |
| --- | --- |
| Old selector | `24ef8d5f5f45fe8f576bb006c1a3d2444adeeeb3`, immediately before continuation-aware frontiers |
| New selector | `b8d23b2f5077c6212de6d6c648b05f4db07c327e` |
| Package | Actual, unmodified `compiler/fusion_regions` sources and tests at each revision |
| Caller | Identical diagnostic caller; one complete single-operation region, two materialized alternatives |
| Toolchain | Both native selectors: moon 0.1.20260920 (`914d7da`), moonc v0.10.14+7d59c7ec9-dev |
| Device | Spark .179 GB10 sm121, UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, PCI `0000000F:01:00.0` |
| Kernels | Frozen Q64/KV128 and Q64/KV64 BF16 paged reference-prefill cubins |
| Data | Head dimension 128, 16 query / 8 KV heads, fragmented paged inputs; varied rows, query count and history |
| Calibration | Five paired trials, 15 repetitions per candidate per trial |
| Validation | Separate five-pair run, same repetition budget; workload and candidate timing orientation reversed |
| Budget queries | Local storage 49,152 / 65,536 / 81,920 / 101,376 bytes; zero scratch; both candidate enumeration orders |

Both selectors receive identical measured costs and resources. They do not
generate new kernels in this experiment. Four budgets times two orders times
eight shapes gives 64 selection queries, **not 64 independent GPU experiments**.
Lower budgets make only KV64 feasible; the main timing table allows both.

Selection is frozen from calibration, uploaded before validation, then evaluated
without reselecting. Validation uses the same shapes at a later time: this is
not an unseen-shape generalization test. Candidate timing pairs share a workload
and process; no claim of formal statistical significance follows from five pairs.

## Independent validation results

Median kernel time in microseconds. KV length includes the current query span;
query count is the current chunk, not total prompt length. These are kernel
workloads, not eight complete model-serving benchmarks.

| Rows | Queries / row | KV length / row | KV128 us | KV64 us | Both select | Reduction versus fixed KV128 |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 8 | 64 | 512 | 43.501 | 44.625 | KV128 | 0.00% |
| 8 | 256 | 8,192 | 2,414.888 | 2,123.492 | KV64 | 12.07% |
| 1 | 2,048 | 30,720 | 6,857.016 | 6,107.292 | KV64 | 10.93% |
| 1 | 7 | 19 | 6.172 | 5.773 | KV64 | 6.46% |
| 4 | 128 | 2,048 | 179.324 | 200.480 | KV128 | 0.00% |
| 16 | 128 | 2,048 | 851.817 | 870.974 | KV128 | 0.00% |
| 2 | 512 | 16,384 | 2,130.526 | 1,800.117 | KV64 | 15.51% |
| 1 | 1,024 | 65,536 | 8,516.678 | 6,928.452 | KV64 | 18.65% |

Both have zero observed regret relative to the lower validation median of these
two candidates. This is not global optimality. KV64 is not universally best:
it is about 11.8% slower for 4 x 128 / 2,048. The 2–3% differences in two other
cells are especially noise-sensitive.

The identical choices mean both dispatch the same measured cubin. Their timing
columns are not separate old/new engine runs, and the computed zero selected-time
difference must **not** be reported as a measured 0.00% whole-service difference.
Compiler search time and memory use were not benchmarked.

## Source-level counterfactual: could the old code produce the winner?

Built the real source-export probes from both full committed snapshots, outside
the dirty working tree. Without overrides, old and new emit identical source
and launch metadata, still KV128. Their SHA-256 is:

`a8def6fcdb029bb53cf652f47202ae3f023812df19f57b9412d121dc2e9da446`.

Then, only in the extracted old snapshot, changed four lines in two files:

1. Reference prefill template KV width: 128 to 64.
2. Matching static shared-storage assertion: 81,920 to 49,152 bytes.
3. Matching key rounding: multiples of 128 to multiples of 64.
4. Matching launch shared-storage requirement: 81,920 to 49,152 bytes.

The old backport and new explicit KV64 exporter produce identical bytes:

`b987815505fa60062a786b08246aa70ab1bf44fe7c3aae4b77acdb39ff5dc51e`.

This is also the hash of `selected.cu` from the
[previous complete-service experiment](BENCHMARK_REFERENCE_PREFILL_SCHEDULE_2026-10-09.md),
which measured 5.22% lower completion time for 32,512-input / 64-output / C1.
That result was not rerun here. Source identity demonstrates the old generator
can express this winning device implementation; it does not prove the complete
old worker admits or deploys that bundle without its own integration changes.

The old backport is a diagnostic control only. It is not automatic tuning, a
safe replacement for every adapter, or a change to production. The new typed
schedule keeps specialization, rounding and resources consistent; that is a
real maintenance benefit, distinct from a demonstrated optimizer speedup.

## Why the architecture did not distinguish itself

At the tested new revision:

- `kernels/luna_fusion_plan/ingress_regions.mbt` and
  `kernels/luna_cuda_attention_tile_aot/chain_selection.mbt` do invoke the shared
  selector. It is not wholly disconnected code.
- Both construct alternatives with the default `Materialized` continuation.
  Non-default `Scoped` continuation classes occur in tests, not these production
  callers. The ability to preserve a locally slower producer for a better
  consumer layout therefore does not yet distinguish these executions.
- `kernels/luna_cuda_reference_attention/plan.mbt` still defaults to
  `Query64Kv128`. The candidate exporter emits both choices, but this prefill
  construction path does not consume measured observations to choose between
  them. The benchmark package's KV64 route was explicitly selected.
- With the same complete materialized alternatives, measured latency and final
  resource limits, the old selector already finds the same feasible minimum.
  A frontier is additional representation capability, not additional hardware
  performance when it leaves the execution unchanged.

The newer joint decode enumeration, timing import and artifact binding are useful
connected machinery. This prefill control neither measures nor dismisses those
features. It isolates ranking with equal alternatives and information.

## What would demonstrate additional compiler value?

The unproven part is a connected **producer–consumer** optimization: enumerate
real compatible layout/ownership alternatives, retain meaningful continuation
classes, measure complete chains under the same budget, and bind the resulting
winner without a manual kernel override. Compare with the old locally selected
plan on independent validation and complete-service latency.

Record candidate generation, feasible frontier, chosen continuation, artifact
identity and workload together. Include shapes on which the current choice
loses. Repeating a minimum-of-two materialized-kernels test, adding more IR names,
or crediting a manually chosen kernel to the architecture would not answer that
question. This report does not implement that next experiment.

## Correctness, resources and reproducibility

All fresh probes passed the existing independent sampled scalar oracle, full
pairwise output tolerance (0.003), deterministic rerun, read-only operand and
inactive-output checks. Numerical permission remains BF16 probabilities and
base-two softmax; no bitwise or model-quality equivalence is claimed. Cubins are
unchanged from the prior qualified experiment; sanitizers were not rerun for a
diagnostic selector comparison.

Actual GPU processes ran under user-systemd `MemoryMax=8G`, `MemorySwapMax=0`
and 120-second per-probe deadlines. GPU runs were serialized and had a 32-GiB
available-memory reserve. Minimum sampled available host memory was 119,970,108
KiB (about 114.4 GiB), not a continuously measured peak. Cgroup accounting is
not a measurement of all GPU allocation. The initial outer-process-only cgroup
attempt is preserved but excluded from the primary results. The GPU was idle
at completion; production was unchanged.

Validation:

- Unmodified old selector: strict native check and 7/7 tests.
- Unmodified new selector: strict native check and 13/13 tests.
- Both isolated modules: `moon info` and formatting checks.
- Probe-output adapter and frozen-choice evaluator: 1/1 test each.
- Verification asserts 8/8 decisions, 64/64 budget/order queries, 8/8 validation
  medians, memory reserve and unchanged versioned selector package bytes.
- Full source-export builds use existing migration warning exclusions
  `-79-20-29-25-92-14`; the strict selector checks above use no exclusions.

Remote evidence: `/home/wlc004s/lunaflux-compiler-value-20261009.nH0Su3Wh`.
Primary measurements are `bounded/calibration` and `bounded/validation`.
The downloaded archive and every member of `FILES.sha256` verified locally.
Remote archive SHA-256:

`57af98bf10b9564bda9c27b9b5e726f10c30969cbdb377c0d89e6f088585b811`.

Local experiment: `/tmp/lunaflux-compiler-value-20261009.ehIbrw`, including
`frozen.json`, `results.json`, `summary.json`, actual selector snapshots/binaries,
`source-control` and `remote-evidence/verified`. The original raw timing records
and all losing cases remain available. Harness source is tracked separately;
no experiment outputs or extracted compiler copies enter the production tree.

Local selector/source bundle: `selection-source-evidence-v1.tar.gz` in that
experiment directory, SHA-256
`ccddb3a7884f895d8de47d5e06241e06126d4826444bdf3d0a5eaf3b12d6060d`.
It preserves the isolated versioned packages, selector binaries, both original
full-source archives, generated-source comparison logs and decision records.
