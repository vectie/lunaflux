# Exact-workload schedule calibration

`calibrate_selected_policies.mbtx` takes:

```text
BENCHMARK_ROOT NEW_EMPTY_ROOT PROBE_SOURCE PERFORMANCE_CLASS SELECTED_CELL
```

Example selected cell: `prefill-q2048-r8-h2048`. Performance class begins
`perf-v1:` and is an operator-declared hardware/configuration class including
target, SM count, cache, memory configuration and relevant power/clock policy.
ISA compatibility alone does not imply performance suitability. UUID is recorded
separately. No per-UUID benchmark is required for ordinary compiler fallback.

Upload `selected_policy_geometry.h` alongside `selected_policy_probe.cu`.
The probe now mirrors runtime token/tile/metadata capped launch rules and emits
actual/captured geometry. Metadata uses profile-row tail capacity, not a
hand-picked evenly distributed-row launch. The standalone C++ geometry test
checks this against the same capacity examples as device_step regressions.

The ten cells retain five-trial medians independently. There is no implicit sum
across short/long, prefill/decode or different concurrency. Each cell contains
its v2 ingress and attention records plus `selected` AOT export. `workloads.v2`
lists their names, selected cell, compatibility class and UUID. Root `selected`
is a separate copy of the explicitly named cell, so `run_selected_runtime.mbtx` still
works. The opposite attention phase has no fabricated observation: it retains
the compiler's unmeasured default. `select_policy_subset.mbtx` can still build
a finalist from the root selected-cell table.

Records retain the original compiler frontier/toolchain/source identity.
Blockwise-F32 decode candidates 450/451 have a distinct numerical law; the
probe reports it and uses an independent attention oracle and declared error
bound, not strict per-key equivalence. Ingress remains bitwise checked.

This removes the maximum-envelope and accidental summed-objective biases.
It does **not** implement runtime cross-cell multi-artifact graph dispatch or
prove a whole-serving winner. Run matched complete serving chains for finalists
and keep their request mix/output length/concurrency explicit. No chain cost
can be inferred by summing these isolated kernel medians. Hardware calibration
and end-to-end comparison remain required before any speed claim.

## Explicit all-new experiment

`prepare_committed_runtime.mbtx MATCHED_BASE NEW_ROOT COMMITTED_SOURCE_ARCHIVE`
rebuilds serving executables from the committed archive, retaining the matched
campaign's documented Spark target substitutions, model and baseline kernels.
It never substitutes old runtime executables for a current-source build.

An optional `--all-new-finalists` suffix on calibration bounds the experiment
to the ingress reference, previous row-packed winner and legal packed-head
policies. The filter accepts `-f1-h2`, not only IDs ending in `-f1` or `-f1-rN`.
All attention/decode candidates and all ten cells are still measured.

`select_all_new.mbtx CALIBRATION NEW_EMPTY_ROOT PREFILL_CELL DECODE_CELL`
selects the fastest **eligible new-family** ingress, fragment-forwarded prefill
and blockwise decode candidate from those named cells. It retains the original
records and export report verbatim. `selection-plan.txt` is the explicit
experimental override; this is not an unconstrained winner or automatic
cross-cell dispatch. Existing aliases are preserved under `default-*` in the
new output. A slower new-family result must be published as a regression.

The serving runner checks packed-head and blockwise identities both in recipes
and in the final runtime bundle. Differential/oracle and memcheck/racecheck/
synccheck coverage includes the exact packaged decode cubin as well as ingress
and prefill. Serving remains sequential, with a 32-GiB available-memory reserve
and 64-GiB per-unit memory ceilings.
