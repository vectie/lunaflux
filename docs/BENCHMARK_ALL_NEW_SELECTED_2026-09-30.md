# All-new compiler paths: serving benchmark

**Result: the new paths execute, but this combined selection regresses.**
4096/64 C16 falls from 190.48 to 132.39 output tok/s; 4096/256 C1 falls
from 101.19 to 18.42 tok/s. The old split-prefill route still handles a small
part of the traced work; this is not an all-kernel replacement claim.

This experiment forces the newly implemented families into the Qwen3-0.6B
serving package. It is not an assertion that every new candidate is faster,
and it does not change production deployment or replace the prior fastest
selection.

## Selection and source

| Workstream | Selected realization |
| --- | --- |
| Multi-head and rotary reuse | `ingress-g4-a4-s2-t1-f1-h2`, two complete heads per CTA, 16 query rows |
| Blockwise decode | c450, 32-key tiles, `blockwise-f32-probability-v1` |
| Fragment forwarding | Current c322 query64/KV64 asynchronous prefill; canonical main and wide entries use the same source |
| Exact-workload tuning | Ten workload-scoped calibration cells, five paired samples; measured new-family finalists selected for the prefill/decode objectives |

The objectives are prefill `(2048 queries, 8 rows, 2048 prior tokens)` and
decode `(16 queries, 16 rows, 4095 prior tokens)`. Calibration uses runtime
bucket launch geometry. This package is a **forced-new-family measured
finalist**, not the unconstrained fastest package. Automatic per-cell runtime
dispatch is still absent; the ten recorded cells do not imply ten live routes.

Kernel source is commit `8f87f6d8`, including the four-path implementation
`7b6f0468`. Startup-only overlays are `14327978` and `ea398327`.
The latter fix v7 recognition through materialization, kernel-root assembly
and worker bootstrap. These overlays do not change the calibrated kernels.
Spark target substitutions remain isolated to the benchmark snapshot:
sm121 and CUDA 13.0.88. Unrelated working-tree changes are excluded.

The run also exposed an unused legacy rotary wrapper emitted beside the new
prepared-value implementation. Commit `8f87f6d8` removes the unreachable
wrapper from this lowering, retaining strict CUDA warning checks. The affected
lowering suite passes 43/43 tests; startup classifier coverage passes 25/25.

## Measurement contract

- Spark GB10, 48 SMs, UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.
- Same Qwen3-0.6B BF16 model and token-ID workload as the
  [previous selected benchmark](BENCHMARK_SELECTED_SCHEDULES_2026-09-30.md).
- Input/output vectors 128/32, 4096/64, 4096/256; concurrency 1, 8, 16.
- One warm-up and three measured trials; medians, not best trials.
- Greedy generation, requested lengths enforced, EOS ignored, prefix reuse off.
- Serialized GPU work; 32 GiB available-memory reserve, 64 GiB serving-unit
  limit and no extra swap. Profiler capture is separate from timing trials.
- Prior LunaFlux, vLLM and SGLang results are historical matched runs, not
  fresh simultaneous baseline measurements. Their pinned images and settings
  are recorded in the previous report. Cross-run drift remains a limitation.

## Isolated results before serving

Times below are unprofiled probe medians on identical runtime-bucket inputs.
They are not full-model timings or additive estimates of serving overhead.

| Operation and objective | Previous family | Selected new family | New / previous |
| --- | ---: | ---: | ---: |
| Ingress, 2048 queries / 8 rows / 2048 history | 742.951 µs, row32 | 1289.821 µs, h2 | 1.74× |
| Decode, 16 queries / 16 rows / 4095 history | 1273.099 µs, c441 | 1809.501 µs, c450 | 1.42× |

The new prefill finalist is c322 at 672.638 µs for the prefill objective.
The ingress h2 lowering reports 90 registers/thread and two resident blocks;
prefill reports 230 registers and two blocks; blockwise decode reports 72
registers and five blocks. Resource counts alone do not determine speed.

All nine checks of the **packaged** ingress/prefill/decode cubins pass:
memcheck with leak checking, racecheck and synccheck. No CUDA errors, leaks or
race hazards are reported. Sampled independent attention-oracle maximum
absolute errors are 0.000407418 for prefill and 0.000243139 for decode.
These are bounded numerical checks, not a broad model-quality evaluation.

Blockwise decode has a distinct numerical contract and entry symbol; it must
not inherit the old split-K entry points. Consequently this all-new experiment
also removes the old split-decode route. End-to-end regression cannot be
attributed solely to one instruction change in the c450 kernel.

## Results and actual-launch verification

The fresh all-new run completes all nine cells and drains with acknowledgement,
child exit code zero and closed child state. Runtime stderr is empty.
Completion medians in milliseconds (lower is better):

| Input/output | C | Prior selected LunaFlux | All-new LunaFlux | Historical vLLM | Historical SGLang |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128/32 | 1 | 246 | 311 | 298 | 290 |
| 128/32 | 8 | 322 | 402 | 283 | 284 |
| 128/32 | 16 | 388 | 465 | 324 | 325 |
| 4096/64 | 1 | 747 | 3502 | 772 | 780 |
| 4096/64 | 8 | 2845 | 5552 | 2275 | 2340 |
| 4096/64 | 16 | 5376 | 7735 | 4207 | 4237 |
| 4096/256 | 1 | 2530 | 13897 | 2705 | 2732 |
| 4096/256 | 8 | 7707 | 16749 | 6618 | 6831 |
| 4096/256 | 16 | 14261 | 19864 | 11819 | 11963 |

All-new throughput is 132.39 tok/s for 4096/64 C16 and 206.20 tok/s for
4096/256 C16, down from 190.48 and 287.22 respectively. Long C1/256 drops
from 101.19 to 18.42 tok/s. This is a regression, not successful performance
promotion of the new compiler paths.

| All-new cell | TTFT p50/p95, ms | ITL p50/p95, ms |
| --- | ---: | ---: |
| 4096/64 C1 | 188 / 189 | 52 / 53 |
| 4096/64 C8 | 1382 / 2016 | 56 / 146 |
| 4096/64 C16 | 2325 / 3913 | 61 / 236 |
| 4096/256 C1 | 192 / 194 | 54 / 55 |
| 4096/256 C8 | 1395 / 2030 | 58 / 59 |
| 4096/256 C16 | 2356 / 3966 | 62 / 65 |

Three-trial throughput ranges for 4096/64 C16 and 4096/256 C1 are
132.28–133.14 and 18.408–18.432 tok/s. The regressions greatly exceed that
within-run spread. They are not explained by a one-percent timing fluctuation.

The separate `trace-v4` capture verifies actual launched symbols and grids:

| New path | Observed launch | Calls in short diagnostic capture |
| --- | --- | ---: |
| Packed ingress | grid 128×16 and 1×16, block128; head grid16 confirms h2 ownership | 1148 |
| Forwarded c322 prefill | canonical prefill symbol, grid63×16, block128 | 952 |
| Blockwise c450 decode | suffixed F32 symbol, grids32×8, 1×8 and 8×8, block64 | 700 |

No old decode symbol appears. **An existing partitioned-prefill route remains
active:** 28 partial and 28 merge launches. These account for 6.412 ms in this
diagnostic capture. Thus all four requested changes are exercised, but not
every attention launch has been replaced by the new family. The unchanged
MLP, output/head and residual kernels also remain in the graph.

The trace uses 4096/4 C1 and C16, not the complete timing matrix. Its timings
include profiling overhead and are not substituted into the unprofiled table.
Production clears child environment variables, which prevents profiler
injection into the sealed worker. An isolated **diagnostic-only launcher**
forwards the profiler environment; it is not a production change. The worker
hash is identical before/after:
`8ae04c9a2f246fdbd5ecf652602039371c54b3deff9c105bc4a3e9a0239a3975`.
CUDA artifacts and numerical execution are unchanged. Both captures drain
cleanly. Earlier failed diagnostic captures are preserved, not counted as
successful kernel traces.

Token agreement across the 225 measured sequences:

| Reference | Exact sequences | Matching first token |
| --- | ---: | ---: |
| Prior LunaFlux | 216/225 | 225/225 |
| Historical vLLM | 222/225 | 225/225 |
| Historical SGLang | 216/225 | 225/225 |

All 150 long-input sequences match all references. Differences occur in short
inputs: 66/75 match prior LunaFlux/SGLang, 72/75 match vLLM. The blockwise
path has a different reduction-order contract; these differences are compatible
with changed rounding but this run does not establish their exact cause or
quality impact. Passing bounded numerical checks must not be described as
bitwise model-level equivalence.

Minimum sampled available memory during the timing run was **104,665,804 KiB
(99.8 GiB)**, above the 32 GiB reserve. No OOM occurred; GPU processes were
absent after completion.

## Interpretation

1. **New reachability is not new profitability.** Head/rotary reuse is active
   but its measured h2 finalist loses to the previous row32 ingress. Packing
   two heads does not by itself eliminate projection reloads or replace the
   larger-row schedule's amortization. The paired timing establishes the
   regression; exact instruction-level attribution still needs counters.
2. **The blockwise path is incomplete as a fast long-context replacement.**
   `source_blockwise_decode.mbt` assigns one CTA per row/KV head and traverses
   the entire key history in a loop. It requires a non-pipelined effect plan.
   The bundle accessor intentionally does not return a split module for its
   v8 ABI. This preserves numerical/ABI correctness but removes history-axis
   parallelism. The severe C1 regression is consistent with that source-level
   limitation; these measurements do not isolate its exact percentage from
   the other simultaneous changes.
3. **Fragment forwarding is selected, not proven faster in isolation.** The
   c322 source puts RHS `ldmatrix` and its ordered MMA consumers inside the
   same PTX region. The c322 identity alone cannot distinguish old/new source;
   the current main/wide source hash is
   `e372de2cfda5a17da8b5bc291eee78563fc37f402f291f42b43fd3f7e972b9e2`.
   This experiment does not establish fewer executed SASS moves or an
   independent end-to-end gain from forwarding.
4. **Tuning records do not yet constitute a complete runtime policy.** The
   harness measures actual workload/bucket geometry, but explicitly restricts
   selection to the new families and exports one objective winner per phase.
   It therefore cannot be described as globally optimal shape dispatch.

The remaining architectural performance work is blockwise history partitioning
with its own lawful merge, pipelined acquisition, a combined row/head ingress
schedule search, and workload-specific whole-chain selection. None requires
model-specific scheduler branches or abandoning immutable, functional IR.
These are follow-up conclusions, not changes silently included in this run.

## Reproduction

The offline MoonBit helpers are in `benchmarks/gpu_pipeline/`:
`calibrate_selected_policies.mbtx`, `select_all_new.mbtx`,
`run_selected_runtime.mbtx`, `trace_all_new_runtime.mbtx` and
`finalize_all_new.mbtx`. They keep original measured records, check selected
recipes before packaging, and reject missing new symbols in actual launches.
Benchmark helper fixes retain canonical prefill entry names and copy real
release directories rather than convenience symlinks.

Remote campaign root:
`/home/wlc004s/lunaflux-allnew-fixed-20260930.Y2DWcn9A`.
`calibration/` holds ten cells, `selection-v3/` the measured all-new selection,
and `serving-v6/` the corrected startup run. Earlier attempts are preserved as
failed attempts; they are not counted as completed serving trials.

The verified selection trace is `trace-v4/`. Benchmark helper fixes are
committed in `4495cc44`, `14327978`, `1b5765d6` and `86d85a24`.
The archive excludes generated build caches and duplicated numeric model
weights, retaining recipes, CUDA sources/cubins, records, requests and trace.
Remote archive:
`/home/wlc004s/lunaflux-allnew-fixed-20260930.Y2DWcn9A-export.HGIj1n/all-new-selected.tar.gz`.
Local copy:
`/tmp/lunaflux-allnew-results.TF71vt/all-new-selected.tar.gz`.
SHA-256:
`b376220f80901ad979625c527239dcf6779bd486d492326ad316b503c7ca8dee`.
