# Typed prefill reads: serving retest and remaining gaps

The typed compiler implementation reaches the actual serving kernel, but it
does **not** fix all performance or numerical questions. A fresh ABBA serving
comparison improves 32K completion time by approximately 1%. The short control
has one losing ABBA half, and C16 output vectors differ. Defaults and production
selection remain unchanged; the measured configuration is not promoted.

## Workload and complete-request results

Spark .179, GB10 sm121, pinned Qwen3-0.6B BF16. Four arms run old/new/new/old,
each with two fresh starts. Each start includes a warmup and three measured
waves per cell: twelve measured waves per side per cell. Input bodies and
output lengths stay fixed. Reported times are unprofiled complete-wave wall
times; throughput is output tokens divided by completion time, not input plus
output throughput. Raw per-request token and timing vectors are retained.

| Input / output / concurrency | Old tok/s | New tok/s | Old completion ms | New completion ms | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 / 256 / 1 | 150.50 | 150.94 | 1701 | 1696 | 0.29% |
| 4096 / 64 / 1 | 93.36 | 93.77 | 685.5 | 682.5 | 0.44% |
| 4096 / 64 / 16 | 224.88 | 225.95 | 4553.5 | 4532 | 0.47% |
| 32512 / 64 / 1 | 16.28 | 16.48 | 3931.5 | 3883 | 1.23% |
| 32512 / 64 / 2 | 16.91 | 17.10 | 7567.5 | 7486 | 1.08% |

The two ABBA halves give 0.996%/1.395% reduction for 32K/C1 and
1.111%/1.161% for 32K/C2. Short/C1 gives −0.237%/+0.382%, so the aggregate
short gain is not a consistent pairwise win. Neither these small differences
nor the kernel-only gains establish general improvement across machines.

| Cell | Median TTFT old → new ms | Mean TPOT median old → new ms |
| --- | ---: | ---: |
| 128/256/C1 | 15 → 15 | 6.518 → 6.510 |
| 4096/64/C1 | 119 → 118 | 8.619 → 8.603 |
| 4096/64/C16 | 1215.5 → 1200 | 48.865 → 48.889 |
| 32512/64/C1 | 2467.5 → 2412 | 22.984 → 22.952 |
| 32512/64/C2 | 3860.5 → 3800 | 55.190 → 54.968 |

The October 6 reference results were **not rerun** in this experiment.
Relative to those historical vLLM/SGLang rates, new C16 completion time remains
approximately 7.6%/7.0% higher; 32K/C2 remains 2.8%/10.2% higher. This is a
historical comparison, not a fresh interleaved three-framework benchmark.

## Propagation verified; calibration is a confound

Only fused module 5 changes. Its baseline/candidate SHA-256 values are
`616ecc0f88c192e59c3134bfc9390a57d59984cb3b31543956b587858e132cdb`
and `301f8fd9db1e0221e4b0b5048d08c03d55ee49f5f50d8ea04e5df33e4582ebf3`.
All eight other module digests match. A fresh serving trace executes
`lunaflux_attention_prefill_tile_compiler_exp2_v1` with the candidate's 236
registers. The candidate bundle's module 5 is its only exporter of that symbol.

The new configuration was recalibrated with complete mixed ordinary/split
chains. Old/new calibration coverage is 170/261 buckets: 170 shared, 91 new
mixed buckets, no missing old buckets, and 21 changed calibration minima among
shared buckets. A later audit of the actual startup selector finds **15 changed
policies**, because an alternative must improve baseline latency by at least
ceil(1%). Neither count alone proves which dispatch executed. Timings use a new
content scope; old measurements are not relabeled.
Consequently the table compares **whole recalibrated configurations**, not an
isolated causal effect of replacing one cubin. Full calibration and selected
decode checks pass, including three sanitizer workloads.

## Numerical scope and exact scalar replay

All cells except C16 retain identical complete output vectors within and
across sides. In C16, 109 of 256 candidate vectors differ from the first
baseline vector (including warmup vectors). Within-side comparisons also vary:
5 of 240 baseline vectors and 13 of 240 candidate vectors differ from their
respective first vectors. This blocks a whole-model bitwise-parity claim;
the attention probe's bitwise equality does not resolve it.

The separate corrected activation capture is replayed with the exact first
layer V weights and matched normalized inputs. All seven changed scalar V
components across samples 1–4 reproduce the solo CUDA result exactly using
32 strided f32 folds, the fixed descending reduction tree and BF16
round-to-nearest-even. Explicit f32 narrowing prevents host FMA contraction.
Some FP64 reference values lie very close to BF16 rounding midpoints.

This establishes scalar reduction-order sensitivity at those components. It
does not replay WMMA, prove all later layers or model quality, or establish
the cause of the C16 serving differences. No slower uniform projection law
or relaxed accuracy threshold is substituted to hide this question.

## Fresh selected execution timeline

One independent profiled start covers 4K/C16 and 32K/C2. These Nsight Systems
times are diagnostics, not the unprofiled table above or fresh Nsight Compute
counters. Candidate prefill appears 896 times in each measured cell, grid
63 × 16, block 128. Grid 63 is a valid grid-stride query loop, not skipped work.

| 32K/C2 measured trace | Time ms |
| --- | ---: |
| GPU activity union | 7275.00 |
| Prefill attention | 3494.85 |
| Decode attention plus merge | 2068.76 |
| Other kernels | 1712.51 |
| No GPU activity within interval | 121.84 |

There are 95 graph steps and 21,850 kernel calls. The earlier final trace's
prefill sum was 3619.35 ms, decode 2071.62 ms, and other kernels 1728.74 ms.
This cross-capture comparison suggests approximately 3.44% less prefill time
and essentially unchanged decode; it is not a paired confidence estimate.
The 4K/C16 trace has 96 graph steps, 503.42 ms prefill, 2103.43 ms decode,
1781.20 ms other kernels, and 120.72 ms without GPU activity. Host gaps of
roughly 2–3% do not explain the entire remaining framework gap.

A trace-analysis bug also inflated the QKV projection grid by including its
rotary-preparation launch. The corrected analyzer excludes that suffix and
has a SQLite fixture regression: actual primary projection grid is 32 × 32,
not 1024 × 1. The latter belongs to rotary preparation. Original reports are
preserved; corrected files end in `primary-grid.analysis.json`.

The [same-binary instruction counters](BENCHMARK_TYPED_PREFILL_READ_PIPELINE_2026-10-07.md)
show fewer fragment rearrangements, but unchanged tensor/shared-read/barrier
counts and additional inserted waiting. They are not new counters from this
serving campaign. The remaining hypothesis is the selected tensor/fragment
dependency chain and mixed-route efficiency, not automatically a shortage of
IR layers or a dominant CPU bubble. A new compiler change requires a matched
selected-workload dependency experiment rather than attributing every wait to
bank conflicts.

## Validation, limits and preserved evidence

The full local native suite passes **4427/4427**, with stripping, two compile
jobs, and legacy/unrelated warning exclusions `-79-20-29-25-92-14`. This is not
an unqualified warning-clean claim. The first full compile ran out of local
disk; only ignored, regenerable native test cache was cleaned before retry.
Source, release artifacts and evidence were preserved. The three new offline
helpers and corrected analyzer pass their focused tests and native
warning-denied checks without exclusions.

GPU jobs were serialized; workers used a 64 GiB process cap, bridge 2 GiB,
no process swap, a 32 GiB MemAvailable reserve and bounded runtime. All eight
unprofiled starts and the trace drain and close with exit zero and empty
runtime stderr. The terminal GPU has no compute process. Production is not
deployed or changed.

Sealed remote root:
`/home/wlc004s/lunaflux-typed-read-serving-20261007.w4d5tFXd`.
Remote `FILES.sha256` verification passes. The non-overwriting local archive
is `/tmp/lunaflux-typed-serving-verified-20261007.SYjFV3hn/evidence.tar.gz`,
SHA-256 `0bc693b5dc3a95a276d3244fc542015f503026c503cf5c09998954a0546f3cec`.
Its local hash verifies; selected metadata is extracted, not the entire
deployment payload. Full local per-file manifest verification is not claimed.

The failed preliminary root ending `08uxNXwZ` remains preserved: its older
probe lacked mixed ordinary-chain support and failed calibration before
serving. The completed run uses probe SHA-256
`821fa177be7e30219dbaf92a58c7337af45529075dca84c9fe162017c7ee1b66`.
Pre-launch identity-guard and coverage-report failures are retained alongside
their fixes; neither is mislabeled as GPU correctness failure.

Automation: `typed_prefill_serving_20261007.mbtx`,
`typed_prefill_serving_report_20261007.mbtx`,
`typed_projection_numeric_replay_20261007.mbtx`, and the corrected
`analyze_remaining_gap_20261006.mbtx` under `benchmarks/gpu_pipeline`.
