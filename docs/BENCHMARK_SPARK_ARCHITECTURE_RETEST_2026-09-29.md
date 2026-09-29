# Spark architecture retest — 2026-09-29

## Source, workload and isolation

This is an end-to-end Qwen3-0.6B BF16 serving retest of committed LunaFlux
`53deab59`, not a kernel-only timing. The source archive SHA-256 is
`9998871009155217b6ff26a07e625ea2e75b7526f55705ac65f1dfa979edeef1`.
An isolated source copy applies the explicit sm120 → sm121 and CUDA
13.1.115 → 13.0.88 portability changes retained in `setup.mbtx`; no schedule
or algorithm is changed by that port. This is still a Spark benchmark port,
not production target qualification.

Hardware: `spark-368c`, GB10, GPU
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, approximately 121 GiB unified RAM.
There were no GPU compute processes or running containers before preparation.
No existing service was stopped. Model source, BF16 numeric conversion and
baseline NVIDIA 26.01 images are reused from the September 22 campaign.
The model weight SHA-256 is
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
These are pinned baselines, not latest-upstream version claims.

The workload repeats the previous synthetic pre-tokenized request pool:
input/output pairs **128/32, 512/64, 4096/64 and 4096/256**, now at concurrency
**1, 2, 4, 8 and 16**. Each cell has one warmup batch and five measured batches.
Greedy generation ignores EOS; prefix reuse is disabled. Output token counts
are checked. Throughput includes prefill and completion. HTTP/token-ID adapters
are the same distinct serving adapters used in the previous comparison.

Only one framework runs on the GPU at a time. Compilation/materialization
finishes before full-route timings. LunaFlux runtime has a 32 GiB systemd limit;
baseline containers have an 80 GiB RAM limit with no additional swap allowance,
and retain their prior 0.5 GPU/static-memory fraction. Driver allocations are
not assumed to be fully charged to cgroups. Available system memory must exceed
32 GiB before each cell, and before/after memory plus GPU temperature, power,
clock and utilization snapshots are retained. No profiler replay is used.

## Route regression discovered by the retest

The `.sh` full-ingress helper explicitly passes `--ingress-evaluate full`.
However, `scripts/build-qwen3-reusable-fused-runtime.mbtx` re-exports the final
metadata-aware bundle without forwarding that selection. The final exporter
therefore chooses `conservative-unmeasured` / `unfused`.

This is not merely removal of QKV fusion: `write_unfused_ingress_bundle` writes
the residual-only bundle and leaves out the optimized attention modules too.
The measured default route is therefore distinct from the prior full-ingress
serving configuration. The initial default-route diagnostic was intentionally
stopped during the medium-input cells; its partial results are retained and
must not be presented as a completed matrix. The runtime drained with exit 0,
drain acknowledged and child closed.

A separate bundle was re-exported from the same compiled artifacts with
`--ingress-evaluate full --query-metadata-v1`. Its manifest reports
`explicit-evaluation` / `full` and includes the optimized attention modules.
The complete three-engine comparison uses this **explicit full route**, not
the unmodified wrapper default. No production source fix is hidden in the run.

## Results

All three complete runs finished 620 measured requests each (plus 124 warmup
requests), with the exact requested output lengths. The table uses the explicit
LunaFlux **full** route described above. Median aggregate output tokens/second:

| Input/output | C | LunaFlux full | vLLM | SGLang |
| --- | ---: | ---: | ---: | ---: |
| 128/32 | 1 | 129.6 | 107.4 | 111.1 |
| 128/32 | 2 | 223.8 | 257.0 | 253.0 |
| 128/32 | 4 | 422.4 | 498.1 | 488.5 |
| 128/32 | 8 | 780.5 | 917.6 | 901.4 |
| 128/32 | 16 | 1296.2 | 1595.0 | 1561.0 |
| 512/64 | 1 | 131.1 | 112.7 | 112.9 |
| 512/64 | 2 | 213.0 | 263.4 | 256.0 |
| 512/64 | 4 | 377.0 | 471.5 | 463.8 |
| 512/64 | 8 | 627.5 | 781.7 | 755.2 |
| 512/64 | 16 | 896.7 | 1159.7 | 1122.8 |
| 4096/64 | 1 | 78.8 | 82.7 | 82.2 |
| 4096/64 | 2 | 102.1 | 142.5 | 140.0 |
| 4096/64 | 4 | 132.4 | 188.8 | 184.3 |
| 4096/64 | 8 | 156.9 | 226.7 | 219.1 |
| 4096/64 | 16 | 164.6 | 243.7 | 242.1 |
| 4096/256 | 1 | 99.3 | 94.3 | 94.2 |
| 4096/256 | 2 | 136.5 | 176.7 | 173.5 |
| 4096/256 | 4 | 198.0 | 246.2 | 241.7 |
| 4096/256 | 8 | 254.1 | 306.9 | 299.6 |
| 4096/256 | 16 | 273.0 | 345.6 | 341.9 |

Selected latency measurements (milliseconds; lower is better):

| Input/output, C | Metric | LunaFlux full | vLLM | SGLang |
| --- | --- | ---: | ---: | ---: |
| 128/32, C16 | Batch completion median | 395 | 321 | 328 |
| 512/64, C16 | Batch completion median | 1142 | 883 | 912 |
| 4096/64, C16 | Batch completion median | 6221 | 4202 | 4229 |
| 4096/64, C16 | TTFT p50 | 2097 | 1128 | 975 |
| 4096/64, C16 | TTFT p95 | 3663 | 2235 | 1719 |
| 4096/64, C16 | Inter-token latency p50 | 45 | 39 | 39 |
| 4096/64, C16 | Inter-token latency p95 | 228 | 84 | 41 |
| 4096/256, C16 | Batch completion median | 15001 | 11852 | 11981 |
| 4096/256, C16 | TTFT p50 | 2099 | 1134 | 976 |
| 4096/256, C16 | Inter-token latency p50 | 45 | 40 | 40 |

At 4096/64 C16, the full route still takes **1.48× vLLM / 1.47× SGLang**
completion time. Its median TTFT is **1.86× / 2.15×**. At 4096/256 C16 the
completion-time ratios narrow to **1.27× / 1.25×**. Short-input C1 remains
faster in total completion time, but this does not imply universally lower TTFT.

Relative to the recorded September 22 LunaFlux run, full-route C16 throughput
increases approximately **18.7% at 128/32**, **16.5% at 512/64**, **12.5% at
4096/64**, and **7.4% at 4096/256**. This is a historical comparison, not a
same-session old/new compiler ablation: the delta spans all intervening source
and route changes, and must not be attributed to the last IR extraction alone.

The default-wrapper diagnostic is materially different:

| Input/output, C1 | Default unfused median ms | Explicit full median ms | Time ratio |
| --- | ---: | ---: | ---: |
| 128/32 | 672 | 247 | 2.72× |
| 512/64 | 3903 | 488 | 8.00× |

Both diagnostic cells contain five measured batches. The unfused run was
stopped during 512/64 C8; no completed long-input default-route result is claimed.
The follow-up source fix is to make the metadata-aware wrapper preserve its
explicit full-ingress evaluation choice, or expose and deliberately forward a
route choice. Until then, the wrapper default is not the fast route in the main
table. This retest does not silently fix or deploy that source change.

### Output agreement and memory

| Pair | First-token agreement | Complete-sequence agreement |
| --- | ---: | ---: |
| LunaFlux full / vLLM | 620/620 | 572/620 |
| LunaFlux full / SGLang | 620/620 | 588/620 |
| vLLM / SGLang | 620/620 | 581/620 |

All **310 long-input sequences** agree exactly across all engines. Short-input
continuations differ in some cases; logits-level numerical equivalence and
natural-language quality are not established by this test.

Minimum observed `MemAvailable` across cell-boundary samples was about
**100.25 GiB LunaFlux**, **54.11 GiB vLLM**, and **53.32 GiB SGLang**.
Swap use did not grow; there was no observed OOM. These are sampled system-memory
readings, not continuous peak GPU-allocation measurements. LunaFlux runtime
stderr was empty and its supervisor verified successful drain and child close.
Both benchmark containers were stopped; the final GPU process list was empty.

The compact [60-cell summary](../benchmarks/spark_e2e_retest_20260929/summary.json)
retains five-batch throughput ranges, batch medians, TTFT and inter-token p50/p95,
and request-latency p95. Raw request outputs and timestamps remain in the archive.

## Retained artifacts

Remote campaign root:
`/home/wlc004s/lunaflux-e2e-53deab59.kZ865GHQ`.
Local preparation and downloaded results:
`/tmp/lunaflux-e2e-remeasure.j4nzQUAl`.

Downloaded archive: `results-53deab59.tar.gz`; local and remote SHA-256 both
`23204c058ef7e472359dd6cd0d11446516d1669a133be2628773f56bb4109a00`.
The archive contains raw measurements, the clean source archive, the explicit
port/setup and benchmark `.mbtx` automation, runtime/bundle identities, full-route
CUBINs, baseline container metadata and logs. Model weights are not duplicated.

The source port, build outputs, generated CUDA, compiled modules, bundle route,
client request bodies, raw SSE, per-token timestamps and memory observations
remain separate from the historical benchmark. An initial symlink model-root
admission failure and an early client request before listener readiness are
preserved and excluded from measurements.

## Interpretation limits

The full-route engine order is LunaFlux → SGLang → vLLM, fixed rather than
rotated across repetitions. Five batches per cell expose run variation but do
not constitute confidence intervals. These are homogeneous synthetic batches,
not mixed-length or natural-language traffic, and this end-to-end retest stops
at 4096 input tokens. Per-token latency is measured at the streaming client
with millisecond resolution; event coalescing can produce zero intervals.
It is not a GPU instruction latency or an independent token-level sample.

No new Nsight counters or CUDA timeline were collected here. A remaining
serving-time gap cannot be attributed to a particular instruction or memory
stall from these metrics alone. The full/unfused diagnostic changes both
ingress and attention routing, so its speed ratio must not be described as
the isolated benefit of QKV fusion.
