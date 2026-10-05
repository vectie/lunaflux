# Explicit exp2 serving propagation — 2026-10-05

## Scope and current status

The [dual-Spark kernel comparison](BENCHMARK_AKO_LONG_EXPONENTIAL_2026-10-05.md)
found a 4.37–5.36% median selected-kernel gain. This follow-up tests whether that
gain survives real serving. The four-start ABBA comparison completed: median
completion time improves 0.84% at 16K/C1, 2.09% at 32K/C1 and 1.95% at 32K/C2.
The kernel gain is not whole-engine gain. A separate completed trace confirms
the approximate symbol executes in serving; it is not a throughput measurement.

`.178` and `.179` are available concurrently. The preceding sweep split paired
cells across both GPUs. This follow-up separates long-history memory checking
on `.178` from route calibration/serving on `.179`, one GPU workload per device.
No extra agents or production defaults are introduced.

The serving experiment reuses the pinned Qwen3-0.6B BF16 worker
`dd4e8e2b5ddfb66cc6c901cf96b5ca14f12f9138841644c4fc2c4b4480928622`.
Only baseline prefill changes to the existing explicit `approx-base2-f32-v1`
symbol `lunaflux_attention_prefill_tile_compiler_exp2_v1`, cubin
`66881bc0055ce4c4340e6001e7746d0abd64a8b7f9a02b741e6b87463f0e2922`.
The separate wide-prefill module remains strict. The other eight module hashes,
symbols and launch geometries are checked unchanged. Approximation is declared
in the bundle, never inferred from a model name or requested implicitly.

The new route scope is
`86ea2cd5831ad63f532332b46b88d7a81d56d8ad74b82ee0379606ceaa6dd60d`.
Six fresh paired route cells completed: three prefill, two mixed, one exact
eight-row-envelope C2 decode chain. They select ordinary prefill/mixed and
partitioned pure C2 decode. Old timings were not relabeled into the new scope.
After binding failures, the completed records are copied unchanged into a new
output directory rather than rerun or overwritten.

## Finite serving comparison

Four fresh starts, ABBA: strict, approximate, approximate, strict. The strict
baseline already contains the previous C2 decode route improvement, so it is
not credited again. Each start uses the unchanged varied token generator,
one warmup and three measured trials per cell:

- 16,384 input tokens, C1, 64 requested output tokens.
- 32,512 input tokens, C1, 64 requested output tokens.
- 32,512 input tokens per request, C2, 64 requested output tokens per request.

Retain completion time, TTFT, TPOT, throughput and every per-request output
token vector, including repeated-start differences. Non-bitwise BF16 probe
acceptance is not model-quality or deterministic-token parity admission.
No fresh vLLM/SGLang result or current cross-framework gap is claimed here.

### Completed unprofiled ABBA results

All requests produced 64 output tokens. Completion and TTFT are milliseconds;
throughput counts output tokens only. Each side has two fresh starts and six
measured trials per cell; warmup vectors are retained separately in the report.

| Input / concurrency | Strict completion | Approximate completion | Reduction | Strict / approximate TTFT | Strict / approximate output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16,384 / C1 | 2,019 | 2,002 | 0.84% | 1,056.5 / 1,040 | 31.70 / 31.97 |
| 32,512 / C1 | 4,606.5 | 4,510 | 2.09% | 3,137 / 3,043.5 | 13.89 / 14.19 |
| 32,512 / C2 | 8,932.5 | 8,758 | 1.95% | 4,743.5 / 4,616.5 | 14.33 / 14.62 |

TPOT strict/approximate: 14.920635/14.920635, 23.047619/23.015873 and
65.928571/65.293651 ms, respectively. Most savings occur before the first
token, consistent with the changed prefill path. C1 token vectors are stable
across repeats and match across sides. C2 has repeat differences on **both**
sides (up to 41 changed positions strict and 45 approximate); cross-side
comparisons have up to 52 changed positions. This remains an unresolved
numerical/trajectory issue, not deterministic or model-quality admission.

Remote: experiment root below, `abba`. Local archive:
`/tmp/lunaflux-ako-exp2-serving-20261005.VLE9f1Iv/abba179.tar.gz`.
SHA-256 `7d21d11d62a8b1f31565cc47b247390b768e2679a0613826042f78da298e1128`;
remote/local hashes match and all 366 manifest entries verify. Raw request
vectors and timings remain in the archive, not just these medians.

### Executed selection

The completed diagnostic trace records the approximate prefill symbol with
grid63×16×1, block128, 234 registers/thread: 672 calls in two C1 waves and
1,400 calls in two C2 waves. The unchanged strict wide-prefill module also
executes (224/392 calls); this is not a claim that every prefill call is exp2.
Partitioned decode remains selected. Trace timings are not substituted for
unprofiled ABBA timings.

Remote trace:
`/home/wlc004s/lunaflux-ako-exp2-selected-trace-v2-20261005.zlWwfeeZ`.
Archive SHA-256:
`d083d0109ae08c7f97e76f06dee046c3d247a809209df6c239435a0b3e1ca137`.
Local archive:
`/tmp/lunaflux-ako-exp2-serving-20261005.VLE9f1Iv/selected-trace179.tar.gz`;
remote/local hashes match and all 105 manifest entries verify.
The first trace at `lunaflux-ako-exp2-selected-trace-20261005.yB4UoqjQ`
preserves a profiler-injection failure: the normal parent's sanitized spawn
environment strips CUPTI injection, so no CUDA kernel table was collected.
The replacement reuses the existing private diagnostic parent; production
spawn isolation, worker and kernel artifacts were not changed.

### Parallel larger-batch probes on `.178`

Five paired samples per cell, query2048, runtime bucket2048/rows32. These are
kernel probes, not serving C4/C8/C16 measurements. Histories differ to remain
within the pinned aggregate KV capacity; do not compare them as equal-history
batch scaling.

| Active rows / history | Strict / approximate median (µs) | Median gain | Worst paired gain | Decision |
| --- | ---: | ---: | ---: | --- |
| 4 / 28,672 | 7,589.45 / 7,304.51 | 3.75% | 2.79% | Inconclusive against conservative threshold |
| 8 / 14,336 | 4,051.42 / 3,871.96 | 4.43% | 2.15% | Inconclusive |
| 16 / 7,168 | 3,017.32 / 2,989.68 | 0.92% | −0.74% | Inconclusive |

The original rows8/history28672 case was rejected before allocation: it
requires 231,424 aggregate KV tokens versus the pinned 131,072 capacity.
This is an artifact capacity bound, not physical-memory exhaustion. The
replacement cells each need 116,736 aggregate tokens. No capacity was raised
or sealed artifact overwritten. R8/R16 maxabs is 0.000488281; FP64 oracle
errors are 0.000419239/0.000389352, below ceiling0.003.

Local archives under `/tmp/lunaflux-ako-exp2-serving-20261005.VLE9f1Iv`:
`larger-batch-failed178.tar.gz` (preserved partial/failure), SHA-256
`ab7d74191a32a9d07d6a8950d419867523e4c320cb77cd5a733812486b20ac12`;
`larger-batch-bounded178.tar.gz` (completed), SHA-256
`c1b42b82502abc6d23b9bb90ae0475d6bae694e74e54af51656c2264493416c1`.
Both download hashes match; all 27 completed manifest entries verify.

Serving children retain MemoryMax64G, MemorySwapMax0 and continuous 32GiB
MemAvailable reserve monitoring. The outer automation's smaller CPU limit is
not incorrectly claimed to bound sibling serving units. Route calibration
uses MemoryMax16G, zero swap and a 600-second limit; its observed peak was
854.5MiB. Bundle preparation uses MemoryMax8G, zero swap and a 300-second limit.

## Completed long-history memory gate

`.178`: Q2048, active rows2, history28672; actual runtime bucket2048/rows32,
grid63×16×1. Compute Sanitizer memcheck reports zero errors. The same probe
reports maxabs0.000488281 and FP64 oracle maxabs0.000377474, below ceiling0.003.
This is a long-history memory/numerical probe, not full serving/model quality.

Remote output:
`/home/wlc003s/lunaflux-ako-long-exp-20261005.YDjCs0sP/long-memcheck`.
Downloaded archive:
`/tmp/lunaflux-ako-exp2-serving-20261005.VLE9f1Iv/long-memcheck178.tar.gz`.
SHA-256:
`876a3671fb4b9c88286a897cd97849b1a7ce11aafb41fb39e89656ba121ba6e7`.
Remote/local hashes match and all seven manifest entries verify.

## Preserved preparation failures

Experiment root:
`/home/wlc004s/lunaflux-ako-exp2-serving-20261005.t6QFbUwT`.

1. `facade`: export succeeds, nested materialization cannot find `moon` under
   systemd's noninteractive PATH. The offline adapter now pins the frozen
   toolchain PATH and source working directory, as existing campaign helpers do.
   `facade-v2` completes preparation, peak1.2GiB, no swap.
2. `facade-v2/routes`: route rebind/materialization/preflight succeed, but
   capacity receipt creation rejects the facade's symlinked runtime path as
   noncanonical. The route helper now resolves the frozen source path before
   rendering executable identities. No validation is weakened. The failed
   directory remains preserved; the replacement binding uses `routes-bound-v2`.

Changes are offline MoonBit automation only. Native warning-denied script
checks and affected tests pass. `moon info` does not accept these standalone
scripts as packages in this toolchain; no production public API is changed.
The compiler's semantic/numeric → schedule/selection → ownership/effects →
device-lowering boundaries remain unchanged. AKO determines the paired finite
budget, explicit approximation and end-to-end propagation requirement.
