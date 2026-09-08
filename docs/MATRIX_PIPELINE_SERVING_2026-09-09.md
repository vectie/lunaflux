# Matrix pipeline: fresh paired Qwen serving benchmark

## Result

The matrix-family compiler extension improves end-to-end throughput by about
2.5–2.8% on the two longer concurrency-eight cases. Short-prompt differences
are below 1%; this small sample does not establish a statistically significant
short-prompt improvement. The isolated vocabulary gain is not a 2–3× serving
gain.

Control is `ff1b6e6`, already including the MLP-down, gate/up, and attention
terminal-storage optimizations. Optimized production code is `d3d81ca`.
Both release executables and AOT runtimes were freshly built, rather than
reusing an older serving binary. Unrelated working-tree changes were excluded.

Qwen3-0.6B BF16 on RTX 5060 Ti, sm120, CUDA 13.1.115. Same model, token-ID
requests, greedy sampling, ignore-EOS policy, tuning records, native-framed
token bridge, token budget 1024, and prefill chunk 1024 in both arms. No other
GPU workload ran concurrently. Startup/model loading is outside measurement.

Order was control→optimized, then optimized→control. Each cell has one warmup
and two timed trials per order: four timed trials per arm. Throughput below is
the arithmetic mean of trial output-token throughput. TTFT and decode interval
are arithmetic means across timed requests, measured at the streaming client;
they are not isolated GPU kernel durations. All 144 timed request sequences
per arm match their paired counterparts exactly, with the required output
counts (288 timed requests total).

## Complete serving vector

| Input tokens | Output tokens | Concurrency | Control tok/s | Optimized tok/s | Change | Control TTFT ms | Optimized TTFT ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 59 | 256 | 1 | 245.92 | 246.27 | +0.14% | 22.25 | 20.50 |
| 59 | 256 | 8 | 1412.67 | 1423.46 | +0.76% | 46.16 | 42.44 |
| 128 | 128 | 1 | 236.50 | 237.48 | +0.41% | 24.75 | 24.75 |
| 128 | 128 | 8 | 1306.13 | 1314.51 | +0.64% | 75.72 | 73.34 |
| 512 | 64 | 1 | 199.23 | 200.31 | +0.54% | 48.00 | 46.50 |
| 512 | 64 | 8 | 706.75 | 724.25 | +2.48% | 236.72 | 224.38 |
| 1528 | 32 | 1 | 109.41 | 111.50 | +1.92% | 142.75 | 137.75 |
| 1528 | 32 | 8 | 193.61 | 198.95 | +2.76% | 888.13 | 858.50 |

| Input/output/concurrency | Control decode ms/token | Optimized decode ms/token |
| --- | ---: | ---: |
| 59/256/1 | 3.983 | 3.980 |
| 59/256/8 | 5.477 | 5.455 |
| 128/128/1 | 4.039 | 4.028 |
| 128/128/8 | 5.544 | 5.519 |
| 512/64/1 | 4.298 | 4.298 |
| 512/64/8 | 7.566 | 7.479 |
| 1528/32/1 | 4.742 | 4.734 |
| 1528/32/8 | 13.561 | 13.374 |

## Why the serving gain is smaller

- Standalone QKV gains approximately 3–4% on long shapes, but Qwen's fused
  ingress has a separate lowering. This extension does not pipeline that
  fused kernel; no new per-kernel serving trace was collected to assign it
  a measured share of total runtime.
- Long dense output projection gains approximately 5–6%, only one component
  of total model execution.
- Multi-warp vocabulary gains 2.00–3.44× versus its former implementation,
  but the existing tuned single-warp head is already faster at small observed
  row counts. That specialization remains unchanged and selected within its
  supported shape range. The rejected single-warp pipeline is not enabled.
- This increment does not change long-prefill attention, continuous batching,
  or execution-graph policy. The control already includes prior MLP pipeline
  improvements, so this table measures only the additional matrix-family work.

The optimization is a generic immutable ordered-fold schedule, not a Qwen
model special case. CUDA asynchronous copies, shared staging, and WMMA are
private lowering details. Functional semantics and ascending K16 reduction
order are preserved. See [the compiler pipeline report](COMPILER_OPERAND_PIPELINE.md)
for the complete isolated kernel vector, rejected schedules, and sanitizer
results. Native tests: 3,620/3,620; all four CUDA sanitizer tools pass.

Remaining coverage includes the fused QKV/QKNorm/RoPE ingress realization and
better device/shape-specific schedule selection. Neither universal pipeline
coverage nor parity with vLLM/SGLang is established. No fresh external-framework
benchmark was run in this increment.

## Reproduction artifacts

Remote campaign root:
`/run/user/1000/lunaflux-matrix-e2e-20260909-r1`.

- Control source archive SHA-256:
  `fd4f7234ffaf49cd1d96239f46151860d8b0ad916f80779704f632a32908c391`.
- Optimized source archive SHA-256:
  `bcfec0602ee66bbe61f27031763ca678fe69b352ec6e8a716e77b4b9cb0f5480`.
- Local raw serving archive:
  `/private/tmp/lunaflux-matrix-results-20260909-r2/serving-results.tar.gz`.
- Serving archive SHA-256:
  `841686a72346af299c988840f9eefd2cf2bebc58dc63fa0fd112aba02480cfe6`.

The archive retains both source snapshots, fresh build logs, serving launch
arguments, policies, tuning records, runtime bundles, raw streamed events,
token IDs, trial timestamps, and measurements. The analysis script is
`/private/tmp/lunaflux-matrix-e2e-summary.mbtx` locally and
`lunaflux-matrix-e2e-summary-v2.mbtx` under the remote campaign root.
The first optimized packaging attempt included macOS AppleDouble metadata;
its failed build was preserved, then the identical source was repackaged
without that metadata and built in a new directory. A subsequent runner path
mistake was corrected before any optimized serving measurement.

All four serving runs completed successfully and the owned benchmark process
groups were stopped; the GPU had no remaining compute process afterward.
Production deployment was not modified.
