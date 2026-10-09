# LLM benchmark restart: availability and source assessment

## Status — 2026-09-26, 10:29 CST observation

No new inference or profiler workload was started. This is a readiness check
and source assessment, **not a new three-engine benchmark**.

- `106.39.18.146:12321`, user `jiaanguo`: two SSH attempts reached TCP but
  closed before SSH key exchange. GPU availability could not be established.
- `106.39.18.146:10422`, user `wlc004s`: password authentication succeeded;
  host `spark-368c`, GB10, driver 580.178.04. It is not an available exclusive
  benchmark device: `glm53-exl3-worker` is running and `VLLM::Worker_TP1`
  owns GPU compute memory. System memory is 121 GiB total, 111 GiB used,
  approximately 9.9 GiB available; swap is 3.3 GiB used.
- GPU utilization was 0% at observation. That does not release the resident
  two-node GLM service or make concurrent profiling a fair comparison.
- `nsys` is on PATH; `ncu` is not. This does not establish that Nsight Compute
  is uninstalled: its installation/container path still needs inspection.

No service was stopped, no runtime code changed, no machine reconfigured.
A maintenance window to pause and subsequently restore the existing GLM
group was requested. A single rank must not be paused as if it were an
unrelated standalone workload.

## Existing measurements: historical, not refreshed

The last recorded three-engine Spark run is
[the September 22 benchmark](BENCHMARK_DGX_SPARK_2026-09-22.md).
It used Qwen3-0.6B BF16 and pinned NVIDIA baseline containers, not current
upstream versions. Representative aggregate output throughput:

| Input/output, concurrency | LunaFlux tok/s | vLLM tok/s | SGLang tok/s |
| --- | ---: | ---: | ---: |
| 128/32, C1 | 129.6 | 113.1 | 113.1 |
| 4096/64, C16 | 146.3 | 249.9 | 242.3 |
| 4096/256, C16 | 254.2 | 355.2 | 343.2 |

The [subsequent Systems trace](BENCHMARK_DGX_SPARK_PROFILE_2026-09-22.md)
reports 6832.18 ms summed kernel duration for long C16. Decode attention,
full ingress, and prefill attention total 5321.07 ms, approximately 77.9%
of that sum. This is a prioritization signal, not an instruction-level cause.
Concurrent intervals must be unioned before deriving GPU idle time.
The prior Compute attempt exhausted unified memory; no successful Spark
hardware-counter comparison is claimed.

## Source findings and discriminating tests

Inspected LunaFlux HEAD: `aad4e7e578b4e2d7f984ce3c9f448f4e90543121`, with an
existing dirty working tree. Sibling checkout revisions are vLLM
`10f9b5d74fb4110adfc310278e029af4faea0565` and SGLang
`a2e88279c28c16945c7c7eacb27f1e066b670a41`. These sibling sources are design
references, **not established source matches for the previously tested images**.
Fresh measurements must identify the installed baseline code as well.

1. **The full ingress fusion constrains the matrix schedule.**
   `kernels/luna_cuda_fused_parallel_aot/qwen_source.mbt` requires a
   16x16x16 projection tile, launches four warps per head, and places the
   reduction loop inside a column-round loop. For head dimension 128,
   four owners of 16 columns need two rounds, each traversing the input
   dimension. Complete row tiles load WMMA operands directly from global
   addresses; the explicit staged producer pipeline of other projection
   paths is not present in this body. This establishes instruction/load
   structure, not the amount reaching DRAM after cache reuse.
   `cmd/lunaflux_qwen3_fused_runtime_bundle_export/ingress_choice.mbt`
   defaults to `FullyFused` when no comparison/evaluation suffix is supplied;
   the ordinary build script invokes the exporter without such a comparison.
   vLLM and SGLang Qwen3 model definitions project QKV separately, then apply
   Q/K norm and RoPE. Their unquantized linear layers dispatch to dedicated
   matrix implementations (including `F.linear`), rather than this head-owned
   full-ingress schedule.
   **Test:** full versus producer-separated ingress at identical row counts,
   numerical policy and KV effects; compare the *whole chain*. Measure loads,
   tensor issue, dependency stalls, barriers and residency, not just launches.

2. **Attention has an optimization-selection problem to measure, not a
   demonstrated absence of a compiler.** The current compiler has immutable
   resource feedback, measured records, a legal schedule frontier, grouped
   ownership and asynchronous transfers. Decode staging uses 8-byte
   `cp.async.ca` transfers and retains addresses/sizes for the following V
   loop; prefill staging uses 16-byte `cp.async.cg` transfers with optional
   page-lookup hoisting. Inspect emitted SASS to learn whether those arrays
   stay in registers and whether address sharing saves instructions or
   extends expensive live ranges. Do not infer a spill from source arrays.
   vLLM's FlashAttention backend passes scheduler metadata and bounded split
   policy; SGLang also has backend-specific attention execution. Actual
   selected backend/kernel must be recorded before comparing these designs.
   **Test:** matched query/history/batch/GQA geometry, selected unsplit and
   split schedules, partial plus merge cost, physical bytes, issue/stalls,
   register allocation and local traffic. Previous broad split and async
   split regressions rule out simply enabling more partitions as a solution.

3. **The Spark port is not Spark tuning.** Export metadata still contains
   `sm_120`; the prior Spark run used a disposable `sm_121`/CUDA 13.0.88 port.
   That demonstrated execution, not optimum GB10 schedules. Pin a fresh
   snapshot and enumerate all portability edits; never silently relabel an
   sm120 artifact. Join selected schedule IDs and actual launch resources to
   each profiled operation.

Potential solutions remain conditional on measurements: retain a pure
shape/device/numerical-policy input, compare legal fusion cuts and tile/
pipeline schedules offline, and lower the winning plan through the device
backend. Do not add Qwen-specific scheduling branches or runtime JIT.

## Planned complete comparison once an exclusive GPU is available

- Same Qwen3-0.6B BF16 source weights, tokenizer, numerical mode and exact
  input token vectors. Reproduce the old synthetic suite, then add diverse
  natural-text and heterogeneous-length batches. Report output-length
  failures and token/logit disagreements separately from speed.
- Input vector 128/512/4096/8192/16384/32768; output vector 64/256;
  concurrency 1/2/4/8/16. Validate context and memory capacity before each
  larger cell; unsupported/OOM cells are explicit results, not omitted wins.
- Cold-prefix comparisons first with prefix reuse disabled; shared-prefix
  warm runs separately. No engine gets a cached-prefix advantage unnoticed.
- Warm to steady state, rotate engine order, at least five measured batches
  per admitted cell. Report batch wall time, aggregate output throughput,
  TTFT and inter-token latency distributions, memory and thermal/power state.
  Use batch-level variability, not correlated tokens as independent samples.
- Systems: mark exact request batches and execution steps; record actual
  prefill/decode work, padding, graph/eager routes, launch/copy/sync intervals,
  and interval-union GPU idle gaps. Separate pure and mixed steps.
- Compute: profile selected hot operation instances separately from latency
  measurements. Collect duration, DRAM/L2/L1 traffic, local loads/stores,
  shared transactions/conflicts, instruction mix, tensor activity, eligible
  warps/issue, long/short scoreboard, barrier and execution-pipeline stalls,
  registers/shared allocation, residency and waves. Query available counters
  for the actual tool/device; do not invent unsupported metric names.
- Begin with small explicit KV budgets and a bounded launch count. Check
  memory headroom and profiler overhead before extending. Kernel replay can
  save and restore accessed memory; application replay avoids that backup
  but requires repeatable launches. Choose replay/cache policy explicitly
  and identically; do not use profiled wall times as serving performance.
  See [NVIDIA profiling guidance](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).
- Join each counter row to framework version, loaded symbol/cubin, actual
  shape and source/SASS. Compare equivalent operation chains; a fused
  QKV+norm+RoPE+KV-write call is not equivalent to baseline QKV alone.

## Architectural tradeoff

LunaFlux's immutable plans, explicit effects, AOT artifacts and bounded hot
path support reproducible compilation and analysis. They do not prove faster
device schedules. Its current full-fusion implementation and target-specific
selection are concrete areas where a generic functional design can still
lower to inefficient machine work. vLLM/SGLang reuse mature matrix/attention
backends and flexible dispatch; their higher-level Python does not imply
slower GPU execution. Conversely, extra intermediate kernels and runtime
layers can cost small-request latency. The old C1 result is consistent with
that opportunity but does not isolate its cause.

Next decision is machine availability, not another speculative compiler
rewrite. No new hardware-counter conclusion is available until the paused
comparison can actually run.
