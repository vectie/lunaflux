# Why the remaining 2.3× gap exists — real Qwen profiling, 2026-09-05

Follow-up: the production control-poll fix in `37cb1ad` now has a real,
same-session A/B: **118.44 → 182.77 tok/s (+54.31%)**, with control endpoints
retained and identical output tokens. See the
[control progress report](CONTROL_REACTOR_PROGRESS_2026-09-05.md). The diagnosis
and diagnostic ablation below remain historical measurements, not the fixed
runtime's current throughput.

The largest measured problem is **control-plane progress blocking inference
progress**, not GEMM. A bounded diagnostic ablation that omits control-listener
progress, without changing any kernel, raises throughput from **118.27 to
184.20 output tokens/s**. It removes **61.8% of the original latency gap to
vLLM**. This ablation deliberately stops servicing the control listener and is
**not a deployable fix**.

The next largest problem is decode attention: the selected compiler kernel
uses only **8 CTAs on a 36-SM GPU**, with KV splitting inside each CTA rather
than across enough CTAs to occupy the device. MLP and vocabulary projection
are already much closer to the baseline than attention is.

## Matched test, not historical baseline substitution

- Qwen3-0.6B BF16, identical admitted weights and tokenizer, RTX 5060 Ti.
  UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, PCI `00000000:17:00.0`.
- Token vector **[input=59, output=256, concurrency=1]**. The exact input IDs
  come from `lunaflux-vector-requests-4a98f7d-20260904/request-i59-o256.json`.
  Greedy generation, ignore EOS, full 256-token output, loopback streaming.
- One excluded warmup and three individually timed requests per normal
  framework. One framework runs on the GPU at a time; no model downloads,
  environment upgrades, production deployment changes, or concurrent jobs.
- LunaFlux uses the committed deterministic-dot-tree kernel from `2564c31`,
  source snapshot `a1cc6a51376db1d364ce28c79a05464e53008474`, and the deployment
  from the preceding real A/B. Runtime, worker, bridge, weights and the other
  kernels are unchanged. No unrelated dirty-worktree code was uploaded.
- vLLM **0.24.0** and SGLang **0.5.2**, using the installed benchmark conda
  environments and the existing pinned-model launchers. BF16, TP=1, max
  concurrency=32, context limit=40960, prefix/radix reuse disabled, stream
  interval=1. Their usual compiled/graph execution remains enabled.
- Timings below are client `curl time_total`, not TTFT or isolated decode ITL.
  No profiler runs in these normal measurements. All requests produced 256
  token IDs and the terminal stream marker.

| Normal runtime | Trial 1 tok/s | Trial 2 tok/s | Trial 3 tok/s | Median tok/s | Relative to LunaFlux |
| --- | ---: | ---: | ---: | ---: | ---: |
| LunaFlux | 118.581 | 118.268 | 118.251 | **118.268** | 1.000× |
| vLLM | 281.161 | 281.073 | 280.815 | **281.073** | **2.377×** |
| SGLang | 267.032 | 270.428 | 267.366 | **267.366** | **2.261×** |

This establishes the performance gap for this vector. It is not a new full
concurrency/length matrix or a claim of cross-framework model-quality
equivalence. Earlier numerical-schedule/token-equality caveats still apply.

## Real CUDA timeline decomposition

Nsight Systems 2025.5.2 captured the worker's actual CUDA graph nodes. Exclude
startup and warmup; use three complete measured requests for LunaFlux/SGLang.
vLLM's shutdown trace has an incomplete final request, so use **only the two
complete measured requests before it**, not the clipped tail.

The table measures the interval from each request's first GPU kernel to its
last GPU kernel, divided by generated output tokens. GPU busy time is the
**union** of kernel intervals, not a sum that double-counts concurrent work.
These are profiled attribution numbers, distinct from the normal speeds above.

| Profiled quantity, ms/output token | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| First-to-last GPU kernel interval | 8.846 | 3.593 | 3.785 |
| GPU kernel interval union | **5.271** | **3.472** | **3.571** |
| Time outside kernel execution inside that interval | **3.575** | **0.120** | **0.214** |
| Kernel-busy fraction of that interval | 59.6% | 96.7% | 94.3% |

For LunaFlux versus vLLM, 65.8% of the extra profiled interval is outside GPU
kernel execution; 34.2% is additional kernel time. LunaFlux's measured gap from
one token's final greedy kernel to the next token's embedding kernel averages
**3.570 ms**, excluding inter-request gaps (765 transitions). Almost all its
non-kernel interval is therefore **between token steps**, not between graph
nodes within a step.

These gaps alone do not prove CPU computation is expensive: they include
waiting, scheduling, submission and dependencies. The ablation below tests a
specific cause.

## Causal test: control-listener progress

The production path is:

1. `RuntimeInstanceOwner::progress` advances framed ingress/service work.
2. `progress_control_after_ingress` invokes control progress immediately on
   ingress backpressure, regardless of its normal 64-transition fairness cap.
3. `LunaOnlineHttpControlServer::accept_on_reactor` awaits a control connection
   with a **1 ms timeout**, on the same progress chain needed by inference.

Targeted syscall tracing of the unchanged runtime observed **1,762 / 1,767 /
1,763 one-millisecond epoll timeouts** in the three request windows on the
runtime reactor thread, totaling **1.866 / 1.871 / 1.869 seconds** respectively.
Those waits overlap GPU execution and must **not** all be counted as added
latency. Syscall tracing itself reduced throughput to about 110 tok/s, so it
is used only to identify waits.

In a disposable source copy, the only additional ablation was to make
`progress_control_after_ingress` return success without advancing the control
listener. The existing inherited drain path stayed enabled. No kernel,
model, numerical operation, scheduler policy, token transport or benchmark
client was changed. This run had neither Nsight nor syscall tracing active.

| Bounded diagnostic ablation | Trial 1 | Trial 2 | Trial 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| Output tok/s | 184.196 | 183.877 | 184.200 | **184.196** |

- Throughput improvement: **55.7%**.
- Median request time: **2.164567 → 1.389824 seconds**.
- Saved time: **3.026 ms/output token**, or **61.8%** of the original latency
  difference to vLLM on this vector.
- Remaining throughput ratio to vLLM: **1.526×**, not 2.377×.
- All **768/768 output IDs** match the corresponding normal LunaFlux trials.
- The benchmark drained normally. **Never deploy this ablation:** it omits
  control-service functionality instead of implementing nonblocking fairness.

Source locations: `ops/runtime_instance/owner.mbt:199`,
`ops/runtime_instance/control_owner.mbt:84`, and
`service/online_tcp/control_server_progress.mbt:78`.

## Which kernels still cost more?

The following are summed kernel durations per generated token from the
profiled windows, not additive host/API timing. QKV fusion boundaries differ,
so the vLLM QKV row deliberately includes its separate QKNorm/RoPE/cache-write
work. Small prefill contributions are included in the request-level totals.

| Operation, approximate ms/output token | LunaFlux | vLLM | Interpretation |
| --- | ---: | ---: | --- |
| Decode attention | **1.494** | **0.279** | Largest kernel deficit, about **5.4×** |
| QKV + QKNorm/RoPE/KV-write | 0.946 | ≈0.671 | Fusion still has a slower physical schedule |
| MLP gate/up + down | **1.371** | **1.343** | Already close; not the main deficit |
| Attention output projection | 0.311 | 0.322 | Already close |
| Vocabulary/LM-head projection | **0.737** | **0.732** | Essentially equal |
| Residual/RMSNorm families | 0.267 | ≈0.058 | Noticeable small-kernel/fusion deficit |
| Greedy/sample reduction core | 0.141 | ≈0.006 | Large ratio, smaller absolute cost |

vLLM's decode matrix operations here include cuBLAS GEMV kernels; its LM head
uses a CUTLASS WMMA kernel. It is incorrect to describe the whole difference
as “they use Tensor Cores while we do not.” LunaFlux also already uses CUDA
Graphs: measured `cuGraphLaunch` cost is approximately 0.399 ms/output token;
vLLM's graph-launch API duration is approximately 0.450 ms/output token. These
API durations can overlap device work and are **not** added to kernel totals.
Likewise, LunaFlux's approximately 5.28 ms `cuEventSynchronize` duration mostly
waits for kernels; counting it again as CPU overhead would double-count work.

### Attention: block-local splitting is not enough

The actual LunaFlux decode launch was `grid=(1,8,1)`, `block=(256,1,1)`, with
34,076 bytes of dynamic shared memory and 56 registers/thread. With eight
CTAs, it can use at most eight of the GPU's 36 SMs simultaneously. This is a
grid-parallelism bound, **not** a measured achieved-occupancy counter.

The selected compiler schedule is the grouped-query **block-local split**
lowering in `kernels/luna_cuda_attention_tile_source/source_grouped_split.mbt`.
It shares K/V tiles and splits key positions among subgroups, but each block
still walks the complete context tile sequence. Do not confuse this with the
older one-warp direct kernel, or say there is no splitting at all.

| Successive decode-token quarters | 1st | 2nd | 3rd | 4th |
| --- | ---: | ---: | ---: | ---: |
| LunaFlux single-layer attention, μs | 27.33 | 44.98 | 62.50 | 79.94 |

vLLM uses split-KV attention plus a combine kernel, about 6.54 + 3.45 μs/layer
on this window. SGLang uses FlashInfer paged decode plus a merge kernel,
approximately 5.80 + 1.38 μs/layer. The compiler needs a **device-wide KV
partition/merge schedule and an occupancy-aware cost model**, not merely a
larger block-local tile or more generic optimization passes.

## Priority and functional-programming implications

1. Separate control-plane waiting from token progress. Preserve control
   responsiveness and cancellation with independent readiness-driven progress
   and bounded fairness; do not ship the diagnostic omission or replace it
   with an unbounded busy-spin loop.
2. Extend the generic attention schedule/cost model to select cross-workgroup
   KV partitions at low batch size; lower the partition and softmax-summary
   merge through the compiler. Numerical/reduction-order policy stays explicit.
3. Improve QKV ingress scheduling, residual norms and greedy reduction.
   Do not prioritize another broad MLP/LM-head rewrite for this C1 gap.

This is not evidence that functional programming is slow or that the compiler
simply needs more passes. The first issue is **effect scheduling on the runtime
critical path**; the second is an insufficient **physical schedule/cost model**
for an otherwise reusable tile computation. Both fixes can remain general
across model families, with hardware geometry confined to device lowering.

## Reproduction and limitations

Normal, CUDA-profile, syscall and ablation launches, raw SSE responses,
request bodies, binary identities, Nsight reports/SQLite files and diagnostic
patches are archived at:

- Remote: `/dev/shm/lunaflux-gap-final-20260905-r1/results.tar.gz`.
- Local: `/private/tmp/lunaflux-gap-results-20260905-r1.tar.gz`.
- SHA-256: `bd793ee738a63002e3f2c7b7b14f151c669e4a4d9b7bb5d3fe87f32cd6b29c57`.
- Local SQL/CSV output: `/private/tmp/lunaflux-gap-analysis-20260905-r1`;
  orchestration: `/private/tmp/lunaflux-gap-analysis.mbtx`,
  `lunaflux-gap-waits-analysis.mbtx`, `lunaflux-gap-output-check.mbtx`.

Selected kernel windows, in Nsight nanoseconds:

- LunaFlux: `[103543911281,105808629870]`, `[105836503226,108101409812]`,
  `[108132437417,110396301036]` — 768 output tokens.
- vLLM: `[158650097152,159570544729]`, `[159587994992,160506979105]` — 512 tokens.
- SGLang: `[163273046916,164247979697]`, `[164267714330,165237789474]`,
  `[165262433708,166224144063]` — 768 tokens.

Initial outer-process LunaFlux captures did not contain CUDA kernel data:
the worker's startup environment sanitation removed the profiler injection.
The successful CUDA-only capture used a separate startup-only C patch to pass
Nsight's variables through the worker exec. The worker/kernel binaries stayed
unchanged. Profiling reduced LunaFlux throughput to approximately 112.37
tok/s, versus 118.27 without instrumentation. OS-runtime tracing caused a
startup abort; CPU perf events were unavailable (`perf_event_paranoid=4`), so
no host settings were changed and no CPU-cycle attribution is claimed.

All diagnostic variants were stopped, normal instances drained, and the GPU
was free of test workloads afterward. No production fix or deployment is
included in this report. Three short trials and a single token vector identify
this bottleneck, but do not establish tail latency, general workload superiority,
long-context quality parity or production readiness.
