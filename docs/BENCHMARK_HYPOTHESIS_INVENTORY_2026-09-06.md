# Benchmark-first bottleneck inventory — 2026-09-06

Status: diagnosis, not a production optimization or promotion. No production
kernel, compiler, scheduler, deployment, or persistent driver setting changed
in this investigation. This is an initial measured hypothesis inventory, not
an assertion that every possible cause has been tested.

## Setup and scope

- Qwen3-0.6B BF16, RTX 5060 Ti, runtime CUDA UUID
  `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`.
- Local HEAD `694ccce`; selected execution code is back at the `5924c1c`
  baseline after the unsuccessful norm experiments were reverted.
- The disposable profiling runtime forwards Nsight environment variables to
  worker execution. Its selected worker kernels are unchanged. Profiling
  measurements are not fresh uninstrumented serving throughput results.
- Input/output vectors: `(59,256)`, `(128,128)`, `(512,64)`, `(1528,32)`;
  concurrency 1 and 8, two trials. Tables below use trial 1, not a confidence
  interval. Each client first learns a single-request reference, then measures
  the requested concurrency. That reference is not an independent oracle.
- C8 `(59,256)` still reports the known batch/single-reference token mismatch.
  Its trace is useful for diagnosis but is not a correctness-qualified result.
- No competing GPU workload remained after the diagnostic processes drained.

## Measured GPU timelines

Intervals start at the first measured-request kernel and end at the last.
Busy time is the union of kernel intervals, not a claim that all SMs are busy.
Gaps include CPU/driver/synchronization effects; they are not exclusively
scheduler time. These intervals exclude client/network time outside the GPU
window and therefore are not end-to-end latencies or pure decode TPOT.

| Input/output | C | GPU-window ms | Kernel union ms | Gap ms | Gap share |
| --- | ---: | ---: | ---: | ---: | ---: |
| 59/256 | 1 | 1288.797 | 1175.418 | 113.379 | 8.8% |
| 59/256 | 8 | 2987.901 | 2778.585 | 209.316 | 7.0% |
| 128/128 | 1 | 653.821 | 594.818 | 59.003 | 9.0% |
| 128/128 | 8 | 1578.562 | 1472.731 | 105.830 | 6.7% |
| 512/64 | 1 | 408.867 | 378.450 | 30.417 | 7.4% |
| 512/64 | 8 | 1345.310 | 1284.191 | 61.119 | 4.5% |
| 1528/32 | 1 | 401.046 | 384.233 | 16.813 | 4.2% |
| 1528/32 | 8 | 2354.413 | 2304.279 | 50.134 | 2.1% |

Short C1 amortized kernel time per output token: MLP ~1.37 ms, fused QKV
~0.97 ms, decode attention ~0.79 ms, LM head ~0.74 ms, output projection
~0.31 ms, norm ~0.27 ms, greedy ~0.14 ms. There is no single tiny operator
whose removal explains the entire remaining gap.

At 1528/32 C8, prefill attention consumes 569.284 ms (24.2% of the GPU
window), decode attention including split/merge 344.808 ms (14.6%). MLP
consumes 689.324 ms (29.3%), QKV 391.002 ms (16.6%). These are whole-window
totals, including prefill and decode. Host-gap removal alone cannot solve this
workload. A separate prefill-only/TTFT benchmark is still needed.

### Learning-request exclusion

An initial analysis grouped kernels using a 10 ms idle-gap threshold. This
incorrectly merged learning and measured requests in several vectors; those
original `*.timeline.csv` and `*.kernels.csv` files are NOT valid results.
The corrected `*.measured-*.csv` and corresponding SQL exclude the learning
request using its terminal greedy dispatch: `output + ceil(input/256) - 1`
dispatches for this fixed-256-token prefill configuration. C1 counts validate
the cut: 256, 128, 65, and 37 measured greedy dispatches respectively.
This rule is specific to this harness/configuration, not a general trace API.

## QKV single-variable experiments

The selected fused QKV uses 16-row WMMA tiles. On partial tiles it pads shared
input inside the output-column/K loops: width 1024 and head width 128 yield
256 padding-related block barriers per partial tile. An offline variant
materializes the entire padded input once, before those loops. Math ordering
and the QKNorm/RoPE/KV-write epilogue remain unchanged.

The cost changes substantially: static shared memory 8704 -> 40960 bytes,
registers/thread 58 -> 80, no register spills in either compilation. Hence
this is not an isolated barrier-count experiment: resource allocation and
input reuse also change. The variant is deliberately NOT wired into serving.

Graph-replayed, warmed deterministic synthetic inputs; microseconds per
launch. A/A controls varied by about 1%; A/B repeated and reversed execution
order showed the same directional effects. Full QKV outputs and both KV
buffers compared byte-for-byte, with zero mismatches/non-finite active values.
This is differential testing, not independent model correctness or a
sanitizer qualification.

| Active tokens | Baseline fixed grid | Hoisted fixed grid | Observation |
| ---: | ---: | ---: | --- |
| 1 | 27.551 | 21.447 | Resource/launch interaction, not padding removal |
| 8 | 59.670 | 34.106 | Large small-batch signal |
| 15 | 76.377 | 36.989 | Large partial-tile signal |
| 16 | 48.973 | 32.315 | Full-tile gain: padding alone cannot explain it |
| 17 | 64.960 | 32.710 | Partial second tile |
| 32 | 63.469 | 36.376 | Full-tile gain |
| 59 | 96.305 | 63.838 | Prefill-sized partial tile |
| 128 | 114.026 | 121.037 | Regression |
| 256 | 210.786 | 227.202 | Regression |

### Grid-only control, identical CUBIN

Change only grid X from fixed 16 to `ceil(active_tokens/16)`:

| Active tokens | Baseline fixed -> compact us | Hoisted fixed -> compact us |
| ---: | ---: | ---: |
| 1 | 27.327 -> 19.413 | 21.451 -> 19.692 |
| 8 | 59.444 -> 55.764 | 34.119 -> 22.222 |
| 16 | 48.912 -> 33.375 | 32.188 -> 21.569 |
| 32 | 63.576 -> 40.088 | 36.373 -> 35.588 |
| 128 | 114.042 -> 115.067 | 121.022 -> 120.751 |
| 256 | 210.767 -> 210.684 | 227.188 -> 227.128 |

Production trace distinction: short C1 executes 5208 QKV calls at grid X=16
and 1932 at X=1; C8's steady decode already executes at X=1. Therefore the
fixed-to-compact benefit cannot be credited to existing C8 decode. Likewise,
the C1 hoisting benefit largely disappears after compacting the grid. Shape
and launch-policy interactions require separate treatment; gains cannot be
added together or multiplied into serving speedup claims.

## Hardware counters: access and findings

The host sets `RmProfilingAdminOnly=1`. Normal-user Nsight Compute failed with
`ERR_NVGPUCTRPERM`; this restricts performance counters, not inference.
After explicit user authorization, administrator Nsight Compute collected C1
and C8 reports successfully. No persistent permission change, driver reload,
reboot, or clock locking was performed. Reports use 40-pass replay and default
profiling cache behavior; their durations/cache rates are not representative
warm-serving benchmark values.

Baseline QKV, fixed grid:

| Metric | C1 | C8 |
| --- | ---: | ---: |
| DRAM throughput / peak | 51.1% | 18.2% |
| SM throughput / peak | 15.6% | 4.1% |
| Achieved occupancy | 17.6% | 14.2% |
| Scheduler cycles with no eligible warp | 85.2% | 93.2% |
| Long-scoreboard share of warp issue interval | ~60.2% | ~73.4% |
| Local/shared spilling requests | 0 | 0 |

C8 also reports ~1.9-way shared-load bank conflicts. Its source counters flag
excess global sectors while DRAM transferred sectors are nearly fully used;
these are different measurements. Global coalescing is not conclusively
excluded. Main actionable inference: latency hiding, memory dependence,
layout and useful parallelism deserve experiments; peak math throughput is
not currently saturated. The reported dominant stall is long scoreboard,
not a direct confirmation that padding barriers alone dominate. Nsight's
estimated local speedups are not measured speedups and are not reported here.

## Hypothesis disposition and next experiments

| Hypothesis | Current conclusion | Missing discriminating benchmark |
| --- | --- | --- |
| Fixed-capacity launch wastes useful parallelism | Confirmed microbench; present in part of C1 graph route | Same-runtime graph-bucket A/B |
| QKV input rematerialization/resource layout matters | Confirmed differential signal; large tiles regress | Compact-grid resource-matched variants, hot-cache counter A/B |
| QKV register spilling is the cause | Not observed in tested variants | No priority for these shapes |
| Long-input slowness is mostly host scheduling | Contradicted by 2.1% GPU-window gaps at C8 | Separate request/TTFT CPU trace for outside-window costs |
| Attention is the long-input bottleneck | Large measured contributor, not the only one | Isolate query hoisting, KV vector loads, page indexing, split policy |
| C8 decode should use a different split policy | Plausible: it selects direct decode; 433 us/call at long context | Matched direct/split, batch x context sweep |
| Prefill parallelism/tile layout is insufficient | Large measured cost; cause not yet isolated here | Query/K tile and split-softmax sweep plus counters |
| LM head / MLP underutilize small batches | Major measured contributor | Shape-matched kernel and counter controls |
| Greedy reduction is too serial | ~0.14 ms/call measured; structural cause not isolated | Partitioned exact argmax, tie-policy tests |
| Norm register-forwarding alone helps serving | Earlier experiments neutral/negative; reverted | Do not reintroduce without a new discriminating result |
| Batching/graph fragmentation adds work | Mixed launch shapes and extra steps observed | Batch occupancy, chunk size and graph-policy controlled sweep |
| GPU clocks/cache/order explain all QKV gains | A/A and reversed A/B weaken this explanation | More randomized trials and serving A/B remain necessary |

Compiler direction remains pure shape/resource planning and semantics-preserving
transformations, with CUDA launch/layout implementation confined to lowering.
This report does not justify a blanket input-hoisting optimization, reordered
floating-point reductions, or a production change before the remaining A/Bs.

## Artifacts

- Remote matrix: `/dev/shm/lunaflux-matrix-profile-20260906-r1`.
- Local matrix: `/private/tmp/lunaflux-matrix-profile-20260906-r1`.
- Remote QKV experiments: `/dev/shm/lunaflux-ingress-hoist-20260906-r1`.
- Local copy: `/private/tmp/lunaflux-ingress-counter-download-20260906-r1`.
- Counter reports: `ncu-admin-r1.ncu-rep`, `ncu-c8-admin-r1.ncu-rep`.
- Local diagnostic automation: `/private/tmp/lunaflux-matrix-analysis.mbtx`,
  `/private/tmp/lunaflux-ingress-experiment.mbtx`,
  `/private/tmp/lunaflux-ingress-measure.mbtx`,
  `/private/tmp/lunaflux-ingress-controls.mbtx`,
  `/private/tmp/lunaflux-grid-controls.mbtx`.

Temporary artifact paths are diagnostic retention locations, not durable
release storage. No new vLLM/SGLang/llama.cpp comparison was run in this pass.
