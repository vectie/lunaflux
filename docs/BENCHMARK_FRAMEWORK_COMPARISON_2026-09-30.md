# LunaFlux vLLM and SGLang matched serving benchmark

The subsequent [priority repair retest](BENCHMARK_POSITIVE_PRIORITY_REPAIRS_2026-09-30.md)
updates the LunaFlux column and confirms the new kernels execute in serving.
The pinned reference column and historical campaign below are preserved.

Fresh end-to-end measurements on September 30 compare the current LunaFlux
working-tree runtime with pinned vLLM and SGLang containers on the same GB10
Spark. LunaFlux is faster at concurrency one in this workload matrix, but
remains slower at concurrency eight and sixteen. For 4096 input and 256 output
tokens at concurrency sixteen, completion takes 13.708 seconds versus 11.855
for vLLM and 12.015 for SGLang: 15.6% and 14.1% longer respectively.

This measures actual serving, not an isolated kernel speedup. The newly tuned
output/down schedules are included in the release and serving bundle. These
results do not establish framework parity, performance on other models or
hardware, or production deployment of the uncommitted working tree.

## Workload and machine

- Qwen3-0.6B, BF16, the same local model directory for every engine.
- GB10, `sm_121`, 48 SMs, approximately 121 GiB system memory.
- Input/output token vectors: `(128,32)`, `(4096,64)`, `(4096,256)`.
- Concurrency vector: `1,8,16`. One warmup and three measured trials per cell.
- Identical integer token inputs, rotated by request row, not retokenized text.
  Inputs repeat a twelve-token pool; this is a controlled throughput matrix,
  not a diverse natural-language or model-quality evaluation.
- Greedy sampling, ignore EOS, exact requested output lengths, streaming.
- Engines run serially, without GPU profiling during timed trials. Reference
  prefix/radix caching is disabled. API/bridge streaming overhead is included;
  vLLM uses completions, SGLang uses its generate endpoint, and LunaFlux uses
  its token-ID bridge over the native framed service.
- LunaFlux uses a 64 GiB service memory ceiling, the containers an 80 GiB
  ceiling; both prohibit additional cgroup swap. A 32 GiB available-memory
  reserve is monitored throughout. Minimum measured available memory was
  99.5 GiB for LunaFlux, 53.6 GiB for vLLM and 54.7 GiB for SGLang.

vLLM image: `nvcr.io/nvidia/vllm:26.01-py3`, digest
`73e1b0f3230a9377a3109d20be7f8607963a41a715eb3000bc647f5f73c198ff`.
SGLang image: `nvcr.io/nvidia/sglang:26.01-py3`, digest
`3a9d399434923689565e1a72f5c3fc409323ce039b65bdbe5c2d6ed14abad3a9`.
Both use BF16, context capacity 40960, maximum 32 running sequences, and
memory utilization/fraction 0.5. Hardware clocks are not locked; per-cell GPU
snapshots and three-trial ranges are retained in the raw archive.

## Output throughput and completion time

Numbers are medians of the three trials. Throughput is generated output tokens
divided by whole-batch completion time, including prefill and transport. Positive
time differences mean LunaFlux is slower; negative differences mean faster.

| Input/output tokens | C | LunaFlux tok/s | vLLM tok/s | SGLang tok/s | LunaFlux time vs vLLM | LunaFlux time vs SGLang |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128/32 | 1 | 131.1 | 107.0 | 110.7 | -18.4% | -15.6% |
| 128/32 | 8 | 752.9 | 898.2 | 892.0 | +19.3% | +18.5% |
| 128/32 | 16 | 1312.8 | 1590.1 | 1546.8 | +21.1% | +17.8% |
| 4096/64 | 1 | 84.1 | 82.6 | 81.8 | -1.8% | -2.7% |
| 4096/64 | 8 | 175.6 | 225.1 | 219.1 | +28.2% | +24.8% |
| 4096/64 | 16 | 200.3 | 243.2 | 241.7 | +21.4% | +20.7% |
| 4096/256 | 1 | 98.7 | 93.7 | 92.8 | -5.1% | -6.0% |
| 4096/256 | 8 | 251.0 | 307.3 | 299.2 | +22.4% | +19.2% |
| 4096/256 | 16 | 298.8 | 345.5 | 340.9 | +15.6% | +14.1% |

The 4096/256 C16 throughput ranges across trials were 297.8–299.5 for LunaFlux,
345.4–345.7 for vLLM and 340.7–341.5 for SGLang. The remaining gap is larger
than the observed run-to-run spread. Three repetitions are not a broad
statistical confidence study.

## First token and streaming interval

These are client-observed, per-request p50 values pooled across measured trials.
They include queueing, chunked prefill and streaming transport, not just GPU
kernel time. C16 does not mean that sixteen rows execute in every GPU step.

| Input/output | C | TTFT Luna/vLLM/SGLang ms | ITL Luna/vLLM/SGLang ms | Batch wall Luna/vLLM/SGLang ms |
| --- | ---: | --- | --- | --- |
| 128/32 | 1 | 15 / 21 / 16 | 7 / 8 / 8 | 244 / 299 / 289 |
| 128/32 | 8 | 44 / 45 / 35 | 9 / 7 / 7 | 340 / 285 / 287 |
| 128/32 | 16 | 56 / 64 / 56 | 10 / 8 / 8 | 390 / 322 / 331 |
| 4096/64 | 1 | 135 / 121 / 115 | 10 / 10 / 10 | 761 / 775 / 782 |
| 4096/64 | 8 | 897 / 628 / 541 | 26 / 22 / 23 | 2916 / 2275 / 2337 |
| 4096/64 | 16 | 1520 / 1139 / 969 | 43 / 39 / 40 | 5112 / 4211 / 4237 |
| 4096/256 | 1 | 138 / 122 / 115 | 10 / 10 / 10 | 2593 / 2733 / 2758 |
| 4096/256 | 8 | 903 / 632 / 544 | 27 / 23 / 23 | 8159 / 6665 / 6845 |
| 4096/256 | 16 | 1530 / 1144 / 973 | 44 / 39 / 40 | 13708 / 11855 / 12015 |

At 4096/256 C16, TTFT p95 was 2663 / 2249 / 1711 ms; ITL p95 was
46 / 59 / 42 ms. LunaFlux's remaining difference is not exclusively prefill:
its median streaming interval also exceeds both references at concurrency.
These service timings identify symptoms, not instruction-level causes. This
run collects no new Nsight counters and must not be described as a fresh
hardware-counter attribution of the residual gap.

A subsequent [fresh timeline and selected-cubin counter investigation](BENCHMARK_FINAL_GAP_INVESTIGATION_2026-09-30.md)
locates the current residual cost, without substituting profiled replay times
for the unprofiled throughput above.

## Selected compiler paths and integration fixes

Offline fold records select the output 64-by-64 async matrix schedule with
producer address reuse, and the down 128-row reuse schedule. The exported
sources match the physically measured record identities:

- Output source `81d4021c4b47ff9a6d0f6c9e083e06ed25a62cdd0bf92e278e8ab6177830f6f7`.
- Gated MLP with down change
  `1c8bf064968f8f35192ac0149a31ce7dbf1fc50c3545ffb0a27aded7964cb5ae`.

Fresh five-sample whole-chain records measured output at 210.001 to 157.315 us
and complete MLP at 893.123 to 830.807 us. Gate/up's experimental segmented
ownership alternative remains unselected because it loses to the ordinary
schedule. The source called `output-wide-final` is a different, slower geometry;
the selected output source is `output-async-square`, not that label.

Release binding previously regenerated untuned launch geometry and therefore
rejected the genuinely selected geometry. Binding now uses the private,
source-bound compiled candidate dimensions, while still rejecting an external
contract with different launch dimensions. The build-time binder loads the
same source-bound fold records as the exporter, and receives the explicit
compiler register ceiling of 255. Default behavior remains 128. Loading and
hashing occur during offline packaging, not token-step execution.

Prefill uses c322 with wide-query candidate 2001; decode uses c452 and newly
measured row/history routing. The existing full ingress policy is
`ingress-g4-a2-s2-t2-f1-r4`. Query-retention and gate alternatives that failed to
win are not forced. The c454 dual-score isolated ablation is not claimed as the
selected serving decoder. Old attention observations no longer describe the
current frontier and are not silently reused as current-source measurements.

Affected binder/projection tests passed 105/105. Packaged ingress, prefill and
decode passed memcheck, racecheck and synccheck. This is a scoped validation:
unrelated dirty tensor-parallel files still produce warning 73 in an all-tree
warning-denied check. Their changes were preserved.

## Output agreement and limitations

All 150 measured long-input sequences match each reference exactly, with exact
requested lengths. Across the complete matrix, exact sequence agreement is
222/225 against vLLM and 216/225 against SGLang. First tokens match 225/225
against both. The short-input differences are real and retained in raw token
outputs; they have not been diagnosed here, so this is not a claim of universal
bitwise agreement. The earlier LunaFlux comparison matches 225/225 sequences.

The earlier selected-ready LunaFlux column in the raw summary is historical,
not a fresh same-session A/B baseline, and must not be presented as an isolated
causal gain from these source changes. End-to-end differences combine new
projection selection, routing, compilation and serving effects. Individual
kernel gains cannot be added to predict throughput.

## Reproduction and retained results

Remote campaign root:
`/home/wlc004s/lunaflux-framework-bench-20260930.sdyit8ET`.
It contains all three raw token-ID clients, server logs, image inspections,
memory samples, selected source inventories, fold records, sanitizers,
timing probes and the full JSON summary. All campaigns ended successfully and
benchmark GPU processes were released before archiving.

The final runtime source was patched after the first upload. Its actual source
snapshot is `benchmark-source.tar.gz`, SHA-256
`110a4a2294205512304ba2061e69d899347cb576de112d6c5f0c0ce667115657`;
the initial upload archive is not substituted for this final identity.
Model identity is
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`;
CUDA 13.0.88 tool identity is
`c3e8741c84713f825ef038de8f80529fbeb55abe9e85066ec89cc7f9aa862857`.

The results archive includes that source snapshot and excludes model weights,
build caches, toolchain copies and initial upload archives. Its SHA-256 is
`2a300c85a2190f6ae3c3559674276c1b8d2e5cd99ab42af89dfdd09c2f8edeec`,
verified again after downloading without overwriting earlier results:
`/private/tmp/lunaflux-framework-bench-20260930.DCkNivYA/results/framework-benchmark.tar.gz`.

Reproduction uses MoonBit automation: `run_selected_runtime.mbtx`,
`remeasure_pinned_baselines.mbtx` and `compare_integrated_serving.mbtx` in
`benchmarks/gpu_pipeline`. The campaign archive contains its exact wrapper and
arguments; a new campaign must use new output directories and the same memory
reserve rather than overwrite these measurements.
