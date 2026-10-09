# DGX Spark: LunaFlux / vLLM / SGLang — 2026-09-22

## Scope and method

Same physical GB10, same Qwen3-0.6B BF16 weights, identical pre-tokenized synthetic
requests, greedy generation, EOS ignored, prefix caching disabled. Each engine
ran separately. This measures loopback streaming end-to-end serving, not isolated
kernel time. Output throughput includes prefill and request completion time.

Host: `spark-368c`, ARM64 DGX Spark, GB10 `sm_121`, driver `580.178.04`.
GPU UUID: `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.

| Engine | Tested build |
| --- | --- |
| LunaFlux | Committed snapshot `aad4e7e5`, isolated Spark benchmark port, native release build; CUDA 13.0.88 |
| vLLM | NVIDIA `26.01-py3`, `0.13.0+faa43dbf.nv26.01`, ARM64 |
| SGLang | NVIDIA `26.01-py3`, image label `0.5.7+31b61bbe`, ARM64 |

These are pinned NVIDIA baseline images, **not a claim to test the latest upstream
releases**. The baseline containers use CUDA forward compatibility; vLLM reports
CUDA 13.1 driver 590.48.01 over host kernel driver 580.178.04.

Model source: ModelScope `Qwen/Qwen3-0.6B`.
Source `model.safetensors` SHA-256:
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
LunaFlux converts these same BF16 weights into its numeric artifact.

Workload coordinates: `(128,32)`, `(512,64)`, `(4096,64)`, `(4096,256)`
input/output tokens, each at concurrency 1, 8 and 16. One warmup batch and three
measured batches per coordinate. All requests must return the exact requested
number of token IDs. Report throughput and wall-time medians across three batches;
TTFT is the median across measured requests. The synthetic token pool is repeated
with row-dependent offsets; this is not a natural-language quality evaluation or
a production traffic distribution.

LunaFlux uses its token-ID bridge; vLLM uses streaming completions with returned
token IDs; SGLang uses streaming `/generate` with token IDs. Thus results include
each serving stack, not a claim of identical HTTP implementation overhead.

## Spark portability changes

The committed exporter is pinned to `sm_120` and the fused validator to CUDA
13.1.115. A disposable source copy was adapted to the actual `sm_121` target and
13.0.88 compiler, including matching export, binding and materialization metadata.
No compute algorithm or scheduling rule was deliberately tuned for Spark.
Generated sources were compiled for the real target; identities were not merely
relabelled after compilation. The local production working tree was not patched
with these benchmark-only hardware constants.

This is a **Spark port measurement**, not finished backend parameterization or a
production deployment qualification. Initial failed export/bind attempts were
preserved. The native pre-port suite passed 3,108/3,108 with selected new-toolchain
warnings suppressed; it was not a warning-free strict build.

## Results

Aggregate output throughput, tokens/second (higher is better):

| Input/output | Concurrency | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: | ---: |
| 128/32 | 1 | 129.6 | 113.1 | 113.1 |
| 128/32 | 8 | 629.0 | 927.5 | 911.0 |
| 128/32 | 16 | 1091.7 | 1610.1 | 1585.1 |
| 512/64 | 1 | 130.9 | 116.2 | 114.3 |
| 512/64 | 8 | 520.3 | 798.8 | 755.2 |
| 512/64 | 16 | 769.9 | 1189.3 | 1135.3 |
| 4096/64 | 1 | 75.7 | 83.7 | 82.5 |
| 4096/64 | 8 | 137.6 | 230.4 | 220.4 |
| 4096/64 | 16 | 146.3 | 249.9 | 242.3 |
| 4096/256 | 1 | 98.0 | 95.7 | 93.7 |
| 4096/256 | 8 | 229.4 | 314.7 | 300.1 |
| 4096/256 | 16 | 254.2 | 355.2 | 343.2 |

TTFT, milliseconds (lower is better):

| Input/output | Concurrency | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: | ---: |
| 128/32 | 1 | 16 | 18 | 15 |
| 128/32 | 8 | 63 | 43 | 33 |
| 128/32 | 16 | 109 | 64 | 51 |
| 512/64 | 1 | 35 | 22 | 21 |
| 512/64 | 8 | 211 | 80 | 77 |
| 512/64 | 16 | 337 | 117 | 151 |
| 4096/64 | 1 | 249 | 121 | 118 |
| 4096/64 | 8 | 1429 | 615 | 541 |
| 4096/64 | 16 | 2491 | 1117 | 967 |
| 4096/256 | 1 | 253 | 118 | 117 |
| 4096/256 | 8 | 1444 | 621 | 539 |
| 4096/256 | 16 | 2503 | 1128 | 970 |

Batch completion time, milliseconds (lower is better):

| Input/output | Concurrency | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: | ---: |
| 128/32 | 1 | 247 | 283 | 283 |
| 128/32 | 8 | 407 | 276 | 281 |
| 128/32 | 16 | 469 | 318 | 323 |
| 512/64 | 1 | 489 | 551 | 560 |
| 512/64 | 8 | 984 | 641 | 678 |
| 512/64 | 16 | 1330 | 861 | 902 |
| 4096/64 | 1 | 846 | 765 | 776 |
| 4096/64 | 8 | 3721 | 2222 | 2323 |
| 4096/64 | 16 | 6999 | 4097 | 4227 |
| 4096/256 | 1 | 2612 | 2674 | 2731 |
| 4096/256 | 8 | 8928 | 6508 | 6824 |
| 4096/256 | 16 | 16115 | 11533 | 11934 |

### Interpretation

- Short-input C1 output throughput is approximately 13–15% higher for LunaFlux.
  This does not extend to higher concurrency, nor does it mean lower TTFT in all
  short-input cases.
- At 4096/64 C16, LunaFlux completion time is 1.71x vLLM and 1.66x SGLang.
  TTFT is 2.23x and 2.58x respectively. The pre-first-token gap is substantial.
- At 4096/256 C1, LunaFlux throughput is close to the baselines (2.4% above vLLM,
  4.6% above SGLang in this run). At C16 it remains about 28% / 26% lower.
- These timings do not identify the underlying instruction, memory, scheduling,
  or graph bottleneck. Actual batch sizes, selected kernels, CUDA timelines and
  per-kernel counters on GB10 are needed before selecting another optimization.
- Spark schedules were not autotuned. Do not infer that an sm120-tuned schedule
  is optimal on GB10 simply because it compiles and executes correctly.

### Output checks

Every engine completed 300 measured requests (plus warmups), with exact requested
output lengths. First generated tokens agree across all engines for all 300
measured requests. Complete-sequence agreement:

| Pair | Exact complete sequences |
| --- | ---: |
| LunaFlux / vLLM | 279/300 |
| LunaFlux / SGLang | 291/300 |
| vLLM / SGLang | 282/300 |

All 150 long-input sequences agree exactly across all three engines. Short-input
continuations sometimes diverge. This is compatible with different numerical
implementations, but no logits-level accuracy investigation was performed; it
must not be described as proven full numerical equivalence. All raw outputs are
retained for inspection. Runtime stderr was empty; LunaFlux drained successfully
with `child_exit_code=0`, `drain_acknowledged=1`, `child_closed=1`.

## Reproduction and limitations

Remote workspace: `/home/wlc004s/lunaflux-spark-test.5kAEkw`.
Benchmark driver: `benchmark.mbtx`; preparation: `prepare-spark.mbtx`.
Source commit: `aad4e7e578b4e2d7f984ce3c9f448f4e90543121`.
Downloaded result archive: `/tmp/lunaflux-spark.7bFfuY/benchmark-results.tar.gz`.
Archive SHA-256: `8780e060ea1196c776b3172dcbbe1bd80f5239c73f8544bc6fc2d601ef391684`.
Raw results contain per-request SSE, token IDs, token timestamps, request bodies,
batch wall times, and warmup flags. Startup and image metadata are retained.

The results are a single-host comparison, with fixed engine order and three
repetitions, not confidence intervals or an autotuned maximum for each engine.
Some CPU-only preparation overlapped baseline measurements; no competing model
GPU workload was intentionally run during measured requests. A stricter repeat
should finish all preparation first and rotate engine order. The briefly active
result download during the earliest LunaFlux batches was stopped before the long
input measurements.

The user-authorized `glm53-exl3-worker` container was stopped, not deleted, for the
test. After measurement, LunaFlux was drained, the token-ID bridge terminated,
and both baseline containers stopped. No compute process remained before the
original GLM worker container was started again. Container restart is not an
end-to-end health claim for the two-host GLM service.
It was observed `running`, with restart time `2026-09-22T09:36:20Z`; its log shows
the original headless distributed executor reconnecting to the head node.
