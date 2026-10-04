# Long context benchmark design and results

This campaign tests whether the existing 4K comparisons hide different scaling
at longer context. It compares the qualified physical-domain and bounded-handoff
LunaFlux runtime with the pinned vLLM and SGLang images on one GB10 Spark. It
does not deploy the dirty local checkout, change kernels, or claim 1M serving.

## Workloads and capacity

The primary input vector is 4096, 8192, 16384 and 32512 tokens. Output lengths
are 64 and 256, so every request remains within the model's advertised 32768
total-context envelope. The serving artifacts already cover 40960 context and
2048-token prefill chunks; only the benchmark deployment's input admission and
host token-storage budget are enlarged. A chunk limit is not a context limit.

| Prompt tokens | Concurrent requests | Maximum aggregate KV with 256 output tokens |
|---|---|---|
| 4096 | 1, 2, 4, 8, 16 | 7.4375 GiB |
| 8192 | 1, 2, 4, 8 | 7.21875 GiB |
| 16384 | 1, 2, 4 | 7.109375 GiB |
| 32512 | 1, 2 | 7 GiB |

The Qwen3-0.6B BF16 KV layout uses 112 KiB per token. The unchanged
131072-token physical pool occupies 14 GiB; table entries are live demand,
not the size of that startup allocation. Mixed-length C4 uses the exact input
vector `[4096,8192,16384,32512]`, again with both output lengths. C1 periodic
controls at 4096 and 32512 distinguish prompt-pattern sensitivity from context
length. Primary prompts use deterministic non-periodic sampling from twelve
admitted low-ID tokens, with a separate sequence per request row. They are
synthetic kernel/serving workloads, not long-context comprehension tests.
An additional 32512/64 C1 control samples all 1000 token IDs in `[0,999]`,
with a separate fresh start, one warm-up and three measured repetitions per
engine. It tests a broader vocabulary while retaining the same adapter bound.

Low-ID tokens deliberately fit the existing diagnostic token-ID bridge's
128 KiB JSON-body ceiling. A first high-ID 32K request exceeded that adapter
ceiling and was rejected before inference; its failed capture is retained.
The experiment does not silently enlarge the adapter or shorten token counts.

## Execution and measurements

Engines run sequentially, never concurrently. Each engine starts fresh once,
then each cell gets one excluded warm-up and three measured trials. This
exploratory matrix is not a three-fresh-start counterbalanced release campaign.
All engines use the same weight files, BF16, greedy decoding, disabled prefix
caching, exact token vectors and output counts. Reference image digests and
complete arguments are retained. Only their KV allocation fraction changes
from 0.5 to 0.4 to fit a 64 GiB no-swap process ceiling.
The actual logs record 2048-token chunks for LunaFlux and vLLM; SGLang uses
its native 8192-token chunks with `max_prefill_tokens=16384` and FlashInfer.
Thus this is a native-serving-strategy comparison, not identical scheduler
parameters across all three engines. Reference KV reserves also differ from
LunaFlux's fixed 14 GiB arena; exact live input/output work is matched.

The launcher maintains at least 32 GiB host MemAvailable, records it every
500 ms and stops its own engine if the reserve is breached. LunaFlux's serving
unit is capped at 64 GiB with no swap; its bridge is separately capped at 2 GiB.
Reference containers have 64 GiB memory and combined memory-plus-swap limits.
GPU-idle checks bookend each engine. Startup, request timeout, incomplete SSE,
drain failure and nonempty LunaFlux stderr are failures, not usable trials.

Report per-cell median/min/max completion time, output token throughput,
request TTFT p50/p95, mean post-first-token interval and p95 token-arrival gap.
Retain every input/output token and timestamp vector. Token agreement and
within-engine repeatability are separate from performance: disagreement must
be reported rather than discarded. Client timestamps have millisecond
resolution; arrival intervals include transport, scheduling and overlapping
prefill, and must not be called isolated GPU decode time.

After ordinary timing, targeted GPU traces should distinguish long-history
attention, projections and host gaps. Profiler timings are not substituted for
the unprofiled timing matrix. Long-history microbenchmarks beyond 32K are a
separate kernel-only experiment and cannot prove this model's supported context.
A full 1048576-token request would require 112 GiB BF16 KV alone; it is excluded
on this Spark under the required memory reserve.

## Reproducibility

The benchmark automation is MoonBit script mode:

- `benchmarks/gpu_pipeline/prepare_long_context.mbtx` copies immutable deployment
  inputs into a fresh root, changes input/storage limits, rebinds digests and
  runs the existing release and capacity validators without opening a GPU.
- `benchmarks/gpu_pipeline/long_context_client.mbtx` owns the token/concurrency
  matrix, deterministic inputs, streaming capture and memory checks.
- `benchmarks/gpu_pipeline/run_long_context.mbtx` adapts the existing bounded
  launcher, gates the matrix on a longest-context smoke, and clones reference
  configurations into separately named benchmark containers.
- `benchmarks/gpu_pipeline/summarize_long_context.mbtx` excludes warm-ups,
  validates exact work vectors and reports quantiles and token divergences.
- `benchmarks/gpu_pipeline/finalize_long_context.mbtx` checks terminal engine
  states and sampled memory reserves, retains reference logs, and invokes the
  existing archive helper. Raw SSE streams are included in the archive.

Capacity and prompt determinism, request serialization size, quantile
calculation and token-divergence detection have focused regression tests.

## Results

Both the 32-cell matrix and the additional diversity cell completed on
2026-10-04. All engines returned the exact requested token counts; both
LunaFlux starts drained normally with empty runtime stderr. Reference
containers stopped without OOM, and the GPU was idle afterwards.

### C1 completion time

These are unprofiled median wall times in seconds, including prefill and the
specified output length. They are not isolated kernel times.

| Input / output tokens | LunaFlux | vLLM | SGLang |
|---|---:|---:|---:|
| 4096 / 64 | 0.684 | 0.768 | 0.778 |
| 4096 / 256 | 2.357 | 2.712 | 2.749 |
| 8192 / 64 | 1.018 | 1.075 | 1.076 |
| 8192 / 256 | 3.114 | 3.419 | 3.462 |
| 16384 / 64 | 2.515 | 1.836 | 1.802 |
| 16384 / 256 | 5.418 | 4.967 | 4.997 |
| 32512 / 64 | 7.472 | 4.018 | 3.741 |
| 32512 / 256 | 11.955 | 8.662 | 8.461 |

For 32512/64 C1, output throughput is **8.57 / 15.93 / 17.11 tok/s** in
LunaFlux/vLLM/SGLang order. Completion time is 86.0%/99.7% higher than the
references, not the roughly 10% gap seen in some earlier 4K comparisons.
For 32512/256 C1 it is 38.0%/41.3% higher: longer generation amortizes prefill.

The 32512/64 first-token medians are **5.977 / 2.481 / 2.162 seconds**,
while post-first-token mean-interval medians are **23.41 / 23.95 / 24.75 ms**.
Thus the C1 long-context gap is concentrated before the first token, not in
its subsequent token cadence. At 16K the corresponding first-token times are
1.530 / 0.788 / 0.732 seconds. TTFT includes ingress, scheduling and model
execution; this observation does not isolate attention from the rest of prefill.

### Context/concurrency ladder

Each row has approximately 64K aggregate prompt tokens and 64 output tokens
per request. The number of output tokens is different across rows; this is
not a constant-work scaling test. Exact work matches between engines in each
row. Median completion times are seconds.

| Input tokens per request / C | LunaFlux | vLLM | SGLang |
|---|---:|---:|---:|
| 4096 / 16 | 4.584 | 4.181 | 4.235 |
| 8192 / 8 | 5.478 | 4.533 | 4.587 |
| 16384 / 4 | 8.827 | 5.476 | 5.399 |
| 32512 / 2 | 15.100 | 7.359 | 6.806 |

The 32512/64 C2 completion gap grows to **105.2%/121.9%**. Its per-request
LunaFlux TTFT medians are 5.969 and 12.078 seconds; mean post-first intervals
are 144.29 and 47.78 ms. The early request overlaps the other request's prefill,
so neither interval is an isolated decode-kernel measurement.

Mixed `[4096,8192,16384,32512]` C4 completion times are
**11.246 / 5.896 / 5.590 s** with 64 outputs, and
**21.305 / 12.868 / 12.805 s** with 256 outputs. For the 64-output cell,
per-request TTFT medians expose latency hidden by aggregate throughput:

| Request input tokens | LunaFlux TTFT (s) | vLLM TTFT (s) | SGLang TTFT (s) |
|---|---:|---:|---:|
| 4096 | 2.106 | 0.123 | 0.377 |
| 8192 | 1.956 | 0.456 | 0.373 |
| 16384 | 1.531 | 3.834 | 1.094 |
| 32512 | 7.989 | 2.900 | 3.208 |

These are measured arrival/scheduling outcomes, not a fixed request execution
order. Some reference short requests also finish generation while long
requests are still prefilling; raw token-time vectors retain that overlap.

### Prompt diversity and output agreement

The 32K periodic control produces LunaFlux/vLLM/SGLang completion times
7.449 / 4.017 / 3.714 seconds, close to the 12-token varied case.
The independent 1000-ID, 32512/64 C1 control gives:

| Engine | Completion (s) | TTFT (s) | Mean post-first interval (ms) |
|---|---:|---:|---:|
| LunaFlux | 7.395 | 5.919 | 23.13 |
| vLLM | 3.945 | 2.425 | 23.83 |
| SGLang | 3.685 | 2.115 | 24.52 |

All three output vectors agree and repeat across all three measured trials in
this control. The output is 64 copies of token 198 (newline). This confirms
matched execution work and repeatability for this control, **not realistic
language generation or long-context comprehension**. The timing gap persists
with broader input vocabulary, so the original prompt pattern does not explain it.

The primary matrix does not achieve strict token agreement throughout. Only
6 of 32 cells agree across all engines and trials. Relative to vLLM's first
measured trial, LunaFlux has 142 and SGLang 139 differing request vectors out
of 366 measured vectors per engine. Within-engine differences versus that
engine's own first trial are 63/244 for LunaFlux, 27/244 for vLLM and 11/244
for SGLang. First differing indices and tokens are retained. For example,
the primary 32K C1/64 LunaFlux-vLLM divergence is token 198 versus 271 at
index 9, with the remaining positions matching in that captured pair.

Do not discard those samples or call them numerical-equivalence passes.
Greedy vector differences alone also do not identify a numerical error or
prove it acceptable; logits/error-contract and meaningful-text checks remain
separate work. These are synthetic performance results, not production promotion.

### Memory and saved captures

Minimum sampled host MemAvailable across main and diversity runs was
**99.8 GiB for LunaFlux, 65.8 GiB for vLLM and 63.7 GiB for SGLang**,
all above the 32 GiB reserve. No engine OOM was observed. Full 1M-context
serving was not attempted; this model's 112 GiB BF16 KV demand alone would
violate the reserve on this machine.

The remote root is `/home/wlc004s/lunaflux-long-context-20261004.SqjTBeqG`.
The archive was downloaded without overwrite to
`/tmp/lunaflux-long-context-results-20261004.y3MAmPf6/measurement.tar.gz`,
and its SHA-256 and all 5231 manifest entries verified locally:

`1042418e554621a0716c136f6f4c66693b617681b15c4b8b30dabe29d33ac749`

Extracted `comparison.json` retains all 32 cells, raw timing samples, p50/p95,
per-row metrics and divergences; `diversity-comparison.json` retains the extra
cell. Every request body, output token/timestamp vector and raw SSE stream is
included. The archive also retains capacity bindings, exact runtime/kernel
identities, reference image/configuration/logs, memory samples and scripts.
Unsuccessful preparation attempts remain in distinct remote roots ending
`tI3pkObK` and `PPrpIDqh`; the adapter rejection is retained under `smoke`.
No production kernel or serving-source change was made for this campaign.

## Source-level follow-up

The pinned bundle's measured attention-route records end at context 8192.
For the representative C1/Q2048 case at that boundary, the embedded historical
observations are 1,850,931 ns for route 1 and 8,196,388 ns for route 2 (five
samples each). These are old route measurements, not this campaign's timings.

`kernels/luna_execution_graph_strategy/measured_routes.mbt` leaves unmeasured
buckets at `-1`; `engine/device_step/graph_bucket.mbt` then selects using
partition/wide heuristics. Therefore 16K/32K runs exercise selection outside
the measured context coverage. This is a concrete coverage gap, not proof that
one kernel or one memory dependency causes the entire elapsed-time gap.

The next causal experiment should record actual graph owners and kernel symbols
at Q2048 with prior history 6144, 8192, 14336 and 28672, plus the final Q1792/H30720
chunk. Compare complete baseline/wide/partitioned chains on those exact shapes,
with accuracy and hardware counters, before changing their selection. Do not
fabricate new measured records or extrapolate a historical winner as a result.
