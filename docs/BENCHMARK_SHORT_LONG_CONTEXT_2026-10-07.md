# Short and long context serving comparison

Fresh Qwen3-0.6B BF16 measurements on Spark .179 show a workload-dependent
gap, not a uniform LunaFlux slowdown. At 128-token input and concurrency one,
LunaFlux completes approximately 20% sooner than both references. At 4K/C16
it takes 8.39%/7.40% longer than vLLM/SGLang. The largest measured gap is
8K/C8: 52.35%/48.95% longer. At 32K/C1 it is approximately level with vLLM;
SGLang has lower first-token latency, while LunaFlux has faster subsequent
token delivery. A 256-token output reverses that total-time comparison.

These are fixed-work throughput comparisons. Several complete generated
token vectors differ across engines, and some batched vectors vary within an
engine. The measurements do not establish numerical or language-quality parity.

## Workload and timing boundary

Eleven cells run on each engine in two fresh starts. Each start has one warmup
and three measured waves per cell: six measured waves per engine/cell,
198 measured waves total, and 264 waves including warmups. The measured waves
contain 882 requests and 63,360 output tokens. Start orders are
vLLM → LunaFlux → SGLang, then SGLang → LunaFlux → vLLM. GPU work is serialized.

All engines use the same pinned model weights and exactly matching input
token-ID vectors, greedy sampling, no early EOS, and fixed output lengths.
Prefix caching is disabled in the references; the LunaFlux route is uncached.
Inputs are deterministic, row-specific pseudorandom selections from twelve
low token IDs. The additional 32K diverse control selects from 1,000 IDs.
Neither distribution is a natural-language corpus. Tokenization is excluded;
client submission, adapter/serving, streaming and response completion are included.

Throughput is aggregate **output tokens divided by median complete-wave wall
time**, including prefill. It is not input-plus-output throughput or standalone
decode speed. Positive completion deltas mean LunaFlux takes longer. Context
length is per request; C denotes concurrent requests. For example, 8K/C8 is
an eight-element input vector of 8,192 tokens, not one 65,536-token context.

## Complete request results

| Input / output / C | LunaFlux tok/s | vLLM tok/s | SGLang tok/s | LunaFlux completion delta vs vLLM / SGLang |
| --- | ---: | ---: | ---: | ---: |
| 128 / 64 / 1 | 143.50 | 115.00 | 116.15 | −19.86% / −19.06% |
| 128 / 256 / 1 | 150.46 | 120.67 | 119.54 | −19.80% / −20.55% |
| 128 / 64 / 16 | 1592.53 | 1785.53 | 1741.50 | +12.12% / +9.35% |
| 4096 / 64 / 1 | 92.89 | 83.12 | 81.42 | −10.52% / −12.34% |
| 4096 / 64 / 16 | 224.44 | 243.26 | 241.05 | +8.39% / +7.40% |
| 8192 / 64 / 8 | 74.46 | 113.44 | 110.91 | +52.35% / +48.95% |
| 16384 / 64 / 1 | 36.17 | 35.13 | 35.54 | −2.88% / −1.75% |
| 32512 / 64 / 1 | 16.16 | 16.09 | 17.12 | −0.45% / +5.97% |
| 32512 / 64 / 2 | 16.79 | 17.59 | 18.82 | +4.78% / +12.12% |
| 32512 / 256 / 1 | 30.58 | 29.74 | 30.28 | −2.72% / −0.98% |
| 32512 diverse / 64 / 1 | 16.16 | 16.05 | 17.15 | −0.65% / +6.15% |

Median complete-wave times, in LunaFlux/vLLM/SGLang order, are 446/556.5/551 ms
for 128/64/C1; 4562.5/4209.5/4248 ms for 4K/C16;
6876.5/4513.5/4616.5 ms for 8K/C8; and 7624.5/7276.5/6800.5 ms for 32K/C2.
The 8K/C8 measured ranges are 6862–6890, 4498–4539 and 4572–4653 ms:
the large gap appears across both starts rather than coming from one slow wave.
The 32K/C1 difference against vLLM is below 1%; no significance or universal
winner is inferred from six waves.

## First token and token delivery

TTFT p50/p95 pools individual requests across the six measured waves.
Mean TPOT is each request's first-to-last token interval divided by
output_count−1; the table gives the median of those request means. Quantiles
use linear interpolation, R7. All times are milliseconds.

| Input / output / C | LunaFlux TTFT p50 / p95 | vLLM TTFT p50 / p95 | SGLang TTFT p50 / p95 | Mean TPOT p50 LunaFlux / vLLM / SGLang |
| --- | ---: | ---: | ---: | ---: |
| 128 / 64 / 1 | 15 / 16 | 21 / 21 | 15 / 16 | 6.50 / 8.18 / 8.18 |
| 128 / 256 / 1 | 15 / 16.75 | 21.5 / 23.75 | 16 / 16.75 | 6.53 / 8.15 / 8.25 |
| 128 / 64 / 16 | 52 / 69 | 65 / 82 | 56 / 62 | 9.00 / 7.85 / 8.03 |
| 4096 / 64 / 1 | 119.5 / 121 | 120 / 120.75 | 116.5 / 119.75 | 8.71 / 10.02 / 10.31 |
| 4096 / 64 / 16 | 1227 / 2467.5 | 1074 / 2223.75 | 981 / 1701.75 | 48.99 / 45.87 / 51.52 |
| 8192 / 64 / 8 | 2523 / 4781.65 | 1376 / 2567.85 | 1188 / 2113.30 | 65.19 / 46.11 / 54.07 |
| 16384 / 64 / 1 | 804 / 808.75 | 786.5 / 792.25 | 735.5 / 737 | 14.98 / 16.13 / 16.58 |
| 32512 / 64 / 1 | 2488.5 / 2495.75 | 2462 / 2468 | 2163.5 / 2167.5 | 23.03 / 23.80 / 24.59 |
| 32512 / 64 / 2 | 3896 / 5322.5 | 3783 / 5151.45 | 3241.5 / 4334.35 | 55.52 / 51.90 / 55.92 |
| 32512 / 256 / 1 | 2487 / 2490 | 2459 / 2471 | 2161 / 2182 | 22.99 / 24.00 / 24.59 |
| 32512 diverse / 64 / 1 | 2488 / 2495.5 | 2463 / 2468.75 | 2161.5 / 2168 | 23.04 / 23.87 / 24.61 |

At 8K/C8, both TTFT and subsequent delivery are slower. Pooled stream-interval
p95 is 213.85 ms for LunaFlux, 97 ms for vLLM and 41 ms for SGLang; p50 is
40/38/39 ms. A median inter-token interval alone would conceal this tail.
At 32K/C2 the corresponding p95 is 197.5/188.25/40 ms.
These are client SSE arrival times, including batching and stream buffering,
not measured GPU kernel durations or proof of a particular synchronization bug.

The 32K diverse control produces almost unchanged timing relative to the
twelve-ID input. That rules out a large distribution effect for this one
C1 control, not sensitivity to all real prompts. The 64-versus-256 output
control shows how decode speed can offset SGLang's shorter first-token delay.
Neither control removes the batched 8K gap.

## Output agreement and repeatability

Agreement counts below compare each entire wave's ordered generated token
vectors against LunaFlux: six paired waves per reference. One unequal request
makes the wave unequal. Within-engine counts compare individual request
vectors with that row's first measured vector across both starts; denominators
exclude that first vector.

| Input / output / C | Whole-wave agreement vLLM / SGLang | Within-engine changed vectors LunaFlux / vLLM / SGLang |
| --- | ---: | ---: |
| 128 / 64 / 1 | 6/6 / 6/6 | 0/5 / 0/5 / 0/5 |
| 128 / 256 / 1 | 6/6 / 6/6 | 0/5 / 0/5 / 0/5 |
| 128 / 64 / 16 | 0/6 / 0/6 | 1/80 / 1/80 / 5/80 |
| 4096 / 64 / 1 | 6/6 / 6/6 | 0/5 / 0/5 / 0/5 |
| 4096 / 64 / 16 | 0/6 / 0/6 | 2/80 / 0/80 / 0/80 |
| 8192 / 64 / 8 | 0/6 / 0/6 | 13/40 / 4/40 / 0/40 |
| 16384 / 64 / 1 | 6/6 / 6/6 | 0/5 / 0/5 / 0/5 |
| 32512 / 64 / 1 | 0/6 / 6/6 | 0/5 / 0/5 / 0/5 |
| 32512 / 64 / 2 | 0/6 / 0/6 | 0/10 / 0/10 / 4/10 |
| 32512 / 256 / 1 | 0/6 / 6/6 | 0/5 / 0/5 / 0/5 |
| 32512 diverse / 64 / 1 | 6/6 / 6/6 | 0/5 / 0/5 / 0/5 |

Identical inputs and counts pass for every measured request. The differing
vectors remain a correctness/quality question; timing cannot explain them.
This campaign neither relaxes accuracy requirements nor attributes all
differences to floating-point rounding without replay evidence.

## Runtime and resource limits

The device is Spark .179's GB10, sm121, 48 SMs, UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`. Model weights SHA-256:
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
LunaFlux uses the current **selected frozen serving configuration**, not a
dirty-tree rebuild or the later unpromoted typed-prefill/synchronization
prototypes. Runtime SHA-256:
`619140a64d70d77d9f6494de4e1e8103b6e1565750fc2015cbf687c0e534f8e8`;
worker SHA-256:
`7da40e6f9f8c07f4152baa1796871651057ff4c9579b9c97fe1ebfa2d3f1ac70`;
selected bundle SHA-256:
`651b0d05ad38074bdd1a63ab608f5add41b5c81c16c9e8fade5a9c228a1c9858`.
The selected bundle and execution manifest are copied into the evidence.

Pinned reference image IDs are
`sha256:73e1b0f3230a9377a3109d20be7f8607963a41a715eb3000bc647f5f73c198ff`
for vLLM and
`sha256:3a9d399434923689565e1a72f5c3fc409323ce039b65bdbe5c2d6ed14abad3a9`
for SGLang. New containers reuse these exact images and recorded commands.
Both use BF16, at most 32 running sequences, a 40,960-token context limit,
and 0.4 static/GPU memory fractions. LunaFlux has a 32,768-token per-request
context limit and 131,072-token aggregate KV capacity. These memory allocation
policies are recorded, not presented as identical. Largest input is 32,512;
with 256 output tokens it fits LunaFlux's context limit exactly. No 1M context
support or larger-model performance follows from these results.

Serving processes/containers are capped at 64 GiB, bridge at 2 GiB and
controller at 8 GiB, with zero additional swap. A 500 ms memory monitor enforces
a 32 GiB host MemAvailable reserve. Minimum observed available memory across
each engine's two starts is 99.68 GiB for LunaFlux, 65.75 GiB for vLLM and
63.71 GiB for SGLang. The reserve is never breached; these host values are not
per-engine allocation measurements. Both LunaFlux starts drain, close their
child, exit zero and have empty runtime stderr. The controller exits zero and
the terminal GPU has no compute process. Production is unchanged.

## Validation and evidence

The new serving and metrics helpers pass warning-denied checks and focused
native tests, one test each. The reused comparison helper passes two focused
native tests. No production kernel/native ABI changes occur in this campaign.

Remote evidence root:
`/home/wlc004s/lunaflux-short-long-20261007.nCHIeH1i`.
Its terminal result is completed. Remote file-manifest verification passes.
Archive SHA-256:
`953136fe8f3c42c4876cef2b5fefcd81d532ce7d49334fab7483445a5de64397`.
The non-overwriting local copy is
`/tmp/lunaflux-short-long-verified-20261007.4sw1arYy/lunaflux-short-long-20261007.nCHIeH1i.measurement.tar.gz`.
The local archive hash matches, and all 4,039 manifest entries verify after
full extraction into that directory's `extracted` subdirectory.

Raw request bodies, SSE, complete token/timestamp vectors, memory samples,
start commands and outcomes remain preserved. `comparison.json` contains six
wave samples per engine/cell; `timing-vectors-summary.json` contains per-request
quantiles, pooled stream intervals and repeatability counts. Automation is
`benchmarks/gpu_pipeline/short_long_serving_20261007.mbtx`,
`short_long_metrics_20261007.mbtx` and the reused
`summarize_frameworks_20261006.mbtx`.

## What this comparison changes

The next diagnostic workload should be **8K/64/C8**, with 4K/C16 and 32K/C2
as controls. Its TTFT and token-delivery tail both worsen even though the
aggregate prompt count matches 4K/C16. Selected-workload traces and counters
are needed to separate scheduling/chunk geometry, kernel execution and
submission waits; this timing-only run does not identify their causal shares.
Short C1 is already faster in these cells, so a universal claim that every
path is 20% slower is not supported. These results also cannot be labeled a
gain from the unpromoted prototypes, which this campaign does not execute.
