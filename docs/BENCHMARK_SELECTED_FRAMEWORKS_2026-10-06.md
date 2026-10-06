# Selected LunaFlux versus vLLM and SGLang — 2026-10-06

## Result

Fresh end-to-end Qwen3-0.6B BF16 measurements reproduce a workload-dependent
result, not a universal speedup. LunaFlux wins short C1, remains slower on long
prefill, and has its largest measured gap at 4K/C16. Increasing the requested
output from 64 to 256 tokens substantially reduces the long-C1 completion gap.
This remeasures the accepted page-batched AOT bundle; it does not deploy the
dirty working tree or any rejected candidate.

All timings below are medians of six measured batches: two fresh starts per
engine, one excluded warmup and three measured trials per case/start. Engine
order is `[vLLM, LunaFlux, SGLang]`, then its reverse. GPUs are serialized per
host. These are fresh measurements for all three engines, not comparisons to
historical reference numbers.

## End-to-end output throughput

Output tok/s includes prefill and client-observed completion time. C16 and C2
numbers are aggregate throughput. Positive time gaps mean LunaFlux takes longer;
they are not percentages of throughput lost. `32K` here means exactly 32,512
input tokens, not 32,768 plus outputs.

| Input / output / concurrency | LunaFlux tok/s | vLLM tok/s | SGLang tok/s | Luna time versus vLLM / SGLang |
| --- | ---: | ---: | ---: | ---: |
| 128 / 256 / C1 | 149.23 | 120.24 | 121.01 | −19.4% / −18.9% |
| 4,096 / 64 / C1 | 85.85 | 83.06 | 82.10 | −3.2% / −4.4% |
| 4,096 / 64 / C16 | 189.47 | 243.95 | 241.99 | +28.8% / +27.7% |
| 16,384 / 64 / C1 | 32.21 | 35.21 | 35.82 | +9.3% / +11.2% |
| 32,512 / 64 / C1 | 14.67 | 16.21 | 17.21 | +10.5% / +17.4% |
| 32,512 / 64 / C2 | 15.21 | 17.64 | 18.94 | +16.0% / +24.5% |
| 32,512 / 256 / C1 | 29.09 | 29.81 | 30.38 | +2.5% / +4.5% |
| 32,512 / 64 / C1, broad vocabulary | 14.60 | 16.10 | 17.20 | +10.3% / +17.8% |

Exact median completion times, milliseconds:

| Input / output / concurrency | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| 128 / 256 / C1 | 1715.5 | 2129 | 2115.5 |
| 4,096 / 64 / C1 | 745.5 | 770.5 | 779.5 |
| 4,096 / 64 / C16 | 5404.5 | 4197.5 | 4231.5 |
| 16,384 / 64 / C1 | 1987 | 1817.5 | 1786.5 |
| 32,512 / 64 / C1 | 4364 | 3948.5 | 3718.5 |
| 32,512 / 64 / C2 | 8413.5 | 7255.5 | 6756.5 |
| 32,512 / 256 / C1 | 8801 | 8588.5 | 8426 |
| 32,512 / 64 / C1, broad vocabulary | 4384.5 | 3976 | 3722 |

## Where the elapsed-time gap appears

For C1, long first-token time is the remaining measured deficit; post-first-token
time is lower in LunaFlux. At 32K/256, LunaFlux spends 449 ms more before the
first token than vLLM, but 230.5 ms less after it. Against SGLang the corresponding
differences are +752 and −375 ms. Thus output length changes the apparent
whole-request gap without changing the input length.

| C1 input / output | TTFT ms, Luna / vLLM / SGLang | Post-first-token ms, Luna / vLLM / SGLang |
| --- | ---: | ---: |
| 128 / 256 | 16 / 22 / 16 | 1677.5 / 2086 / 2084 |
| 4,096 / 64 | 174.5 / 120 / 117 | 550 / 629.5 / 642.5 |
| 16,384 / 64 | 1017 / 776.5 / 723 | 945.5 / 1014.5 / 1042.5 |
| 32,512 / 64 | 2881 / 2427 / 2147 | 1455.5 / 1500.5 / 1549.5 |
| 32,512 / 256 | 2896 / 2447 / 2144 | 5884.5 / 6115 / 6259.5 |

This does **not** mean concurrency is solved. At 4K/C16 the median of each
batch's mean request TTFT is 1646.22 / 1117.16 / 959.97 ms, and mean-request
post-first-token spans are 3583.25 / 2799.22 / 3244.59 ms. At 32K/C2 they are
4367.25 / 3786.5 / 3211.25 ms and 4030.5 / 3256.25 / 3523.25 ms respectively.
Both spans are worse for LunaFlux. Concurrent spans can contain interleaved
prefill, waiting and decode; they are not pure decode GPU time. Their medians
must not be added to explain maximum batch completion time.

The broad-vocabulary control reproduces the long-C1 gap, so it is not confined
to the original 12-token input pool. This campaign does not collect new hardware
counters or an execution timeline; it localizes client-observed time, not the
instruction-level cause. The next profiling target should include the largest
measured C16 deficit, not assume that only long C1 attention remains.

## Workload and numerical limits

Inputs are deterministic token-ID vectors, bypassing tokenization in all three
adapters. The varied pool contains 12 low-ID tokens; the independent broad
control spans 1,000 token IDs. Every measured request's actual input IDs are
checked against the corresponding vLLM input, including row-specific C16/C2
vectors. Sampling is greedy, prefix caches are disabled, EOS is ignored and
every request must return exactly 64 or 256 tokens. The 128 KiB diagnostic body
ceiling and model's 32,768-token total-context boundary are unchanged.

The campaign completes 192 batches / 576 individual requests including warmups;
144 batches / 432 requests are measured. Raw SSE, output IDs, per-token
timestamps, TTFT and completion samples are preserved. Timing starts after
request-body preparation and includes curl/HTTP/SSE handling and per-request
result persistence. These are client-observed millisecond timings, not kernel
CUDA-event timings or natural-language quality measurements.

Output equality must be considered separately from fixed-work performance:

- Short C1, 4K/C1, 16K/C1 and broad-vocabulary 32K/C1 match both references in
  all six paired complete output vectors.
- Varied 32K/C1, for both output lengths, matches SGLang 6/6 but vLLM 0/6.
- 4K/C16 has two complete LunaFlux trajectories across six measured runs,
  versus one per reference; neither reference matches LunaFlux 6/6.
- 32K/C2 has two LunaFlux and two SGLang trajectories, versus one vLLM
  trajectory. Complete-vector agreement is 4/6 with SGLang and 0/6 with vLLM.

Consequently this is not an admission of deterministic concurrent behavior or
cross-framework quality parity. The pre-existing concurrent trajectory issue
remains visible rather than being hidden by an aggregate tok/s number.

## Selected artifacts and configuration

Matched framework host is Spark .179, NVIDIA GB10, GPU
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, host driver 580.178.04.
LunaFlux uses the accepted page-batched explicit `approx-base2-f32-v1` prefill
path from [the October 5 experiment](BENCHMARK_AKO_PAGE_BATCH_2026-10-05.md).
The later rejected candidates are not selected. No inference source, cubin or
production service is modified in this campaign.

- Worker: `dd4e8e2b5ddfb66cc6c901cf96b5ca14f12f9138841644c4fc2c4b4480928622`.
- Full nine-module bundle: `add504e337b0ad8867342d043ed96cb3a11212f74231984925dc7ca6ecb79724`.
- Changed prefill cubin: `7b42b3531466f5fe22c59e5aac2883db68984907c49a5637ac5f1358ad2eab57`.
- Actual launch: `0e8de3bd397f9bec4f5c99440d77c7c9abbea631c6abdcf8503bc5324b865b68`,
  identical to the accepted October 5 serving campaign. The older PREPARED
  receipt's launch hash is not used as the actual campaign identity.
- vLLM: `0.13.0+faa43dbf.nv26.01`, image
  `73e1b0f3230a9377a3109d20be7f8607963a41a715eb3000bc647f5f73c198ff`;
  runtime reports FlashAttention and 2048-token compile-range/chunk planning.
- SGLang: package `0.5.7+nv26.1`, banner `0.5.7+31b61bbe`, image
  `3a9d399434923689565e1a72f5c3fc409323ce039b65bdbe5c2d6ed14abad3a9`;
  runtime reports FlashInfer and 8192-token prefill chunks.

These are pinned NVIDIA-container builds, not a claim about the latest upstream
releases. Full image IDs, arguments, server logs and startup policies are saved.
LunaFlux keeps its selected 2048-token prefill chunk and 131,072-token KV pool
(14 GiB). References retain memory fraction 0.4. All engines have external
64 GiB/no-swap envelopes; the LunaFlux bridge has a separate 2 GiB envelope.
The campaign controller's 8 GiB cap does not cover sibling engine units or
containers. MemAvailable is sampled every 500 ms with a 32 GiB stop threshold.
The minimum is 63.58 GiB. Both LunaFlux starts drain, exit zero and close their
children; reference containers stop and GPU returns idle.

A CPU-only report-helper build briefly overlaps the first vLLM pass; the second
pass has no such build. Raw samples retain both starts rather than discarding
slower measurements. The largest gaps reproduce across starts; no sub-percent
speedup is claimed from these runs. Initial CPU-only harness/import and helper
invocation failures are preserved in separate attempt directories/user journals.

## Second Spark cross-check and saved results

.178 has no cached vLLM/SGLang images. It runs a separate paired-kernel check,
not a second three-framework serving comparison. GPU is
`GPU-9c3d3cf0-439a-5da2-67e9-20414255879f`; its accepted cubin hash matches .179.
Five alternating paired trials per probe shape confirm page batching gains:

| Query / probe rows / history | Median paired time reduction | Decision |
| --- | ---: | --- |
| 129 / 2 / 127 | 1.10% | Inconclusive |
| 2048 / 4 / 28672 | 10.31% | Improved; all pairs exceed 3% |
| 2048 / 8 / 14336 | 8.23% | Improved; all pairs exceed 3% |
| 2048 / 16 / 7168 | 2.98% | Inconclusive; minimum pair 2.17% |

These are kernel measurements against the pre-page-batched kernel, not serving
gains against another framework. Existing correctness/oracle checks complete;
no new sanitizer or soak is needed for unchanged AOT artifacts.

Remote serving results:
`/home/wlc004s/lunaflux-benchmark-20261006.B5GyxQoX`.
Local [comparison JSON](/tmp/lunaflux-benchmark-20261006.zyACpk8d/full179/comparison.json)
and raw results are in `/tmp/lunaflux-benchmark-20261006.zyACpk8d/full179`.
Serving archive SHA-256:
`dda8ab8c6cdab70c83973af9d4849b39193ec859284b7694ed0719a437bea85f`.

The .178 archive hash is
`a1fd4b11675e7278555b884e8d9ab50fa37c6919914a3930fb1abdeeadd3f2c5`;
raw paired results are in `/tmp/lunaflux-benchmark-20261006.zyACpk8d/crosscheck178`.
Both archive hashes match their downloaded copies, and all 2,156 manifest
members verify locally. Nothing is overwritten or promoted.

Offline helpers are `remeasure_frameworks_20261006.mbtx` and
`summarize_frameworks_20261006.mbtx`. Warning-denied checks pass; their three
tests pass, including ambiguous template seams, median computation and adapter
input normalization. No full dirty-tree inference build is represented as
tested by this campaign.
