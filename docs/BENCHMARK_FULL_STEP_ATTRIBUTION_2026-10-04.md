# Current full-serving-step measurement

The remaining gap is not one attention-pipeline problem. Fresh full-window
traces place the largest classified excess against vLLM in projection/post-ops,
followed by attention and intervals with no observed GPU activity. Exact-work
matches also expose slower mixed-step projection and attention chains.
No production kernel, selector, numerical contract or deployment was changed.

## Scope and timing

DGX Spark GB10, sm121, 48 SMs, CUDA 13.0.88;
UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`. Qwen3-0.6B,
16 requests, each with exactly 4096 explicit input tokens and 256 output tokens.
GPU runs were serialized. All 16 output token-ID sequences agree across engines.

The unchanged current serving route was copied into an isolated diagnostic
runtime. CPU-only query/history markers were added to it and the pinned
reference runners. This is Nsight Systems timing/activity measurement, not a
new Nsight Compute instruction-counter capture. Profiler durations are not
ordinary throughput results or proof of a particular instruction bottleneck.

| Profiled client completion | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| Milliseconds | 12,794 | 11,932 | 11,990 |

The separately recorded client epoch window includes small recording-edge
overheads: 12,795.080 / 11,934.425 / 11,991.019 ms. Its exact differences are
860.654 / 804.060 ms. These are the windows used for additive accounting below.

For ordinary performance, the preceding three-round
[serving campaign](BENCHMARK_SPARK_CLOSURE_2026-10-04.md) remains authoritative:
4096/256 C16 is 319.53 / 346.09 / 340.79 output tok/s, with LunaFlux completion
time 8.3% / 6.7% higher. Do not substitute profiled durations into that matrix.

## Additive window accounting

Each row is exclusive wall time. Overlap between different chains is its own
row; individual kernel-duration sums are retained separately and are not
added to this table. Unclassified activity remains visible.

| Exclusive activity, ms | LunaFlux | vLLM | SGLang | Luna−vLLM | Luna−SGLang |
| --- | ---: | ---: | ---: | ---: | ---: |
| Projection and post-ops | 2700.035 | 2215.684 | 2334.602 | +484.351 | +365.433 |
| Attention, all phases | 9116.169 | 8904.425 | 8912.674 | +211.744 | +203.495 |
| No observed GPU activity | 408.315 | 230.224 | 55.214 | +178.091 | +353.101 |
| Normalization and embedding | 172.142 | 108.838 | 148.942 | +63.304 | +23.200 |
| Head and sampling | 394.156 | 394.628 | 365.530 | −0.471 | +28.626 |
| Transfer | 1.253 | 2.319 | 1.841 | −1.066 | −0.588 |
| Unmapped | 0 | 77.018 | 104.016 | −77.018 | −104.016 |
| Cross-chain overlap | 3.009 | 1.289 | 68.199 | +1.720 | −65.190 |
| Total window | 12795.080 | 11934.425 | 11991.019 | +860.654 | +804.060 |

The projection/post-op row groups GEMM and its associated activation,
normalization/rotary/KV-write operations, not GEMM alone. Generic reference
GEMV and some reduction names cannot be assigned safely to a model operation
from their launch grid. They remain unmapped. Consequently the table localizes
classified activity but is not a complete operation-level causal attribution.
In particular, it does not prove sampling is solved or every projection is slow.

## Actual work, not inferred launch labels

Query/history markers establish the following logical-work ledgers. A
one-query row is classified as such, not presumed to be decode from its grid.

| Ledger | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| Serving steps | 288 | 288 | 265 |
| Total query tokens | 69,616 | 69,616 | 69,632 |
| Query tokens in multi-query rows | 65,536 | 65,536 | 65,536 |
| Query tokens in one-query rows | 4,080 | 4,080 | 4,096 |
| Causal QK pairs per query head | 151,484,416 | 151,484,416 | 151,554,048 |
| Pure one-query C16 steps | 236 | 224 | 256 |
| Multi-query-only / mixed steps | 14 / 19 | 2 / 31 | 9 / 0 |

SGLang performs one extra query per request and packs work differently.
LunaFlux/vLLM have ten common exact query/history vectors; LunaFlux/SGLang
have none. Whole-window comparisons are useful but cannot be presented as
identical per-step work. Query/history equality alone also does not establish
identical page locality, operands, or floating-point association/precision.

## Exact-work examples against vLLM

Notation is `query_count:prior_history`; a comma separates active rows.
Times are whole 28-layer chain kernel sums for the matched vector, not single
kernel time or exclusive wall time.

| Work vector and chain | LunaFlux ms | vLLM ms | Difference ms |
| --- | ---: | ---: | ---: |
| `2048:2048` attention | 23.183 | 23.277 | −0.095 |
| `2048:2048` projection/post-ops | 39.816 | 33.477 | +6.339 |
| `1:4097,2047:2047` attention | 27.118 | 22.876 | +4.242 |
| `1:4097,2047:2047` projection/post-ops | 39.575 | 31.726 | +7.849 |
| `1:4097,2047:2047` norm/embedding | 2.344 | 2.757 | −0.413 |
| `1:4097,2047:2047` head/sampling | 1.404 | 1.379 | +0.025 |

For the mixed example, complete kernel activity is 70.441 versus 58.757 ms,
including vLLM's 0.018 ms unmapped activity. Thus fewer launches and fusion do
not establish a faster chain. Conversely, the first example disproves a blanket
claim that every long-prefill attention invocation remains slower.

Some one-query-tail reference projections use generic GEMV symbols. Their
7.408 ms of unmapped activity in one matched tail must not be mistaken for
zero projection/head time, or used to claim a 34× LunaFlux projection slowdown.

## Between-step GPU envelopes

Kernel activity is joined to actual graph correlations or same-thread CPU work
markers. A running maximum of prior end times prevents overlap double counting.
These gaps can contain transfers, synchronization or profiler overhead; they
are not automatically CPU scheduler cost.

| Envelope measurement, ms | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| Between-step kernel gaps | 387.641 | 90.857 | 0.018 |
| Leading window edge | 7.565 | 12.526 | 11.227 |
| Trailing window edge | 10.528 | 13.455 | 0.662 |

SGLang's step envelopes overlap or nearly touch; this does not mean all its
kernels have zero internal gaps. LunaFlux's no-activity time lies mostly between
steps, not at client-window edges. These envelope numbers are a second lens,
not additional rows to add to the exclusive budget. All 59,248 observed LunaFlux
kernel calls are graph nodes: generic eager fallback is not the explanation.

## What this measurement justifies next

1. Resolve generic reference GEMV/reduction operation identities, then isolate
   the matched projection/post-op chains. Capture their selected hardware
   counters and full-chain unprofiled durations, not an attention-only probe.
2. Follow the host submission/completion fences across step boundaries to
   explain LunaFlux's 388 ms envelope gap. Separate mandatory output feedback
   from avoidable drains and missing submission overlap.
3. Measure matched mixed-attention paths separately from pure prefill and
   recurring C16 one-query paths. Include split/combine and numerical-law
   differences; do not re-propose already implemented async retirement.

This ordering follows measured time, not a promise that another pipeline-stage
change will close the whole gap. The preceding
[hardware-counter/source comparison](BENCHMARK_REFERENCE_LATENCY_HIDING_2026-10-04.md)
remains useful but its replay counters must not be converted into additive
serving milliseconds.

## Reproduction, safety and preserved data

Runner: `benchmarks/gpu_pipeline/measure_current_full_step.mbtx`.
Offline summaries: `summarize_full_step_measurement.mbtx` and
`measure_current_step_gaps.mbtx`; post-campaign checks:
`finalize_current_full_step.mbtx`. These use MoonBit automation. The MoonBit
guide influenced tooling only; production execution was untouched.
The runner requires the pinned serving tree and helper sources recorded in the
archive, including `profile_reference_gap.mbtx`; it is not a standalone
fresh-checkout benchmark. The archive preserves the exact executed helper
versions and CPU-marker reference patches.
Affected scripts passed formatting and warning-denied native checks. Memory
limit, query-phase/unknown classification and memory-minimum regressions passed
(one test each). Existing interval/work-ledger self-tests also passed.

Reference containers were limited to 64 GiB memory without swap and report
`OOMKilled=false`; minimum sampled host MemAvailable was 53,113,472 KiB,
above the 32 GiB reserve. The final compute-process query is empty.

Unchanged serving worker SHA-256:
`ea8d9982c2381cca35210c72aa83918aa9c4f84d1819111001ad5696caecf597`.
CPU-marker diagnostic worker:
`bc5088f26100ff8c8cc5ce29d86916565f1376ba17ff2a4398cc8114d62b2b9e`.
Pinned images: vLLM
`73e1b0f3230a9377a3109d20be7f8607963a41a715eb3000bc647f5f73c198ff`;
SGLang `3a9d399434923689565e1a72f5c3fc409323ce039b65bdbe5c2d6ed14abad3a9`.

First root: `/home/wlc004s/lunaflux-full-step-measure-20261004.4iNCg48j`.
Its Luna capture completed; the first reference launcher failed on slash-containing
container names before reference GPU work. The corrected launcher replaces
every slash, not just the first. The completed Luna capture was reused without
rerunning or overwriting it in the successful comparison root:
`/home/wlc004s/lunaflux-full-step-measure-v2-20261004.tIVT3hCB`.
An initially expensive offline SQLite join was stopped; the retained query was
replaced by bounded, materialized marker joins. No GPU capture was interrupted.

Both roots are preserved. The archive contains 541 readable files; build,
dependency, deployment/model copies are explicitly excluded in its inventory,
not deleted. Local summaries, raw traces and inventory:
[`benchmarks/results/full-step-20261004.yctDvljQ`](../benchmarks/results/full-step-20261004.yctDvljQ/).
Remote seal: `/home/wlc004s/lunaflux-full-step-archive-20261004.1Qt5ths4/sealed`.
Archive SHA-256:
`7f31672d87458e2f2cdc16d4d41ba715bd8186971968cbd7b911f45599cb36b6`.
The downloaded archive hash matches. Nothing was extracted into the module or
overwritten during download.
