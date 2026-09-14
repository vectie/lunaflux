# Long-input capacity and timeline diagnosis

This supplements `THREEWAY_LONG_INPUT_AUDIT_2026-09-14.md`. The historical
numbers remain valid observations, but the C16 result includes KV-pressure
recomputation and must not be attributed entirely to kernel efficiency.

## Measured service work

The unmodified benchmark worker/CUBINs were profiled with an isolated parent
that records submitted plan sizes. All observed kernel calls used graphs.
4096 input / 64 output, one warmup and one profiled measurement:

| Concurrency | Steps | Output tokens | Kernel time | Inter-step GPU gaps |
| --- | ---: | ---: | ---: | ---: |
| 8 | 80 | 512 | 2292.81 ms | 70.94 ms |
| 16 | 126 | 1024 | 5666.62 ms | 149.34 ms |

C16 submitted **94,352 prefill tokens for 65,536 input tokens**, a 43.97%
amplification. Pure decode reached 15 rows; the problem is not simply absent
batching. Profiled rates (212.62 / 174.09 tok/s) agree closely with the plain
measurements (214.69 / 175.32); profiler timings are not replacement scores.

## Capacity correction

The old 8,192 pages × 8 tokens accommodate exactly the prompts, with no space
for generation. A conservative non-sharing capacity is
`concurrency * ceil((input + output) / tokens_per_page) + reserve_pages`:
8,321 pages for this vector. Requests are rounded separately. Different
trial cells use the maximum, not the sum. Prefix reuse must not be assumed.

The physical KV pool and flattened worker page-table envelope must both grow.
Increasing only the pool fails `WorkerPlanPages` capacity resolution. Fused
export previously imposed an 8,192-entry ceiling, too small for the 8,320
live entries in this batch. Its bounded ceiling is now 16,384; regression
coverage includes 8,193, 8,320, 9,216, 16,384 and rejects 16,385. This does not
increase any allocation automatically or add work to a live token step.

Before an ordinary throughput campaign, run the pure offline planner against
the materialized capacity and the full vector, for example:

```sh
moon run scripts/plan-benchmark-kv-capacity.mbtx -- --self-test
moon run scripts/plan-benchmark-kv-capacity.mbtx -- 8 1 9216 4096 64 16
```

Insufficient-capacity trials belong in a separately labeled pressure campaign.
Do not silently change the declared pool while reusing old layout-bound
artifacts. The new 9,216-page artifacts are regenerated; fresh observations
again selected attention candidate 322 over 318 and 324.

## Decode split is not a blanket fix

The production grouped decode and compiler split-8 path were tested against
the same double-precision oracle, including preserved KV and resource release.
Both partial and merge launches are timed. Representative microseconds:

| Context / batch | Ordinary | Split-8 |
| --- | ---: | ---: |
| 4096 / 1 | 272.60 | 63.52 |
| 4096 / 4 | 328.56 | 246.51 |
| 4096 / 8 | 340.13 | 476.51 |
| 4096 / 16 | 695.22 | 883.81 |
| 1528 / 8 | 136.56 | 198.33 |

Thus removing the C4 dispatch limit would regress these larger batches. The
test now includes independent context and concurrency axes instead of only
C1 plus one C8 example. These are synthetic operator tests, not serving scores.

## Raw diagnostics

Local service trace and derived per-step tables:
`/tmp/lunaflux-long-profile-local-20260914`. Remote originals:
`/tmp/lflong.ZRjL4F/lunaflux`. The trace runner's initial stop command lacked
its TMPDIR; the trace was recovered and exported, then the exact child
process group was stopped. Do not call the original runner result a pass.

Decode operator results: `/tmp/lfcapacity.3qmhRb/decode.stdout` and
`decode.stderr` (empty), exit 0. The separate failed capacity-only
materializations are retained as diagnostic logs. No production deployment
was changed. This document does not claim parity with either baseline.

## Completed capacity-fixed rerun

The new isolated runtime used 9,216 physical pages and 9,216 plan entries.
The same BF16 Qwen3-0.6B token-ID vector used one warmup and five measured
trials per cell, prefix reuse disabled. Rates are output tokens/second:

| Input / output / concurrency | Previous LunaFlux | Capacity-fixed LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: | ---: |
| 512 / 64 / 1 | 211.65 | 209.31 | 244.29 | 238.46 |
| 512 / 64 / 8 | 1033.52 | 1030.61 | 1180.81 | 1135.26 |
| 512 / 64 / 16 | 1375.68 | 1381.55 | 1648.43 | 1454.50 |
| 1528 / 32 / 1 | 149.42 | 149.13 | 178.22 | 175.06 |
| 1528 / 32 / 8 | 367.29 | 367.50 | 443.99 | 418.03 |
| 1528 / 32 / 16 | 405.45 | 404.62 | 493.07 | 467.07 |
| 3072 / 32 / 1 | 107.54 | 106.32 | 127.20 | 122.72 |
| 3072 / 32 / 8 | 184.07 | 184.28 | 222.88 | 216.25 |
| 3072 / 32 / 16 | 195.03 | 195.06 | 232.35 | 228.94 |
| 4096 / 64 / 1 | 116.53 | 115.78 | 140.79 | 136.94 |
| 4096 / 64 / 8 | 214.69 | 214.42 | 255.36 | 251.85 |
| 4096 / 64 / 16 | 175.32 | 228.01 | 266.38 | 260.71 |

The pressure-affected cell gains 30.05% throughput; its wall time falls from
5,840.8 to 4,491 ms. Other cells are essentially unchanged. Token counts
passed, but not all generated sequences are bitwise identical across engines;
these results do not establish cross-engine numerical parity.

Fresh traces confirm C16 prefill work is exactly 65,536 tokens, not 94,352.
Steps fall from 126 to 96; pure decode uses 63 steps and prefill/mixed uses 33.
LunaFlux kernel time is 4,320.19 ms, with 136.98 ms inter-step gaps. Independently
profiled vLLM/SGLang kernel totals are 3,805.98/3,748.00 ms. All LunaFlux kernel
calls in this trace use graphs. The remaining difference cannot be explained
as missing batching or graph fallback alone.

Plain results: `/tmp/lfcomplete.B1rRhK/THREEWAY.json` and `lunaflux/` on the
test host. Local fixed trace: `/tmp/lunaflux-complete-profile-final-20260914`;
baseline traces: `/tmp/lunaflux-baseline-profiles-20260914`.

## Lane-varying rotary basis storage

The pure compiler already computes the rotary basis ahead of time. Its CUDA
lowering nevertheless placed the lane-indexed table in constant memory.
Using read-only device storage retains the same basis, operations and rounding
boundaries while allowing contiguous lane accesses. Both fusion cuts consume
this shared lowering; there is no model-specific dispatch or runtime tuning.

Alternated old/new microbenchmarks passed bitwise output and preserved-KV
checks. At 2,048 tokens / one row the measured duration falls from approximately
169.47 to 136.76 microseconds; at 1,528 / one row, 96.76 to 71.37. Small one-row
decode is unchanged. Separate selected-kernel counters report 598,016 to
507,904 LDC instructions and 191.46 to 141.95 microseconds under profiling.
The profiled durations are not substituted for ordinary benchmark timings.
Memcheck and leak checking passed with zero errors and zero leaked allocations.
The rebuilt runtime completed the same 12-cell end-to-end vector. Long-input
rates after both fixes are:

| Input / output / concurrency | LunaFlux after both fixes | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| 3072 / 32 / 1 | 107.24 | 127.20 | 122.72 |
| 3072 / 32 / 8 | 186.13 | 222.88 | 216.25 |
| 3072 / 32 / 16 | 196.80 | 232.35 | 228.94 |
| 4096 / 64 / 1 | 116.41 | 140.79 | 136.94 |
| 4096 / 64 / 8 | 216.25 | 255.36 | 251.85 |
| 4096 / 64 / 16 | 229.68 | 266.38 | 260.71 |

The rotary storage change adds approximately 0.73% throughput at 4096/C16;
combined with the capacity correction the gain is approximately 31.0%.
This is not the operator's 16–26% gain applied to the whole engine. C16 wall
time is now 4,458.4 ms, still 16.0% above vLLM and 13.1% above SGLang.
All 500 measured requests again have the required output length. Existing
cross-engine sequence-pool caveats still apply.

The tested source archive is
`db8a55198c670cd96fc58759ee220f76b423ae6cfb81d1ceac1228440cc716fd`.
Remote results: `/tmp/lfreadonly.OZNvH2`. Downloaded archives:

- `/tmp/lunaflux-capacity-final-20260914.tar.gz`, SHA-256
  `6db076c6ad09027236d6342240c849bccd9306aac211bed23c141225fde54772`.
- `/tmp/lunaflux-readonly-final-20260914.tar.gz`, SHA-256
  `55b83d52240554a83f84838e749f5248a7f3ba4f6a71c42d4dc02586b9d918c6`.

Local warning-denied native check and all 3,741 native tests passed. Native C
allocation-probe macro warnings remain distinct from MoonBit warnings. The
benchmark server was stopped after completion; no production cutover occurred.

## Rejected QKV selection-hoisting experiment

An isolated generated-source experiment moved Q/K/V segment base selection
outside `stage_operands`, where all columns in these aligned tiles belong to
one segment. Launch geometry matched the recipe (4,096 CTAs, 256 threads,
8,192 dynamic shared bytes). Full output bytes and sampled scalar references
passed at 1, 8, 16, 128, 1,528 and 2,048 tokens. Register count stayed at 72.
However, 2,048-token timing increased from approximately 438.43 to 453.80 us;
1,528-token timing also regressed. No production compiler change was made.
Fewer source-level selections alone are not evidence of a faster instruction
schedule. Raw experiment: `/tmp/lfreadonly.OZNvH2/qkv-segment-v2` on the GPU
host. The first probe build failed because of a diagnostic macro name clash;
that failed build is preserved separately under `qkv-segment`.
