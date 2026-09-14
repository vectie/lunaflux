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
was changed. Final plain reruns and any remaining differences must be
reported separately; this document does not claim parity with either baseline.
