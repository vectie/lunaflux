# Async attention measurement audit — 2026-09-14

## Scope and correction

This is a diagnostic measurement, not a production kernel change. The earlier
approximately 1% end-to-end change used synchronous c318; it was not a serving
A/B of c318 against asynchronous c322. It cannot establish that asynchronous
attention has no production benefit.

The new microbenchmark uses the same metadata-v1 ABI, input buffers, layouts,
grid and block for both artifacts. It compares c318 with c322 on RTX 5060 Ti,
GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6. No competing GPU workload was launched.
Automation follows the MoonBit agent guide and uses diagnostic `.mbtx` drivers.

## Measurements completed

- 16 shapes, six process blocks per shape, five alternating A/B pairs per block:
  480 paired observations. Each timed kernel measurement uses three warmups and
  30 CUDA-event-timed launches.
- All 480 comparisons passed bitwise output equality, sampled independent scalar
  reference checks, and unchanged-KV checks. This is not full model correctness.
- 20 Nsight Compute reports: five shapes, two kernels, cache-control `all` and
  `none`. Captured launch resources, occupancy, eligible warps, issue activity,
  tensor activity, memory analysis, executed instructions, PC-sampled stalls,
  and SASS source pages.

Query tokens below are **total across rows**, not tokens per request. History is
the fixture's base history per row. For multiple rows the fixture additionally
adds `(row % 3) * 8` history tokens, even when the base is zero. Those cases must
not be described as uniformly zero-history prefill.

| Total query | Rows | History base | Sync median µs | Async median µs | Mean paired-block async time change |
|---:|---:|---:|---:|---:|---:|
| 63 | 1 | 0 | 19.42 | 18.88 | -2.48% |
| 64 | 1 | 0 | 19.21 | 18.80 | -1.80% |
| 65 | 1 | 0 | 22.63 | 24.20 | +6.82% |
| 512 | 8 | 0 | 38.05 | 39.57 | +4.04% |
| 520 | 8 | 0 | 55.41 | 59.44 | +7.30% |
| 1528 | 1 | 0 | 459.68 | 498.89 | +8.55% |
| 1528 | 8 | 0 | 112.55 | 122.56 | +8.94% |
| 2048 | 1 | 0 | 762.84 | 829.00 | +8.74% |
| 2048 | 8 | 0 | 164.02 | 177.69 | +8.56% |
| 2048 | 16 | 0 | 119.19 | 127.90 | +7.31% |
| 2048 | 1 | 2048 | 2177.63 | 2233.03 | +2.56% |
| 2048 | 8 | 2048 | 1639.01 | 1631.63 | -0.64% |
| 1528 | 8 | 4096 | 2593.52 | 2367.71 | -8.74% |
| 2048 | 1 | 4096 | 3640.79 | 3669.62 | +0.86% |
| 2048 | 8 | 4096 | 3156.53 | 3115.65 | -1.36% |
| 2048 | 16 | 2048 | 1608.39 | 1594.71 | -1.07% |

Negative means faster. Change is the mean of within-block paired changes, not
the ratio of the two marginal medians. Approximate block-t 95% intervals include
[8.50%, 8.61%] for 1528/1/0 and [-8.88%, -8.60%] for 1528/8/4096. These describe
this session, not cross-day uncertainty or 30 independent samples per shape.

## Hardware observations

Cache-control-none profiles, with three warmup launches:

| Metric | 1528/1/0 sync → async | 1528/8/4096 sync → async |
|---|---:|---:|
| Executed warp instructions | 32,432,520 → 43,808,018 | 166,931,872 → 222,423,936 |
| Tensor active / sustained elapsed peak | 42.79% → 39.24% | 40.62% → 44.85% |
| Eligible warps / active cycle | 0.22 → 0.28 | 0.19 → 0.28 |
| Issue-active cycles | 20.11% → 24.70% | 17.06% → 25.08% |
| Achieved occupancy | 15.77% → 15.76% | 16.07% → 15.84% |

Both launch 128 threads per block and consume 50,192 bytes shared memory per
block including driver overhead. Registers rise from 168 to 173, but shared
memory already limits both to two resident blocks per SM. A claim that the
register increase halves residency would be incorrect for these launches.

The async implementation executes roughly 33–35% more instructions in these
two cases. Higher issue activity therefore does not by itself mean more useful
work. Its tensor activity decreases in the no-history case and increases in the
history-heavy case. The measured benefit is workload-dependent; these counters
support investigating the added instruction work, not assuming a universal
bandwidth or occupancy explanation.

PC stall samples and per-instruction pages are retained for attribution. They
are samples, not percentages of total kernel elapsed time. Derived local/shared
spilling requests are zero in the collected summaries; this does not substitute
for explicit local-memory traffic counters. Exact DRAM read/write totals and
hardware bank-conflict totals were not explicitly requested in this collection.

## Validity limits and remaining measurements

- Synthetic inputs and repeatedly reused buffers are not real Qwen activations.
  Ordinary timings are warm-cache measurements. Cache-flushed profiler runs are
  a separate experiment, not ordinary serving latency.
- NCU replay reports warn about uncontrolled clocks/caches. One derived L2 hit
  percentage exceeds 100%; it is inconsistent and must not be interpreted as a
  physical hit rate. Do not use replay timings as replacement benchmark scores.
- Production trace attempt r1 completed client traffic but exported **no CUDA
  events**. Its successful client exit is not successful trace validation.
  Attempt r2 added periodic flushing and an explicit profiler stop, with the
  stop session recovered using the matching TMPDIR. It also contains no CUDA
  kernel data. Owned diagnostic services were stopped and GPU idle confirmed.
  Flushing alone therefore does not fix this instrumentation gap; child-process
  profiler injection must be checked before another serving trace.
- Production-selected shapes, per-step row distribution, kernel gaps, host waits,
  and a full synchronous/asynchronous serving A/B remain necessary before making
  an end-to-end async recommendation. The follow-up matched vLLM/SGLang
  attention comparison and explicit traffic counters are now recorded in
  [the three-way report](ATTENTION_THREEWAY_METRICS_2026-09-14.md); this original
  collection itself did not include those baselines.

## Raw results

Remote root: `/run/lunaflux-toolchain-4896771-20260913`.

- `async-metrics-timing-20260914-r1`: timings, correctness, SUMMARY.json.
- `async-metrics-counters-20260914-r2`: 20 reports, raw metrics, SASS pages,
  COUNTER_SUMMARY_EXPANDED.json.
- `async-production-trace-20260914-r1`: preserved failed CUDA collection.
- `async-production-trace-20260914-r2`: preserved second failed CUDA collection.

Timing and NCU archive downloaded without overwrite:
`/tmp/lunaflux-async-metrics-download.stJoOM/lunaflux-async-metrics-20260914-r1.tar.gz`.
Remote and local SHA-256 both:
`7d93261d936ca2f8fb651755804e56cf52e5cee5c7354370c874a5896f1836f0`.

The immediate conclusion is not “the test set is wrong” or “async is useless”.
The old aggregate did not test that hypothesis; the expanded A/B demonstrates
both regressions and improvements depending on query/history geometry.
