# Concurrency sweep: ordinary timing and isolated GPU diagnosis

## Scope

This is a diagnostic follow-up to
[the source-zero matrix and three-engine comparison](SOURCE_COUNTERS_AND_QWEN_COMPARISON_2026-09-10.md).
The ordinary runtime is the unchanged `13369c5` release on the RTX 5060 Ti
(`GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, PCI `00000000:17:00.0`).
The preceding implementation and report were pushed through `26cdb81` before
this campaign. No production service, worker binary, or kernel was changed.

Model: Qwen3-0.6B BF16. Frozen token-ID inputs, greedy generation, ignored EOS,
prefix reuse disabled. Each ordinary cell has one excluded warmup and three
measured bursts at concurrency 1, 2, 4, 8, or 16. This measures synchronized
finite bursts, not a steady-state arrival-rate or saturation test.

## Ordinary measurements

All numbers below are means of three measurements without a profiler or
concurrent compilation. Output lengths and generated-token accounting passed;
failure, cancellation, worker-failure, and worker-restart counters did not
increase. Client post-first spacing is not a GPU timer.

| Input / output tokens | Concurrency | Aggregate output tok/s | Mean TTFT ms | Client post-first ms/token |
| --- | ---: | ---: | ---: | ---: |
| 59 / 256 | 1 | 247.10 | 19.33 | 3.97 |
| 59 / 256 | 2 | 178.00 | 20.67 | 11.18 |
| 59 / 256 | 4 | 347.00 | 24.42 | 11.45 |
| 59 / 256 | 8 | 668.12 | 34.29 | 11.85 |
| 59 / 256 | 16 | 1480.49 | 48.33 | 10.61 |
| 1528 / 32 | 1 | 133.16 | 91.67 | 4.73 |
| 1528 / 32 | 2 | 115.25 | 165.00 | 12.48 |
| 1528 / 32 | 4 | 167.91 | 293.83 | 14.84 |
| 1528 / 32 | 8 | 227.49 | 552.63 | 17.73 |
| 1528 / 32 | 16 | 276.26 | 1042.25 | 24.28 |

The C1-to-C2 cliff is reproducible: short-input aggregate throughput falls
28.0%, while client token spacing rises 2.81 times. Long-input TTFT also grows
substantially with concurrency. These observations alone do not distinguish
GPU inefficiency from host scheduling, synchronization, or graph selection.

## Telemetry interpretation

`engine/worker_wire/graph_telemetry.mbt` reports the first four steps and then
every 64th sequence, carrying cumulative counters. Consequently, a short
request's metrics-before/after delta can be zero or include work preceding
that request. It is incorrect to divide a cell's token count by this delta
and call the result an exact batch-size histogram. Periodic samples showed
no graph misses, but exact per-step attribution requires the trace.

## Diagnostic isolation

Ordinary timing is under `plain-clean` in the campaign directory
`/run/user/1000/lunaflux-batching-20260910-r1`. The earlier `plain` run is retained
but not used because its first trials overlapped the end of diagnostic
compilation.

The diagnostic parent preserves profiler injection across worker exec and
records the submitted plan's sequence, rows, tokens, prefill rows, decode rows,
and output-producing rows. The worker and kernel artifacts remain unchanged.
These parent-only changes are temporary instrumentation, not production
token-step logging or a compiler optimization.

NVTX-enabled startup attempts failed in optional inherited-credential
preparation. CUDA-only startup succeeded. The replacement marker uses a
realtime timestamp and the parent's existing stdout descriptor, avoiding an
additional diagnostic descriptor. Raw startup failures are retained; they
are not counted as model or kernel benchmark failures. A further attempt
reached readiness but the diagnostic controller did not resolve the
profiler's descendant process tree; that controller was corrected before
the full capture retry.

The first full request capture (`profile-v5`) lost its GPU tail: the stop
command used a different `TMPDIR` from the profiler and failed to find the
session. Its audit found 185 markers without kernels. It is not used for
the following tables. The corrected capture, `profile-v6`, stopped and
flushed successfully before terminating its own process group. Benchmark
ports closed and the GPU was idle afterward.

## Complete trace: batching works, small-batch kernels dominate

The final capture has 2,978 submitted-plan markers and 719,422 kernel calls.
Every marker has GPU work; no kernel crosses its assigned next-plan boundary.
All 719,422 calls have a nonzero CUDA graph ID. Each measured cell accounts
for exactly `concurrency × output_tokens` output-producing rows and
`concurrency × (output_tokens − 1)` decode rows. No graph fallback was observed.

One warmed diagnostic trial is analyzed per cell; it is not an ordinary
throughput sample. Profiling lowers measured throughput approximately 1–8%
relative to the ordinary means. GPU durations below are trace measurements,
not inferred from SSE timing. Kernel sums can include overlapping execution;
busy time uses the union of kernel and memcpy intervals.

| Input / output | C | Full-size pure-decode steps / all pure-decode steps | Kernel time per full-size decode step, ms | GPU span per full-size step, ms | Total between-step GPU gaps, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 59 / 256 | 1 | 255 / 255 | 3.841 | 4.180 | 38.52 |
| 59 / 256 | 2 | 254 / 255 | 10.978 | 11.332 | 60.46 |
| 59 / 256 | 4 | 254 / 255 | 11.130 | 11.448 | 75.18 |
| 59 / 256 | 8 | 253 / 255 | 11.251 | 11.568 | 129.29 |
| 59 / 256 | 16 | 252 / 255 | 9.678 | 9.981 | 229.36 |
| 1528 / 32 | 1 | 31 / 31 | 4.632 | 4.969 | 6.73 |
| 1528 / 32 | 2 | 31 / 31 | 12.188 | 12.509 | 8.24 |
| 1528 / 32 | 4 | 30 / 31 | 13.452 | 13.806 | 14.07 |
| 1528 / 32 | 8 | 28 / 31 | 14.605 | 14.920 | 25.55 |
| 1528 / 32 | 16 | 23 / 31 | 16.332 | 16.662 | 56.29 |

The last column sums gaps between adjacent submitted steps' GPU activity
within that cell. It is not the duration of one step, and does not include
intra-step gaps. A full-size step means the submitted plan actually has C
decode rows, not merely that a C-sized kernel bucket was selected.

For short C2, 508 of 510 pure-decode output rows belong to two-row steps.
For short C8, 2,024 of 2,040 belong to eight-row steps. The scheduler is
therefore effectively batching the steady portion of these bursts.

Short C2 spends 2,806.36 ms busy on the GPU within a 2,957.73 ms GPU activity
envelope (94.9%); only 60.46 ms is between steps. C1 spends 985.78 ms busy
within 1,110.60 ms. The approximately 1,821 ms additional GPU work dominates
the C1-to-C2 slowdown. Eliminating host gaps alone cannot recover it.
Short C16 still has 229.36 ms between-step gaps (8.1% of its activity
envelope), so host-side improvement is worthwhile but secondary here.

### Where the short-input decode increase comes from

Times are the sum of each family across a full model step (28 layers where
applicable), in milliseconds, restricted to actual full-size decode plans.

| Family | C1 | C2 | C8 | C16 | C2 minus C1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP down | 0.511 | 2.627 | 2.683 | 0.959 | +2.117 |
| Vocabulary head | 0.738 | 2.390 | 2.479 | 2.636 | +1.652 |
| MLP gate/up | 0.865 | 2.472 | 2.482 | 2.532 | +1.607 |
| Output projection | 0.311 | 1.154 | 1.160 | 1.188 | +0.843 |
| QKV projection | 0.587 | 1.424 | 1.434 | 0.786 | +0.837 |
| Attention | 0.439 | 0.516 | 0.593 | 1.098 | +0.076 |
| Residual/norm | 0.267 | 0.265 | 0.258 | 0.254 | −0.002 |
| QKNorm/RoPE/KV-write auxiliary | 0.113 | 0.119 | 0.147 | 0.202 | +0.006 |
| Sampling | 0.009 | 0.009 | 0.015 | 0.022 | ~0 |

Down, head, and gate/up account for about 75% of the 7.137 ms per-step
increase; adding output and QKV accounts for about 99%. Sampling and
residual normalization are not the cause of this cliff.

The selected launch records identify an actual schedule discontinuity:

| Kernel | Actual rows | Selected variant suffix | Grid X / block X | Registers/thread | Shared bytes | Mean call µs |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| Down | 2 | `rows8_down` | 32 / 64 | 115 | 26,624 | 93.83 |
| Down | 16 | `down` | 16 / 128 | 112 | 36,864 | 34.27 |
| QKV | 2 | `rows8` | 256 / 32 | 111 | 16,384 | 50.87 |
| QKV | 16 | base variant | 32 / 256 | 102 | 38,912 | 28.07 |

Doing eight times as many rows can be faster in these families because it
selects a different schedule. This is measured evidence for prioritizing
the compiler's small-row strategy and bucket dispatch. It does not, by
itself, attribute the difference to a particular instruction, bank conflict,
or exact number of padded FLOPs; those require selected-kernel counters or
a controlled schedule substitution.

## Long inputs: prefill/mixed work and decode attention both matter

| C | Pure-prefill kernel ms | Mixed-step kernel ms | Pure-decode kernel ms |
| ---: | ---: | ---: | ---: |
| 1 | 79.46 | 0 | 143.58 |
| 2 | 155.75 | 0 | 377.84 |
| 4 | 252.25 | 64.99 | 415.80 |
| 8 | 443.39 | 195.99 | 449.07 |
| 16 | 768.70 | 532.51 | 490.67 |

Mixed-step totals include both prefill and decode work and must not be
called pure-prefill time. C16 executes 55 total steps: 16 pure-prefill,
8 mixed, and 31 pure-decode. Fifteen of its pure-prefill steps have one
request row and 1,024 tokens, averaging 47.64 ms of kernel execution each.
The remaining prefill/mixed plans also expose the 1,024-token work budget.
This is a finite prompt-processing ramp, not proof that requests never batch.
Whether a larger budget improves TTFT without harming active decode needs a
controlled experiment, not an automatic increase.

The long-context decode attention family costs 1.220 / 1.697 / 3.883 /
7.729 ms per full-size step at C1 / C2 / C8 / C16. It contributes 47.3% of
the long C16 decode kernel sum. The selected C16 decode-attention kernel
uses grid `(16,8)`, block 256, 71 registers/thread and 33,820 shared bytes;
its mean call is 276.03 µs. Long-context attention remains a distinct target.

At long C8, a pure-prefill 1,024-token step averages about 11.61 ms QKV,
10.77 ms attention, 10.11 ms gate/up, 5.19 ms output projection, 4.99 ms
down, and 4.05 ms QKNorm/RoPE/KV-write auxiliary work. Optimizing only
attention will not remove the entire prefill cost.

## Next optimization boundaries

1. Fix the backend-neutral small-row strategy/cost model and its CUDA
   lowering choices for down, head, gate/up, QKV and output projection.
   Keep one semantic compiler path; compare schedules at actual rows
   1/2/4/8/16, including padding and launch geometry.
2. Optimize long-context decode attention separately from prefill attention.
3. Evaluate the 1,024-token budget and mixed-phase planning using matched
   workload experiments, preserving decode-latency constraints.
4. Reduce host/transport and intra-graph gaps after the dominant GPU work.

No optimization is implemented by this report. No fresh vLLM/SGLang C2/C4/C16
results were collected; the prior same-hardware C1/C8 comparison remains the
only baseline comparison and should not be extrapolated to unmatched cells.

## Reproducibility

Raw campaign archive (includes failed attempts, ordinary measurements,
final `.nsys-rep`, parent instrumentation delta/binary and `.mbtx` drivers):
`lunaflux-batching-results-20260910-r1.tar`, 67,205,120 bytes,
SHA-256 `f06bdcc57703447b835a5b15822ac413900db8bd102a0e1e4585529cd2354dd5`.
Downloaded under `/private/tmp/lunaflux-batching-results-20260910-r2/` and
verified against the remote hash. No existing result was overwritten.

Final raw SQLite SHA-256:
`818df65657ff5a95b7f4a4b1cd710373682c7dd1acdca2d32cd84c6c8c52e980`.
Final `.nsys-rep` SHA-256:
`37b2d01f3eb04d9b0d9ec86e9e3ce6c60dc91d6805bdda81f1eaa2809935b7bd`.
Local analysis in the same directory includes `trace-audit.json`, per-cell
step/family/kernel JSON, `compact.json`, and `ANALYSIS_RESULT.txt`. Analysis
uses MoonBit `.mbtx` orchestration and SQLite window queries; no production
runtime dependency or authentication policy was changed.

The derived JSON and analysis scripts are also archived as
`lunaflux-batching-analysis-20260910-r1.tar` in that local directory,
SHA-256 `47e4efa88f6ed93e05ff218ae5eeda1da62b3fdd3f2fee22eac4a70f6e7a1f2f`.
