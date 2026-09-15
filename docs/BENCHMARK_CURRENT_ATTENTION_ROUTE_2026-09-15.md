# Current serving attention route: measured, not inferred

Base commit: `3b2f25d4`. Qwen3-0.6B on the same RTX 5060 Ti. The ordinary
uninstrumented matrix is documented in
[the integrated benchmark](BENCHMARK_REGISTER_EPILOGUE_2026-09-15.md).
Nsight measurements below are diagnostic; they do not replace that matrix.

## Exact current-source long-input timeline

The corrected profile measures 4096 input / 64 output tokens at concurrency 16,
one warmup followed by one measured trial. The first profiler invocation's
profile-mode vector override still selected 59/256 and 1528/32; that run is
preserved but is not used for the long-input claims. The ordinary benchmark
did not use profile mode and is unaffected.

The trace joins 192 execution markers to 192 graph launches with no unmatched
steps (warmup plus measured). The measured half contains 96 steps, exactly
65,536 prefill tokens, 1,008 decode tokens and 1,024 outputs. Query bucket slack
is 43. Route selection totals 103.34 microseconds; kernel execution totals
4,249.856 ms; between-step GPU gaps total 181.753 ms. This is not a failure to
capture the graphs or duplicate-prefill work.

| Measured kernel family | GPU time (ms) |
| --- | ---: |
| Attention | 2314.20 |
| Gate/up | 679.75 |
| QKV | 448.39 |
| Down, including row-specialized calls | 305.36 |
| Output | 239.72 |
| Other | 262.43 |

Attention accounts for approximately 54.5% of kernel time. The down optimization
does not target that time. Pure decode occupies 1,374.41 ms; mixed steps
1,612.63 ms; pure prefill 1,262.82 ms. The primary unsplit decode attention group
alone accounts for 1,084.56 ms across 1,540 calls, with grid 16 by 8 and 256-thread
blocks. The existing compiler-readonly split policy only permits batch rows up
to four; the C16 pure-decode route is therefore unsplit.

## Rejected experiment: enabling the existing split route up to 32 rows

An isolated worker changes only that limit from four to 32. Existing kernel
artifacts and the remaining runtime are unchanged. No tracing is installed in
this experiment. Each cell has one warmup and five measured trials; all expected
output lengths pass. It is not a production change.

| Input/output, concurrency | Current tok/s | Experimental split tok/s |
| --- | ---: | ---: |
| 512/64, C8 | 1036.44 | 936.01 |
| 512/64, C16 | 1395.10 | 1239.71 |
| 1528/32, C8 | 373.18 | 351.17 |
| 1528/32, C16 | 412.57 | 395.67 |
| 3072/32, C16 | 198.60 | 193.28 |
| 4096/64, C8 | 217.50 | 202.13 |
| 4096/64, C16 | 231.57 | 219.27 |

For long C16 the median wall time increases from 4422 to 4670 ms (+5.61%),
while pooled TTFT p50/p95 stays essentially unchanged: 1689/2926 versus
1688/2925 ms. Simply enabling more split work is not the missing optimization.
The additional route requires selected partial/merge timing before attributing
the regression to bandwidth, arithmetic, or synchronization. It must not be
shipped based on CTA count alone.

The follow-up split trace confirms the same 96 steps and exact token work.
Pure-decode kernel time rises from 1374.41 to 1622.93 ms; mixed and prefill
remain 1613.15 and 1262.92 ms. In pure decode, the experimental partial kernels
take 1378.45 ms and merges only 12.58 ms. The baseline's unsplit plus partial
plus merge attention is 1122.42 + 18.29 + 0.33 ms. Thus partial execution, not
the merge alone, accounts for the regression. This trace does not by itself
identify an instruction-level stall.

Source inspection also finds the partitioned adapter supplies
`supports_async_copy=false` and `max_pipeline_stages=1` to both compilation
and runtime geometry planning. Broad split therefore changes more than the
number of partitions relative to the async unsplit implementation. The next
isolated experiment enables the existing two-stage capability in both places;
it is not an unconditional production default or a claim of measured gain.

That experiment is now complete and also rejected. With both compiler and
geometry capabilities updated, 3108/3108 tests and seven release builds pass;
fresh AOT export, compile and binding pass. The emitted split source uses a
64-key double-buffered tile and actual `cp.async` instructions. The same
uninstrumented matrix gives 4096/64 C16 **207.33 tok/s** (4939 ms), versus
219.27 for broad synchronous split and 231.57 for the current route. C8 is
192.34 tok/s, and 512/64 C16 is 1117.90 tok/s. No output-length failures occur;
the 3072-token cross-batch divergence remains and is not numerical acceptance.

The first experimental patch accidentally updated only one occurrence of the
capability settings, because `String::replace` replaces the first match. Its
contract-test failure was an experiment-construction error, not a product bug.
The corrected patch updates both occurrences and passes the full suite. Failed
and corrected logs are retained separately. No contract assertion was removed.

This rejects enabling asynchronous split as a sufficient solution. Comparing
transfer tile sizes together with actual residency and partial-kernel counters
is still required; this matrix alone does not establish the hardware stall
responsible for the additional regression.

A separate 32-key double-buffer experiment (per-block candidate budget of
49152 bytes, explicitly not the hardware limit) completes the same matrix.
It reaches 235.29 tok/s at 4096/64 C16 (4352 ms), 218.43 at C8 and 123.08 at C1.
However 512/64 C16 is 1323.00 versus the current 1395.10, so it is not a global
replacement. Long C16 improves 1.61% in throughput while short C16 regresses.
All lengths pass; the 3072-token divergence remains. This establishes that
the selected transfer schedule matters, not that asynchronous execution is
universally beneficial. The experiment is not the subsequent operand-lifetime
compaction, which is validated separately.

The 3072-token cross-batch output difference at token index 31 remains present;
these trials do not close the independent reference-accuracy question.

## Reproduction locations

- Current trace: `/tmp/lunaflux-profile-3b2f25d4.VX7CYp/long-c16/lunaflux` on the GPU host.
- Downloaded SQLite: `/tmp/lunaflux-profile-analysis.MtJqA2/trace.sqlite`, SHA-256
  `39380db6c2c6aff60c10b3cfbb32a8aca57ec9b5abe22f6b3232e0809d66f6cd`.
- Local joined execution tables and JSON are alongside that SQLite file.
- Split experiment: `/tmp/lunaflux-split32-3b2f25d4.S3pPqE`, including source,
  build logs, complete matrix, summary and owned-server shutdown confirmation.
- Split trace: `/tmp/lunaflux-profile-split32.I24Hoq/lunaflux`; downloaded and
  joined at `/tmp/lunaflux-split-profile-analysis.Hkuw1r`.
- Downloaded baseline profile archive SHA-256:
  `f9d742e6d225c84141f22fcaf3a6d0a571b8cb48222565e77d54b62512bb0030`.
- Downloaded split experiment archive SHA-256:
  `9950f4c86c54be92ef645f387ec4d5966a8f6ce4a22a8a00739b50e78b5232e5`.
- Async split experiment: `/tmp/lunaflux-async-split-experiment.vHQziO`;
  downloaded result archive SHA-256:
  `816ccbd80add6d7aa7dc1b1ea09b38f6bbc2e3527752117c379327e85f10c094`.
- Async 32-key experiment: `/tmp/lunaflux-async32-split.nmWfXb`; downloaded
  archive SHA-256 `839223d4d0f58090a2905aaa877353dfb60a16aa137c59b09ff113f90ce1c48e`.

Future selection should compare whole attention routes by workload bucket,
including partial and merge costs, using immutable startup measurements. A
hardware lowering may change freely while respecting numerical semantics;
byte-identical emitted source is not an optimization acceptance criterion.
