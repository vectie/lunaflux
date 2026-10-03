# Dual score decode domain specialization and serving results

The compiler now specializes complete KV tiles in the dual-score decode fold,
and the serving exporter can bind that numerical law explicitly. Fresh Spark
serving runs improve C8 completion time by 1.95–4.04%. C16 improves only
0.03–0.36%, which is not a meaningful demonstrated gain. This repair does not
close the remaining vLLM/SGLang gap.

This follows the [operand-copy repairs](BENCHMARK_COMPILER_COPY_REPAIR_2026-10-03.md).
No new reference-framework run or production deployment was performed. Earlier
reference timings must not be combined with these samples to claim a current
cross-framework ratio.

## Compiler change

The existing immutable `BlockwiseFold` supplies the KV extent, score ownership
and numerical law. For more than one score owner, CUDA lowering emits a complete
domain arm with statically unrolled QK and PV loops. That arm removes key-bound
predicates that the full-tile condition makes redundant. The ragged arm keeps
the original bounds and ascending-key fold. Both arms share the same per-key
expressions. No fast math, contraction or arithmetic reassociation is added.

This is a pure domain specialization followed by device lowering, not a Qwen
condition, runtime compiler, extra IR layer or token-step validation. Single
score-owner schedules retain their existing iteration strategy: applying the
same unrolling there was measured and rejected. Regression tests cover source
generation for families 450–457 and require both the complete-domain and guarded
tail behavior appropriate to the physical ownership plan.

The implementation is in
[source_blockwise_fold.mbt](../kernels/luna_cuda_attention_tile_source/source_blockwise_fold.mbt),
with [regression tests](../kernels/luna_cuda_attention_tile_source/complete_key_domain_wbtest.mbt).
The compiler lowering file used by the serving treatment matches the committed file:
SHA-256 `01db264c01a256a785cb4b2238f3ed018712e0d093256168f58759d2c6bd9f68`.

## Serving propagation repairs

Dual-score execution already existed as a compiler alternative, but the reusable
serving exporter did not expose its own ABI contract. The repair adds
`--decode-dual-score-blockwise-f32-v2`, the
`paged-attention-readonly-dual-score-blockwise-f32-decode-production-v10` ABI,
and bundle schema `lunaflux-reusable-fused-runtime-bundle.v9`. Admission,
materialization, split-decode classification and bootstrap consume the same
law and function identity. Strict and FMA options remain distinct and mutually
exclusive. Default strict behavior is not silently changed.

The dual-score law changes QK component association relative to strict
single-owner decode. Consequently, the isolated compiler comparison below is
same-law and bitwise checked, while the serving comparison is an explicitly
declared strict-to-dual alternative. Matching model outputs in the tested
requests does not establish universal intermediate bitwise equivalence.

The serving preparation helper also had an independent packaging bug. Its
saved exporter arguments omitted the device target, so a rebuilt exporter
defaulted to 12.0 although the admitted execution manifest specified 12.1.
The worker correctly rejected that mismatch. Preparation now obtains the
target from the execution manifest, supplies it to the exporter, and checks
the emitted target. It does not weaken worker admission or patch production
identity to make a diagnostic start.

Controlled preparation now reuses the exact native worker and launcher binaries
between arms. Benchmark unit names incorporate their full campaign path to
avoid collisions with preserved transient units. Failed units terminate the
readiness check promptly. The workload vector includes C8 and C16 rather than
testing only C16.

## Isolated kernel measurements

Hardware is DGX Spark GB10, sm121, 48 SMs, UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`. Compilation uses CUDA 13.0.88 and
MoonBit 0.1.20260920. These are ordinary CUDA-event medians of five alternating
old/new samples, not profiler replay times. History is preceding tokens, not
launch capacity. The probe uses fragmented physical page IDs, identical
operands, and a separate scalar attention oracle.

| Dual-score schedule | Request rows | History | Before µs | After µs | Less time |
| --- | ---: | ---: | ---: | ---: | ---: |
| KV32, eight partitions | 1 | 4096 | 94.052 | 57.375 | 39.00% |
| KV32, eight partitions | 8 | 4096 | 643.700 | 597.520 | 7.17% |
| KV32, eight partitions | 16 | 4096 | 1196.381 | 1176.722 | 1.64% |
| KV64, eight partitions | 1 | 4096 | 163.930 | 90.011 | 45.09% |
| KV64, eight partitions | 8 | 4096 | 882.287 | 598.746 | 32.14% |
| KV64, eight partitions | 16 | 4096 | 1734.423 | 1206.128 | 30.46% |

The changed dual-score variants cover 30 head-dimension-128 cells: KV32/KV64,
request rows 1/8/16, and histories 126/127/4095/4096/8191.
Same-law old/new results are bitwise equal
and pass the independent oracle. Memcheck, racecheck and synccheck pass for
both variants. Explicit full leak checks report zero leaked bytes. The KV64
improvement is relative to its own slower previous implementation; it does not
prove KV64 is faster than the optimized KV32 schedule or justify selecting it
universally.

An additional 30-cell FMA single-owner campaign passes numerical and sanitizer
checks. This is control coverage, not an FMA optimization: single-owner plans
do not select the new complete-domain arm. Timing variation in those cells is
retained and is not attributed to this repair.

## Ordinary serving measurements

The arms use identical worker binaries and identical AOT modules 0–7. Only the
decode module and its declared ABI differ. Module-scoped route timings are
remeasured for each arm, so this is a package-and-selection comparison rather
than a fixed-route kernel-only intervention. Both arms retain the same forced
mixed-decode diagnostic dispatch; no production route admission is fabricated.

Baseline/treatment/treatment/baseline order gives two fresh starts per arm.
Each cell has one measured trial after warmup. Inputs are literal matched token
IDs; all measured requests produce the required output count and identical
output token sequences. Runtime stderr is empty and workers drain successfully.

| Input / output / concurrency | Before samples ms | After samples ms | Before median ms | After median ms | Less time | Output tok/s before → after |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 4096 / 64 / 8 | 2647, 2643 | 2594, 2593 | 2645.0 | 2593.5 | 1.95% | 193.57 → 197.42 |
| 4096 / 256 / 8 | 7590, 7584 | 7282, 7279 | 7587.0 | 7280.5 | 4.04% | 269.94 → 281.30 |
| 4096 / 64 / 16 | 4742, 4750 | 4726, 4732 | 4746.0 | 4729.0 | 0.36% | 215.76 → 216.54 |
| 4096 / 256 / 16 | 12965, 12954 | 12941, 12971 | 12959.5 | 12956.0 | 0.03% | 316.06 → 316.15 |

These small sample sets have no confidence interval. The long-output C16 ranges
overlap; its result is effectively neutral. Minimum sampled MemAvailable is
104,514,008 KiB, approximately 99.67 GiB, above the 32 GiB reserve. GPU workloads
were serialized; diagnostic containers use an 8 GiB limit without swap and
the existing serving harness uses a 64 GiB limit.

The worker SHA-256 is
`e52e695b8320d4d29d8fce43611e1e6ac3932e31e530dd8cf11c3dbd629a8675`.
Control decode module SHA-256 is
`4ac6d5f35ae8003b9eb218a40d528106acd6379d44133fa3a322fa3c4fbcae25`;
treatment is
`462ebdcff2374a5ccbbb90a6d349601f8b1bce4ba28239319d283305e995aa79`.

## Executed selection and remaining hardware limits

A fresh Nsight Systems trace uses the same treatment worker and AOT artifacts.
Only its separate diagnostic launcher preserves profiler injection. The trace
contains 224 ordinary dual-score decode calls, 4,368 partitioned partial calls,
and 4,368 merge calls. The ordinary and partial functions both report 92
registers per thread and 33,040 bytes dynamic shared memory. The profile drains
with `child_exit_code=0`, `child_closed=1` and empty worker stderr. The new module
therefore reaches actual serving execution, not just export or an isolated probe.

Those call counts and durations include warmup and measurement. Partitioned
partial activity totals 4,069.223 ms and merge activity 23.446 ms in that aggregate
capture. These locate continuing activity in the partial fold, but are not an
exclusive wall-time attribution or a measured-only cross-framework comparison.

The same-law KV32 C16/history4096 Nsight Compute pair records:

| Metric | Before | After |
| --- | ---: | ---: |
| Executed warp instructions | 62,037,504 | 46,967,552 |
| Registers per thread | 84 | 92 |
| Allocated registers per thread | 88 | 96 |
| Shared-memory residency limit in blocks per SM | 2 | 2 |
| Average active warps per SM | 3.78 | 4.00 |
| SM issue activity as percent of sustained elapsed peak | 10.46% | 8.05% |
| Local spilling requests | 0 | 0 |
| Source-correlated excessive shared wavefronts | 0 | 0 |
| Long-scoreboard stalls per active issue | 2.48 | 5.80 |
| Barrier stalls per active issue | 0.36 | 1.47 |
| Profiler invocation duration µs | 1245.024 | 1224.736 |

Instruction count falls 24.29%, but elapsed time falls only 1.63%. The selected
shared footprint still limits residency and the kernel still has substantial
load, short-scoreboard and publication waits. Stall ratios change denominators
when fewer instructions issue; they must not be added as wall-time percentages
or interpreted as proof that absolute barrier time quadrupled. Aggregate
hardware bank metrics remain nonzero despite zero source-correlated excessive
wavefronts. No DRAM-byte capture was collected for this pair, so it does not
establish a bandwidth roofline.

The remaining engineering target is to improve operand availability and
residency without introducing extra transfers, synchronization or arithmetic
work. The measurements do not support another claim that one pipeline edit
will necessarily close the whole framework gap.

## Rejected experiments

Joining K/V readiness into one publication slowed the tested long C8 kernel
about 4.4%. Retaining a larger page-identity window slowed tested cells roughly
2–5%. Narrowing proven affine offsets gave mixed timing and increased total
instructions. Unrolling all single-owner complete folds regressed long C8/C16;
a PV-only version also regressed them. These experimental lowerings were removed.
Their logs, sources and counters remain preserved rather than relabeled as wins.

## Validation and reproduction

The warning-denied native check passes. A clean checkout of compiler commit
`cfd1f794` passes the affected 308 tests without the unrelated dirty working tree.
Its first full-suite run passes 3,293/3,294; the unrelated online TCP zero-wait
socket test times out. No socket code or timeout is changed to suppress it.
The focused socket rerun passes 54/54, and the cached clean full-suite rerun
passes 3,294/3,294. Formatting passes, including the separate snapshot whitespace
normalization.
The earlier full working-tree suite passed 4,278/4,278. Regression coverage
includes ABI downgrade rejection, mutually exclusive exporter flags, all Qwen
layer spans, wrong symbols and complete versus tail source generation.

Implementation commits are `b3973a0d` for serving propagation and `cfd1f794` for
the fold specialization. Benchmark helpers are committed as `b556c525`.
All first-party automation remains MoonBit `.mbtx`. The preparation helper's
self-test covers explicit device targets 12.1 and 8.9; the measurement helper's
self-test and the real ABBA summarizer pass.
Two pre-existing snapshot blank lines are normalized separately for the current
formatter; the snapshot values and generated CUDA are unchanged.

Primary remote roots are:

- `/home/wlc004s/lunaflux-complete-domain-20261003.CFZEWbXv` for same-law KV32 timing and counters.
- `/home/wlc004s/lunaflux-wide-dual-domain-20261003.I1MNtUrN` for KV64 coverage.
- `/home/wlc004s/lunaflux-contracted-domain-20261003.FpiHwluu` for unchanged FMA controls.
- `/home/wlc004s/lunaflux-dual-serving-20261003.jTrfnSvt/abba-target` for valid serving timing.
- `/home/wlc004s/lunaflux-dual-serving-trace-20261003.rfrla3gW` for selected execution.

Earlier target-mismatch starts and a stale-unit collision are retained as failed
diagnostics, not valid timing trials. The repaired target-specific arms are
`control-target` and `treatment-target`; their ABBA result is `abba-target`.

The compact archive preserves successful and failed campaigns, numerical checks,
sanitizer logs, counters, traces and artifact recipes. Its inventory explicitly
excludes disposable build/source and large serving materializations; original
remote roots are preserved. Remote archive:
`/home/wlc004s/lunaflux-dual-domain-archive-20261003.XSdYyARC/sealed/gap-repairs.tar.gz`.
Downloaded archive:
`/private/tmp/lunaflux-dual-domain-20261003.OrK3jMm4/gap-repairs.tar.gz`.
The local hash matches the remote SHA-256:
`4b0c572911025dd9d100951a067e39958739168e234d0a8bb9fe53cf0727cd22`.
