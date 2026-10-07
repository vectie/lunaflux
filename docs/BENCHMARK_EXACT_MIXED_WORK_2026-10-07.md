# Exact mixed-work attribution and rejected score-storage experiment

The remaining 8K/C8 gap is **not one 493 ms decode-only penalty**. Fresh
logical-row markers match all 96 LunaFlux and vLLM steps, and show that the
reference split-KV kernel handles multi-query and single-query rows together.
Classifying its name as decode-only misattributes the work.

The measurement/probe fixes below are retained. The experimental compiler
rewrite was tested, did not improve the target workload robustly, and was
removed from production source. Its reproducible patch is retained offline.
There is **no new serving speedup or production promotion** in this report.

## Workload and controls

- Qwen3-0.6B BF16, 8192 input / 64 output tokens, eight concurrent requests.
- Spark .179, GB10 sm121, 48 SMs, UUID
  `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, PCI `0000000F:01:00.0`.
- Same frozen model, runtime, kernel bundle and repaired route table from
  [mixed calibration repair](BENCHMARK_MIXED_CALIBRATION_REPAIR_2026-10-07.md).
- LunaFlux and vLLM both execute 66,040 query tokens: 65,536 in multi-query
  rows and 504 in one-query rows; both have 272,613,120 causal QK pairs per
  query head. All 96 sorted query/history vectors match. No orphan kernels.
- SGLang uses a different prefill chunk policy: 72 steps, 66,048 query
  tokens, 272,679,168 QK pairs. Its capture is retained, but individual steps
  are not claimed to be equal-work matches.

Only 34 of 96 steps are steady eight-row decode. There are 33 prefill/mixed
steps and 29 other single-query steps while the batch shrinks. The mixed
domain is not always one long row plus decode rows. One observed vector is:

```
query = [79, 1963, 1, 1, 1, 1, 1, 1]
past  = [8113, 0, 8211, 8207, 8203, 8199, 8195, 8216]
```

CPU-side diagnostic markers do not read GPU metadata back. They are confined
to copied diagnostic workers/reference launchers, not production execution.
Nsight capture times are not substituted for unprofiled throughput.

## Corrected attribution

These are sums of GPU kernel durations over matching logical phases. They
are **not exclusive wall-time contributions**, and cannot be subtracted from
an unprofiled result captured in another run to infer a CPU bubble.

| Logical phase / work | LunaFlux ms | vLLM ms | Difference ms |
| --- | ---: | ---: | ---: |
| Initial prefill attention | 113.09 | 116.02 | −2.93 |
| Initial prefill non-attention | 160.11 | 149.40 | +10.71 |
| Mixed attention, complete chain | 1319.70 | 1191.53 | +128.17 |
| Mixed non-attention | 1176.87 | 1025.87 | +151.00 |
| Single-query attention, all row counts | 1591.55 | 1520.74 | +70.81 |
| Single-query non-attention | 426.49 | 391.92 | +34.56 |

The total kernel-activity difference is 392.32 ms: 196.05 ms attention and
196.27 ms non-attention. Mixed steps account for 279.17 ms of that difference.
This rules out treating a single ordinary-C8 decode experiment as coverage
of the entire remaining gap.

For multi-query steps, reference projection roles were checked against the
pinned sequential Qwen attention/MLP source order, attention and SiLU anchors,
and complete four-projection-per-layer cycles. Fused and separate postops
are compared as chains, not omitted from the reference:

| Multi-query projection chain | LunaFlux ms | vLLM ms | Difference ms |
| --- | ---: | ---: | ---: |
| QKV + QKNorm/RoPE/KV write | 383.85 | 375.97 | +7.88 |
| Output projection | 163.26 | 105.08 | +58.19 |
| Gate/up + activation | 456.91 | 423.09 | +33.82 |
| Down projection | 210.45 | 130.86 | +79.60 |

These are nested subsets, not extra contributions to add to the table above.
The order validator accepts 92 steps; four final C1 steps use GEMV symbols
outside its supported matrix-cycle matcher and remain explicitly unresolved.
All multi-query steps used in this table pass the source-order check.

## Probe repairs

`selected_policy_probe.cu` previously sized a mixed split-decode companion
from the **query-token bucket**. The actual graph uses the **request-row
bucket**. A three-row mixed step must replay a four-row split grid, not a
32-row grid derived from its 2048-token prefill budget. Ordinary companions
retain their admitted envelope; this is not an across-the-board grid shrink.

The probe now accepts exact per-row query/history vectors, validates them
before CUDA allocation, and supports split-to-split replacement comparisons.
The row parser checks malformed input, phase roles, page capacity and integer
overflow. The trace summarizer no longer infers a reference invocation's
logical phase from `flash_fwd`/`BatchPrefill` names.

All 29 observed mixed vectors were replayed with whole attention-chain timing,
full output comparison, a sampled FP64 oracle and unchanged KV checks. The
ordinary-versus-split choices vary by row/history distribution; another global
switch is not supported by these measurements.

## Score-register experiment: resource limit improved, target time did not

This extends the earlier [register-exchange experiment](BENCHMARK_DECODE_REGISTER_EXCHANGE_2026-10-07.md)
with a typed physical ownership plan and a compact shared-memory reservation.
It is not a newly discovered general solution to the same transport problem.

The pure plan maps each logical key to a dot-owner lane and register slot.
The CUDA backend realizes transport using subgroup shuffles. Arithmetic,
reduction trees, F32 probability precision, ascending PV order and independent
K/V refill effects remain unchanged. Unsupported subgroup/tile mappings fail
closed. No model-name test or request-path tuning is introduced.

Removing the score scratch reduces the live reservation from 33,040 to
32,768 bytes. On this device the shared-memory block limit rises from two to
three. The first emitter accidentally omitted the complete-tile fast path:
it increased instructions 15.2% and was rejected. Restoring full/ragged domain
specialization and calculating fragments before exchanging them fixes that
regression, but does not produce a robust 8K/C8 win.

Fresh ordinary-C8 hardware replay, identical grid 8 × 8 and block 64:

| Metric | Selected control | Corrected compact experiment |
| --- | ---: | ---: |
| Registers/thread | 148 | 137 |
| Shared bytes/block | 33,040 | 32,768 |
| Shared-memory resident-block limit | 2 | 3 |
| Theoretical occupancy | 8.33% | 12.50% |
| Achieved occupancy | 5.73% | 5.76% |
| Warp instructions | 41,381,632 | 42,361,984 |
| Branch instructions | 366,208 | 398,976 |
| Eligible warps/scheduler | 0.12 | 0.12 |
| No eligible warp | 87.90% | 87.68% |
| Register spills | 0 | 0 |

Only 64 CTAs exist for 48 SMs. Increasing the per-SM residency ceiling does
not create extra independent work, and the extra indexed shuffles still
increase executed instructions by 2.37%. L1TEX dependency waits remain the
largest reported stall class, 32.8% versus 30.5% of average warp issue latency.
These stall shares are not percentages of request wall time. Clocks were not
locked; paired unprofiled samples, not profiler timing, decide acceptance.

Selected examples from five-pair trials (independent medians shown; acceptance
uses the minimum paired gain, requiring 2% in every pair):

| Work / chain | Control µs | Compact experiment µs | Decision |
| --- | ---: | ---: | --- |
| 128 keys, C8 ordinary | 10.266 | 9.737 | improved |
| 8192 keys, C1 ordinary | 325.809 | 265.308 | improved, **not selected serving chain** |
| 8192 keys, C1 split + merge | 114.333 | 113.788 | inconclusive |
| 8192 keys, C8 ordinary | 1153.531 | 1167.099 | regression |
| 8192 keys, C8 split + merge | 1176.818 | 1168.476 | inconclusive |
| 32768 keys, C1 ordinary | 1331.231 | 1061.374 | improved, **not selected serving chain** |
| 32768 keys, C1 split + merge | 596.090 | 590.524 | no robust gain |

The final exact-vector split-to-split replay covers all 29 mixed steps and
also shows no consistent chain gain. For example, the eight-row/two-prefill
step is 1402.94 → 1416.96 µs. All replacement outputs are bitwise equal.
The retained raw log's fixed `old_launches=2` label is stale for this mode:
both sides execute three launches (prefill, split decode, merge). The probe's
label now derives counts from the loaded chains; measured dispatch was correct.
No new serving bundle or end-to-end claim is justified by these results.

## Validation, retained code and next target

- Experimental physical-IR/source tests: 128/128 with the existing toolchain
  compatibility warning exclusions `-79-20-29-25`; no full-repository pass claimed.
- Both experimental versions: 24 workload/reservation cells each, five paired
  trials, mandatory bitwise equality, unchanged 0.003 sampled-oracle tolerance.
- Final mixed replacement: 29 vectors × five pairs, bitwise equality.
- Compact ragged split memcheck: zero errors and zero leaked bytes.
- Geometry/parser tests pass Clang warnings-as-errors and ASan/UBSan.
- GPU work serialized; probe/compiler/controller caps 8 GiB, serving 64 GiB,
  bridge 2 GiB, zero process swap, at least 32 GiB host-memory reserve.

The rejected compiler patch is
`benchmarks/gpu_pipeline/experiments/register_score_residency_20261007.patch`.
Apply only in a disposable checkout for reproduction; the export helper
requires it. Production compiler files and serving defaults are unchanged.
Removing the experiment from the active source does not delete its evidence.

The next implementation targets supported by this capture are the complete
mixed-attention chain and multi-query down/output schedules. Any proposal
must reduce their exact traced work time, not merely raise a residency ceiling,
remove shared accesses or accelerate an unselected C1 path. A causal fraction
of end-to-end improvement still requires a fresh alternating serving test.

The last verified unprofiled 8K/C8 rate remains **102.98 output tok/s**,
versus historical matched vLLM/SGLang **113.44/110.91**: completion-time
overheads **10.16%/7.70%**. These rates were not remeasured here. Previously
reported batched output-vector repeatability remains open.

## Evidence

Remote root: `/home/wlc004s/lunaflux-mixed-work-20261007.NnA6SsKN`.
Contains all three fresh framework traces, exact logical-vector ledgers,
source-bound projection mapping, replay commands/results, both rejected
experiment versions, CUDA modules, Nsight reports and sanitizer log.

Archive: `/home/wlc004s/lunaflux-mixed-work-20261007.NnA6SsKN-sealed/gap-repairs.tar.gz`.
SHA-256: `1a0bd3e6de7a5cee6c269e9a66ea25e0729ec1f9a6ade27120e7aa3667383fd3`.
Local directory: `/tmp/lunaflux-mixed-work-20261007.i8H99Gk4/`.
The archive SHA-256 and all 5,829 manifest entries were verified locally in
`/tmp/lunaflux-mixed-verified-20261007.C7AcGJI4/` after path/link checks and
non-overwriting extraction. Build/dependency caches and other
explicit exclusions are listed separately. Frozen source, logs and artifacts
were not overwritten. Preparation failures are retained and not counted as
successful GPU experiments.
