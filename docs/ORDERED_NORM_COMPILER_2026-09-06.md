# Ordered norm scheduling experiment: not selected

## Decision

The norm-only rewrite did **not** improve Qwen serving throughput. Both
experimental changes are reverted from the selected compiler path, including
their otherwise-unused portable planner and API. The experiments remain
recoverable in commits `1d7b33d` and `6c96bad`; no production deployment changed.
Fewer barriers or memory instructions alone are not a measured optimization.

The selected implementation remains the `5924c1c` baseline, including the
previously successful functional decode partition/merge optimization. This
report does not undo or attribute that earlier speedup to the norm experiment.

## What was tested

`1d7b33d` added a pure workgroup/subgroup reduction schedule and bounded
producer-value retention, consumed by residual/RMSNorm CUDA lowering. It
preserved the descending pairwise addition tree and BF16 rounding boundary;
it did not assume floating-point associativity. At most 16 rounded values per
lane were forwarded in registers, avoiding rereading residual output. Wider
rows kept the materialized route. Geometry and register budgets belonged to
the backend, not model builders or the scheduler.

`6c96bad` additionally rematerialized the tiny pure shared-memory prefix in
each subgroup. This reduced block barriers to one without regrouping the
addition tree. The six-pointer ABI and 128-thread launch stayed unchanged.
The actual width-1024 Qwen kernel compiled to 32 registers with no spills.

Both variants had mixed/negative small-batch micro results. The final variant
was checked directly against the baseline CUBIN across 18 cases (six active
token counts and three input patterns, including exact residual cancellation).
Residual and norm output bytes matched exactly; inactive rows were untouched.
A separate FP64 norm oracle observed maximum absolute error 0.00781038523
against BF16 output (tolerance 0.016). Memcheck reported zero errors/bytes leaked;
racecheck reported zero hazards. Physical shape coverage here is width 1024,
epsilon 1e-6; broader shape tests were software/source tests only.

Representative CUDA-graph event timings, in microseconds per kernel:

| Active tokens | Baseline | Final experiment | Observation |
| --- | ---: | ---: | --- |
| 8 | 3.24 | 3.29 | Slightly slower |
| 32 | 3.24 | 3.30 | Slightly slower |
| 128 | 3.49 | 3.43 | About 2% faster |
| 256 | 3.65 | 3.56 | About 3% faster |

Graphs amortize host submission over 1,000 launches; these are not serving
rates. Eager launch timings were about 4 microseconds and obscured small
device-time differences. No claim is made that either measurement establishes
the cause of the small regression.

## Same-machine Qwen A/B

RTX 5060 Ti, Qwen3-0.6B BF16, identical input token IDs, fixed output lengths,
greedy, eager execution, c32 capacity. Only the residual/RMSNorm module changed;
runtime, worker, other modules and request configuration stayed fixed.
Each cell is the median of three trials with one warmup each, four requests
at C1 and eight at C8. New ran before baseline; the experiments were isolated,
not interleaved. Differences below one percent are not a robust speedup claim.

| Input / output / concurrency | Baseline token/s | Experiment token/s | Change |
| --- | ---: | ---: | ---: |
| 59 / 256 / 1 | 210.62 | 210.19 | −0.21% |
| 128 / 128 / 1 | 207.59 | 207.62 | +0.01% |
| 512 / 64 / 1 | 165.07 | 165.08 | +0.01% |
| 1528 / 32 / 1 | 81.47 | 81.23 | −0.30% |
| 59 / 256 / 8 | 702.54 | 701.14 | −0.20% |
| 128 / 128 / 8 | 662.07 | 658.62 | −0.52% |
| 512 / 64 / 8 | 384.17 | 383.83 | −0.09% |
| 1528 / 32 / 8 | 109.34 | 109.28 | −0.05% |

All four old/new single-request token streams match exactly. Each C1 variant
has 12/12 requests matching its warmup for every vector. C8 exact agreement
with the single-request reference is 0/24 for 59/256 in **both** variants,
and 24/24 for the other three vectors. This preexisting batch-dependent output
disagreement remains unresolved; warmup agreement is not an independent
quality oracle. No failed/mismatched requests were removed from throughput.

## Refreshed bottleneck profile

A separate Nsight capture of the latest decode baseline used unchanged worker
and kernel binaries. A disposable runtime only forwarded profiler environment
variables through worker exec; that diagnostic modification is not in the
repository or a production artifact. The following are summed kernel durations
per output token for three 59/256 C1 requests after excluding warmup. They are
profile attribution, not unprofiled throughput or additive CPU/API durations.
The vLLM column is the **earlier September 5 capture**, not a new competitor run.

| Operation, ms/output token | Latest LunaFlux | Earlier vLLM |
| --- | ---: | ---: |
| Decode attention, including partial/merge | 0.790 | 0.279 |
| QKV + QKNorm/RoPE/KV-write | 0.969 | ≈0.671 |
| MLP gate/up + down | 1.370 | 1.343 |
| Attention output projection | 0.313 | 0.322 |
| LM-head projection | 0.737 | 0.732 |
| Residual/RMSNorm including final norm | 0.267 | ≈0.058 |
| Greedy reduction core | 0.141 | ≈0.006 |

Decode attention fell from the earlier LunaFlux 1.494 to 0.790 ms/output token;
QKV, norms and greedy are essentially unchanged. The long 1528/32 profile
still spends substantial time in QKV, MLP and prefill/decode attention. Its
prefill costs are amortized over only 32 outputs, not isolated decode latency.

Next compiler work should target the physical schedules of the remaining
attention and QKV bottlenecks, and cross-operation fusion/launch reduction,
not repeat this barrier-only norm rewrite. MLP/LM-head GEMM is already close
on this short-input workload. A portable cost model must consider launch floor,
occupancy, live registers and end-to-end benefit, not merely instruction count.

## Reproduction and checks

- Experiment compiler commits: `1d7b33d`, then `6c96bad`.
- Runtime/worker: unchanged `4cfae98` build used by the latest decode baseline;
  existing attention artifacts and all non-norm modules were reused.
- GPU UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, PCI
  `00000000:17:00.0`; CUDA 13.1.115; `-O3 --fmad=false`.
- Experimental focused tests: 53/53. Final restored fused and IR packages:
  36/36. Formatting and warning-denied native check pass. The full suite was
  not rerun; the separately documented preexisting FP8 fixture failure remains.
- Both Qwen instances drained with `child_exit_code=0`, `child_closed=1`, and
  empty stderr. No competing framework or production service was changed.
- Remote archive: `/dev/shm/lunaflux-norm-results-6c96bad-20260906-r1.tar.gz`;
  SHA-256 `51e6b370801cd9a4ac39983f49c9c717b575c331b4cbe69815c3e12a415802e6`.
  It includes raw trials, token streams, lifecycle logs, norm sources/CUBIN,
  numerical/sanitizer results, experiment source archives and Nsight capture.
- Short profile window: Nsight timestamps `124827340715..128717351329`,
  divided by 768 output tokens. Long window: `129160421130..130409297444`,
  divided by 96. Warmups are excluded from both.
