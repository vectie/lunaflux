# GLM long-prefill investigation — 19 October 2026

## Scope and pre-registered changes

Keep numerical comparison outside the compiler. Do not hash payloads. Both
Sparks retain a 100 GiB process/GPU envelope, no unit swap and at least 16 GiB
host MemAvailable. Production is temporarily stopped for the paired experiment
and must be restored after it. At most three candidate changes in this round.

The historical roughly 11% gap used 22–36 input tokens in a serial resident
diagnostic. The new 5.90x completion-time ratio is a matched one-output control:
7,973 input tokens, LunaFlux 103.887 s versus vLLM 17.622 s, paged HTTP serving.
These are different workloads, not evidence of a sixfold same-workload
regression. Complete-answer and latency results remain in
`GLM_FULL_SERVICE_WIRING_2026-10-18.md`.

The measured-loop plan is: trace actual selected long-prefill work; capture
hardware counters on its generated sources; alter the largest demonstrated
cost through pure physical plans/private lowering; verify propagation,
correctness, memory and unprofiled end-to-end time before promotion.

## Evidence so far (not a completed optimization result)

The exact 5f864415 8K/8-slot instrumented baseline is retained at
`/tmp/lunaflux-glm-http-long-profile-20261019-v1`. Collection drove .179's host
reserve to 12,985,792 KiB; the watchdog stopped both units cleanly. Both units
reported zero swap. This is a failed profiling envelope, not a completed
benchmark. The partial seven-step trace is useful only for the captured early
chunks; it must not be extrapolated as a complete 8K attribution. Its second
through seventh steps show sparse index time increasing from 1.16 to 8.04 ms,
and attention from 11.58 to 34.25 ms (11 sparse layers per step). Experts
dominate those early chunks. Analysis is in
`/tmp/lunaflux-glm-long-trace-analysis-20261019-v1`.

A diagnostic retry retains context 8192, physical pages 513 and prefill rows64
but reserves only one active request slot and two admission lanes. It is not a
change to the public service's concurrency policy.

The generated-source replay at
`/tmp/lunaflux-long-paged-sparse-20261019-v2` uses 64 queries, history8000,
32 index heads x128, 32 attention/KV heads x256, pool4 and top-k2048.
Inputs are synthetic; this isolates implementation cost, not model quality.
It asserts nonempty selections and nonzero finite outputs before timing.
The earlier v1 replay had a harness-only page-table endpoint of513 instead of
512; kernels correctly rejected it. Those zero-work timings are invalid and
retained, not claimed as performance evidence.

| Actual generated leaf | CUDA-event time | NCU time | Warp instructions | Active warps | Long-scoreboard stalls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Index | 61.27 ms | 61.19 ms | 872.27 M | 3.52% | 48.68% |
| Attention | 21.19 ms | 20.54 ms | 1,912.05 M | 97.10% | 45.82% |

NCU used five replay passes per leaf; these are not full-model timings. The
indexer scores in parallel but subsequently scans all candidates on thread0
for every selected pool. Its eligibility is only0.081 warps/cycle. Attention
also repeats quadratic duplicate checking per head and has strided score
loads; these are separate candidates, not yet causal attributions.

The deployed vLLM source, copied without launching GPU work, is retained at
`/tmp/lunaflux-glm-long-indexer-vllm-20261019.py`. It caches completed pooled
keys, skips index scoring when all keys are selected, computes long-history
scores using DeepGEMM, and calls parallel per-row top-k. Local `../vlm`'s
`csrc/libtorch_stable/sampler.cu` shows block-wide histogram/selection and
sorting. These are implementation references, not correctness oracles.

## Candidate 1: stable parallel selection

`RetainedIndexPlan` chooses an immutable physical selection schedule. Independent
candidate ranks count all valid higher scores (ties by lower candidate ID).
This preserves the existing ordered score computation and selected order,
including infinities, signed-zero ties and ignored NaNs. A work bound chooses
cooperative repeated maxima for very large domains so quadratic ranking does
not become an unbounded replacement for top-k. Both paged and contiguous
lowering consume the same plan. Scratch capacity, external ports, numerical IR
and effect order are unchanged. No model-specific shape exception is added.

Required gates before selection: actual-source old/new index and output
comparison across history/tails; paged generation/reuse and malformed-list
regressions; memcheck/leaks, race/sync/init checks; selected-source counters;
commit-pinned full-model long/short answer and timing replay. No candidate is
promoted merely because this source exists.

Candidate1's full64-query history8000 source replay reduces index61.24 ms to
7.65 ms (8.0x). Its 29-query tail reduces57.95 ms to4.32 ms (13.4x); these
are different row counts, not a14x full64-query result. All selected indices
and attention outputs match at histories64/256/2048/4096/8000/7997. Full64
query memcheck/leaks, racecheck, synccheck and initcheck report zero errors.
NCU instructions fall872.27M to212.21M; active warps rise3.52% to12.07%.
There is no end-to-end claim yet.

The one-slot Nsight retry also exceeded the reserve when collection started
(13,507,140 KiB available), and the watchdog retired the pair. Do not repeat
this unsupported full-resident profiling setup or present its interrupted
request as an unprofiled timing. Kernel probes remain independently bounded
to2 GiB, zero swap. Full-model performance tests will run without collection.

## Candidate 2: exact bounded membership

The attention physical schedule selects a bounded bitmap when its windowed
work is lower than pairwise duplicate checking. It uses at most one32-bit
membership word per logical worker, retires readers before reusing a window,
and still rejects duplicate or malformed selections. This is not a trusted
input shortcut, a changed numeric law, or an attention approximation. At8K,
one1KiB bitmap replaces the quadratic per-head scan. The unchanged independent
score/value reductions remain an acknowledged separate cost. Compare this
candidate against the retained candidate1 source, not only the original.

Candidate2 (`/tmp/lunaflux-long-paged-sparse-20261019-v4`) is exact across the
same six histories, with zero errors in all four sanitizer tools and leak
check. Instructions fall1,912.05M to800.69M, but standalone time improves only
slightly (roughly21.05 to20.39 ms): load waiting, not just instruction count,
still limits this leaf. No instruction-count percentage is called a speedup.

## Candidate 3: cooperative score-operand copies

The physical IR now represents a bijective cooperative key-component tile.
CUDA lowering copies contiguous key components into swizzled shared storage,
publishes the copy, executes each key's unchanged ordered component fold, and
retires all readers before reuse. It retains the original scalar schedule for
insufficient visible keys and unsupported tile domains. The new layout does
not change score/probability arithmetic, input precision or selected order.

`/tmp/lunaflux-long-paged-sparse-20261019-v5` compares against candidate2:
attention at history8000/full64 queries improves approximately20.38 to14.40 ms
(29% less time); all output/index cells still match at six histories. Both
stable-selection schedules additionally pass explicit NaN/infinity/signed-zero,
tie, invalid-candidate and nonmultiple-length controls. Full64-query memcheck,
leaks, racecheck, synccheck and initcheck pass. The early64-history leaf regresses
roughly1.03 to1.23 ms because the combined kernel still reserves shared resources
even when its scalar branch runs. This must remain visible in the full-model
short/long retest; no universal speedup is claimed.

The three changes together reduce the late64-row index+attention pair from
about82.4 to22.1 ms in this synthetic exact-source replay. This is not yet an
end-to-end service result. No new candidate variants will be added before the
matched service retest in this round.
