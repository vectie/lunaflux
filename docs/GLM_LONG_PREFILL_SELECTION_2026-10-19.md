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

## Initial evidence and failed profiling attempts

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
indexer scored in parallel but subsequently scanned all candidates on thread0
for every selected pool. Its eligibility is only0.081 warps/cycle. Attention
also repeats quadratic duplicate checking per head and has strided score
loads; these were separate candidates, tested individually below.

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
The end-to-end results below are separate evidence, not inferred from this leaf.

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

## Executable integration and validation

Implementation `e4228c73` and pinned-toolchain syntax repair `13a9ff61` are
pushed. The first ARM build rejected the current formatter's new inclusive-range
spelling; preserving the existing pinned spelling repaired it without changing
generated CUDA. The successful clean ARM build is
`/tmp/lunaflux-checkpoint-build-glm-long-selection-20261019-v2` (1.3 GiB
reported peak, zero swap). Full-model AOT at
`/tmp/lunaflux-glm-http-long-selection-20261019-v1` contains both new selection
and cooperative-copy paths in all 11 sparse layers. It keeps context8192,
eight resident slots,16 admission lanes,513 physical pages and64 prefill rows.

The existing paged sparse mixed-request/reorder/generation-reuse fixture passes
run and all four sanitizer tools at
`/tmp/lunaflux-paged-sparse-long-tile-20261019-v1`. The independent retained
sparse serial-reference suite and single-key analytical controls pass at
`/tmp/lunaflux-glm-sparse-retained-long-tile-20261019-v1`, including malformed
selection lists and odd shapes. No output tolerance or reference was changed.

Native check and all5301 tests pass with the repository's existing migration
warning exclusions (`-20-25-29-35-79-92`). The unexcluded warning-denied check
and test commands were also attempted: both stop on49 existing migration
diagnostics, including unqualified tests in `compiler/parallel_ir` and derived
methods in `runtime/monotonic_clock`. This is not an unexcluded clean-suite
claim; unrelated migration files and warning policy are unchanged.

The native one-generated-token campaign is complete at
`/tmp/lunaflux-glm-service-benchmark-long-selection-latency-20261019-v1`.
It reports actual prompt counts165/2085/6181/7973, zero cached tokens, and times
1.281/13.452/42.323/55.779 seconds. The previous5f864415 times were
1.289/17.599/71.979/103.887 seconds. Thus the8K control improves46.31%,
not merely a standalone-kernel gain. One-token truncations are not answer
quality evidence. Complete-answer/concurrency checks and a fresh vLLM run
are recorded separately below.

Native complete-answer C1 times at the same four lengths are
1.978/14.391/43.639/57.234 seconds, all10 generated tokens, exact expected
labels and natural stops. The previous5f8644158K answer took110.694 seconds
with the same10 outputs:48.30% less completion time. Native short165-token
cohorts C1/C4/C8/C16 take1.948/5.806/10.145/19.991 seconds. All29 cohort
answers pass; generated counts vary6–10, so cohort timing is not a fixed-output
kernel attribution. In particular, the earlier C8 cohort generated130 tokens
versus68 now, so its16.256 to10.145 second change is not a pure implementation
speedup. C16 generated144 in both observations. With the four long-vector C1
cases, all33 answers pass.
Roots end in `long-selection-answers-20261019-v1` and
`long-selection-short-20261019-v1` under `/tmp/lunaflux-glm-service-benchmark-`.

Unprofiled native observation samples minimum host availability
17,363,028/19,071,072 KiB (16.56/18.19 GiB), peak GPU processes97,951/97,979 MiB
(95.66/95.68 GiB), and zero owned-unit swap. Both owners retire on the requested
stop; the watcher reports its existing one-rank-exited condition during this
intentional paired shutdown, not a memory violation or failed request. These
successful unprofiled limits do not erase the earlier profiling failures.

## Interpretation and still-missing implementation work

The prior short CLI comparison was not evidence of long-context parity. It
used22–36 input tokens and a smaller resident geometry; the new comparison
exercises8K paged HTTP serving. Our previous broad parity inference missed
this coverage gap. These measurements demonstrate a long-prefill algorithm
problem; they do not demonstrate a sixfold same-workload compiler regression.

Do not conclude that instruction count alone explains this result. Bitmap
validation removes58% of the leaf's instructions but barely changes time.
The subsequent cooperative copy increases instructions versus bitmap-only
(800.69M to1,737.42M) while reducing NCU time20.53 to14.52 ms. Its L2 read
sectors fall168.55M to145.37M. Long-scoreboard percentage rises51.18% to61.79%
even as elapsed time falls; an averaged stall percentage is not an additive
wall-time attribution. Controlled source changes and unprofiled full requests
establish the benefit, not a single counter.

vLLM still has executable work reductions that this three-candidate round
does not implement: completed pooled keys are cached, all-selected short
contexts bypass index scoring, long index logits use matrix kernels, and sparse
MLA uses a dedicated backend. LunaFlux still recomputes completed pools and uses
ordered scalar score/value folds. Its expanded BF16 history also differs from
vLLM's compressed FP8 MLA cache. These are source-confirmed differences, not
measured percentages of the remaining end-to-end gap. The interrupted seven-step
trace cannot support a complete8K breakdown; do not extrapolate it as one.

Also avoid misreading service graph telemetry: `cmd/glm_service/worker.mbt`
labels a step a hit only when graphs are enabled and `prefill_count()==0`.
Prefill therefore increments misses by this classification; that is not a
hardware count proving every constituent operation ran without a graph. The
partial actual prefill trace contains constituent graph launches. This counter
alone cannot explain or quantify the remaining performance gap.

The changes belong in existing pure physical plans (stable selection,
membership layout and cooperative operand ownership), consumed by private CUDA
lowering. Numerical expressions and model semantics are unchanged. External
counter, reference and answer evaluation remain outside the compiler. More IR
layers alone would not repair the previously serial executable schedule.

## Fresh matched vLLM retest

The reference pair was loaded afresh from the same cached vLLM image and NVFP4
checkpoint, TP2, context8192, maximum sequences8,64-token prefill chunks,
256MiB KV and capture maximum8; prefix caching/speculation remain disabled.
The root is `/tmp/lunaflux-reference-bench-glm-long-selection-20261019-v1`.
The first complete one-token vector warms all contexts; the second vector is
the reported warm comparison. Startup and graph capture are excluded.

| Actual input tokens, one output | Previous LunaFlux | New LunaFlux | Fresh warm vLLM | New/reference |
| ---: | ---: | ---: | ---: | ---: |
| 165 | 1.289 s | 1.281 s | 0.813 s | 1.58x |
| 2,085 | 17.599 s | 13.452 s | 6.792 s | 1.98x |
| 6,181 | 71.979 s | 42.323 s | 14.128 s | 3.00x |
| 7,973 | 103.887 s | 55.779 s | 17.656 s | 3.16x |

Each table cell is one request, not a statistical confidence interval. The
fresh reference warmup vector is0.847/6.805/14.196/17.730 seconds. Complete
8K answers are57.234 seconds native versus18.205 seconds reference, both10
outputs and the exact expected label; the previous native result was110.694
seconds. Thus the control and complete-answer results independently show a
large real gain, while the remaining roughly3.1x gap is explicitly not solved.
All four native and reference C1 complete answers pass. Shorter answer output
counts can differ, which is why the one-token table is the fixed-output control.

Fresh external request reports are
`/tmp/lunaflux-glm-service-benchmark-vllm-long-selection-latency-20261019-v{1,2}`
and `vllm-long-selection-answers-20261019-v1` under the same benchmark prefix.
Comparison reports are `/tmp/lunaflux-glm-long-selection-latency-comparison-20261019-v1.json`
and `/tmp/lunaflux-glm-long-selection-answers-comparison-20261019-v1.json`.
They check literal serialized prompts, expected labels, output budgets and
actual token counts. They do not import into the compiler or assert that the
reference implementation defines correctness.

Fresh short165-token cohort times for C1/C4/C8/C16 are native
1.948/5.806/10.145/19.991 seconds versus vLLM1.130/4.632/9.547/18.474 seconds.
All29 answers in each engine pass, but respective output totals are
10/40/68/144 versus6/27/59/109. These are user-visible cohort timings, not
equal-generated-work throughput. Native sampled peak active requests reaches5,
whereas the memory-constrained reference reaches1; neither is a measured
per-step GPU batch size. Preserve the bounded-memory setup when interpreting
these results. The fresh reference memory minima are17,630,660/18,533,892 KiB
(16.81/17.68 GiB), GPU peaks94,546/94,544 MiB (92.33 GiB), and its containers
use equal memory/memory-swap ceilings (no additional swap). Both reference
owners retire and both GPUs are observed empty before native restoration.

## Live restoration and public acceptance

The validated13a9ff61 executable/AOT pair is installed in durable
`glm53-lunaflux-13a9ff61` roots on both Sparks. The prior5f864415 release and
unit definitions are retained for rollback. LunaNexa route metadata identifies
the new native revision; NVFP4, TP2,8K context,eight resident slots and16
admission/preparation lanes are unchanged. No comparison code enters the engine.

Public company-route acceptance at
`/tmp/lunanexa-glm-public-long-selection-20261019-v1` passes arithmetic,
multi-turn roles, complete Chinese SSE/usage/finish, two simultaneous requests,
disconnect after a real content token and successful subsequent reuse. Its
literal7,973-token public prompt returns `orchid-70`,10 output tokens and a
natural stop. `FINAL_PUBLIC_ACCEPTANCE.json` records8 admissions,7 completions,
one cancellation, zero failures/deadlines/worker restarts, zero queued/active
requests and zero used KV pages after retirement. All513 pages are free.

Live observation at `/tmp/lunanexa-glm-long-selection-observation-20261019-v1`
records minimum host availability17,406,016/19,067,168 KiB (16.60/18.18 GiB),
GPU peaks97,951/97,979 MiB,100GiB unit ceilings and zero unit swap. The bounded
acceptance observer ends without stopping the persistent serving pair. The
temporary local company credential is deleted after these tests. No payload
hashing or integrity scan was performed.

Outcome: the demonstrated long-prefill bottlenecks are materially reduced and
the tested build is online, but long-context parity is **not** achieved. Use
the fresh3.16x fixed-output ratio, not the old6x ratio or historical short-CLI
roughly10% figure, for the current remaining gap.
