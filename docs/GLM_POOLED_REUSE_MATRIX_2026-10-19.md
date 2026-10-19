# Pooled-key reuse and matrix sparse attention

Status: experimental implementation and bounded GPU qualification in progress.
This document does not claim a new full-model speed or online deployment.

## Measured scope

The frozen baseline is the actual source retained at
`/tmp/lunaflux-long-paged-sparse-20261019-v5/next.cuh`. Only its symbol namespace
is changed for side-by-side replay. The probe uses 64 query rows, 32 heads,
256-dimensional Q/K/V, 8,192-token logical history, 513 physical 16-row pages
and 2,048 selected keys plus the incomplete pooling tail.

The existing 7,973-input/one-output HTTP result is 55.779 s versus vLLM
17.656 s (3.16x). The earlier roughly 10% gap described short CLI workloads,
not this long HTTP prefill. See GLM_LONG_PREFILL_SELECTION_2026-10-19.md.
No new full-resident software timeline is attempted: previous attempts
violated the 16 GiB host reserve and were stopped.

## Architecture and experiment sequence

1. PoolUpdatePlan gives one writer to each physical pool touched by the
   current write footprint. Unchanged pooled keys remain resident. This is
   independent of request ordering and still recomputes same-generation
   overwrites. Resident validity is initialized once, outside the token path.
   Page-table inputs and persistent read/write state are explicit effects.
2. AttentionAccumulationLaw::TiledFloat32 explicitly permits parallel F32
   reductions and matrix products. Ordered arithmetic stays the default.
   BF16 probability publication remains in the numerical program.
3. SelectedQueryReuse owns bounded score storage, query cohorts and
   per-sequence ownership. Every row retains its own selected-key mask and
   causal boundary. A mixed-sequence cohort is partitioned, not treated as
   one request. Device lowering implements matrix layout; no model,
   reference output or benchmark dependency enters IR.
4. The option propagates through the sparse projection owner, decoder
   slice/stage and AOT/runtime service configuration. The explicit service
   value is `sparse_attention_accumulation: tiled-f32`. Code presence is not
   serving-selection evidence: export and runtime must be tested.

The query-reuse schedule traverses a dense physical key domain with each
row's sparse mask. This trades extra dot arithmetic for shared K/V reads.
It stores F32 scores, publishes BF16 probabilities, then performs a shared
value matrix product. Its larger scratch allocation must pass the full
100 GiB/Spark and 16 GiB host-reserve gate before deployment.

## Preserved experiments

All roots below are /tmp/lunaflux-long-paged-sparse-20261019-vN.

| Version | Change | Observation / decision |
|---|---|---|
| v6 | Incremental pool ownership, serial duplicate-writer scan | Pool 0.00655 to 0.0127 ms: rejected schedule |
| v8 | Parallel pool ownership; one live matrix query row | Pool about 0.00406 ms; attention about 14.7 to 13.7 ms, modest benefit |
| v9 | 16-query history reuse, serial softmax | Normal timing about 14.3 to 8.5 ms; **not qualified**: sanitizer exposed an empty-tile flag-reader race and output instability |
| v10 | Reader retirement before empty-tile reuse; parallel softmax | Normal late probe about 14.5 to 6.5 ms; full 64-row memcheck/leak/race/sync/init gates pass |
| v11 | Cohort prefill plus row-wise dispatch below 16 active queries | Late attention 6.394–6.700 ms versus 14.254–14.367 ms; full 64-row sanitizer gates pass |

v8 Nsight Compute capture (counters-next.ncu-rep): old/new attention
executed 1.737/1.072 billion warp instructions but read 145.35/141.97 million
L2 sectors (4.65/4.54 GB). Profiled time 14.294/13.326 ms. Long-scoreboard
contributions were 61.5%/57.5%; barriers 21.3%/23.3%. These are warp-stall
metrics, not additive fractions of whole-service time. Index selection
remained about 7.55 ms with 212 million instructions and 12.1% active-warps
occupancy. Pooling microseconds cannot explain seconds of service latency.

v9's failure is retained rather than relabeled as an arithmetic difference:
empty tiles could advance and overwrite the shared flag before all readers
retired. v10 adds reader retirement before the uniform skip. Normal v10
output differs from ordered accumulation, as permitted by a different
numerical program; this is not a bitwise-equivalence claim or a waiver.

## Qualification boundary

Comparison remains external. Diagnostics report exact selected indices,
all-output error statistics, an independent F64 attention calculation with
the declared BF16 probability boundary, and sanitizer results. A bounded
analytic fixture is not full-model correctness. Mixed sequences, invalid
selection, page-generation reuse, tails and deterministic replay require
separate coverage. Full-model complete answers and matched unprofiled timing
remain necessary before changing the online service.

No payload hashes or integrity scans are performed. Only task-owned paths
may be committed; unrelated toolchain edits must remain untouched.

## Executable qualification

Implementation 2fa80cd2 is explicit, not an unconditional default numerical
change. Clean pinned ARM build at
`/tmp/lunaflux-checkpoint-build-glm-pooled-matrix-20261019-v1` passes
(1.2 GiB reported peak, zero swap). The full-model AOT/configuration at
`/tmp/lunaflux-glm-http-pooled-matrix-20261019-v1` selects tiled-f32 on both
ranks; all 11 sparse-layer sources contain the cohort dispatch. Context8192,
8 slots,16 lanes,513 physical pages and64 prefill rows are unchanged.

`/tmp/lunaflux-query-reuse-20261019-v2` runs all rows/heads against an
independent F64 calculation on 12 fixtures: mixed sequences, GQA, noncontiguous
pages, generation mismatch, malformed selections, empty selections/counts,
1/15/16/17/31 active rows and inactive tails. Three repeated executions are
bitwise deterministic within each selected program. Full fixtures pass
memcheck/leak/race/sync/init gates. A BF16 rounding envelope on an analytical
fixture is not a relaxed model tolerance or full-model quality evidence.

`/tmp/lunaflux-paged-sparse-pool-reuse-20261019-v2` passes five stateful
mixed-request/reorder/generation-reuse steps, plus a same-generation overwrite
of a completed pool. The updated pool matches recomputation and untouched
pools remain bitwise. All sanitizer/leak gates pass.

v10's counters show the substantive traffic reduction: attention L2 read
sectors145.51M to37.38M (4.66GB to1.20GB), warp instructions1.737B to260.33M.
NCU durations14.524/7.020ms are separate from unprofiled event times.
Long-scoreboard remains59.3%, barrier12.7%, active warps42.6%. Higher occupancy
is not the objective: the smaller data demand is what this schedule changes.
The current index leaf still takes about7.6ms and remains a distinct frontier.
Do not attribute the complete HTTP gap solely to these synthetic-input leaves.

## Full-service memory finding and correction

Initial explicit 2fa80cd2 warm one-output vector (165/2085/6181/7973 input
tokens) completed in 1.221/11.247/32.939/44.459 seconds. The separate first
vector was 1.270/11.205/32.857/43.286 seconds. These are not confidence
intervals. The subsequent complete-answer C4 test did **not** qualify:
rank0 MemAvailable reached 16,428,828 KiB (15.67 GiB), and the external
watchdog stopped both ranks. GPU memory stayed below 100 GiB and both owned
units reported zero swap. The incomplete responses are retained at
`/tmp/lunaflux-glm-service-benchmark-pooled-matrix-answers-20261019-v1`.

The new dense score scratch was independently allocated by all 11 sparse
layers even though their lifetimes do not overlap. A pure `SerialScratch`
storage plan now budgets the maximum, not the sum, for explicitly serialized
phases. The decoder lends one allocation through the projection/attention
frames, closes every borrower before the allocation, and never aliases
persistent KV, pooled keys or selected indices. Expert producer forks do not
access it and join within their layer. This saves 640 MiB per rank at this
geometry without shrinking context/admission capacity or changing arithmetic.
The existing generated CUDA/AOT source is unchanged. This ownership correction
requires a fresh full-service memory/concurrency retest before deployment.

The correction is committed as0ee75887. Clean pinned ARM build succeeds
(1.2 GiB peak, zero owned-unit swap). Both exported rank sources compare
directly equal to2fa80cd2 and the existing AOT files are reused without GPU
compilation or payload hashing. The local phase suite passes5310/5310 with
the existing migration-warning exclusions; targeted `moon info` reports1065
existing migration warnings and zero errors. Bare warning-denied validation
remains blocked by the already recorded unrelated migration diagnostics.

## Remaining source differences (not an additive timing attribution)

The actual installed vLLM variant is retained at
`/tmp/lunaflux-glm-long-indexer-vllm-20261019.py`; it is not identical to
the model coverage in the local `../vlm` checkout. Its pooled index path calls
DeepGEMM matrix logits and has a no-scoring path when all candidates are
selected. LunaFlux's current paged index still assigns each query to one
workgroup, and each candidate lane loops through 32 heads and 128 components
in ordered scalar F32 arithmetic before stable selection. The late probe's
roughly 7.6 ms index duration is therefore a separate executable gap; caching
the pooled keys does not remove scoring work.

The new matrix attention retains the existing expanded BF16 K/V representation.
vLLM's selected implementation uses compressed MLA/FP8 state. Query reuse
reduces repeated native reads, but does not make these representations or
memory demand equivalent. Converting the representation would require a
separate semantic/numerical and ownership change, not a hidden lowering cast.
The synthetic 64-query probes identify these mechanisms; they do not establish
their individual fractions of the remaining end-to-end service gap.

## Final-kernel counters and fresh reference

After the reference pair released both devices, v11 was captured serially in
the bounded2 GiB/no-swap probe. The final exported kernel includes the
small-row dispatch, not just the earlier cohort prototype. Raw report:
`/tmp/lunaflux-long-paged-sparse-20261019-v11/counters-next.ncu-rep`.

| Late-history attention counter | Frozen old | Final v11 |
|---|---:|---:|
| Profiled duration |14.216 ms|6.941 ms|
| Warp instructions |1,737,421,824|263,902,848|
| L2 read sectors (32 B) |145,337,501|37,241,952|
| Long-scoreboard contribution |61.21%|57.71%|
| Barrier contribution |21.20%|12.44%|
| Active warps |80.61%|43.06%|
| Issue-active metric |23.91%|7.88%|

Lower instruction demand and traffic reduce duration despite lower occupancy
and issue-active percentages. Neither percentage is an objective by itself.
The final index probe remains7.681 ms,212,210,635 warp instructions,12.08%
active warps,51.47% long-scoreboard contribution. It is now slower than the
attention leaf in this late-history probe, not proven the largest whole-model
component. The common scratch ownership change does not alter these kernels.

Fresh vLLM uses the same NVFP4 checkpoint,TP2,8192 context,8 maximum sequences,
64-token prefill chunks,256 MiB KV and capture maximum8 as the previous bounded
baseline. Prefix caching and speculation are disabled. It completes both
one-output vectors and all64 complete-answer requests. Warm second-vector
times for165/2085/6181/7973 inputs are0.808/6.057/15.018/17.642 seconds.
Complete8K answer takes18.457 seconds and10 generated tokens, exact label.

The broader reference answer vector has57/64 exact answers and64/64 completed
requests. Seven responses contain the label but also repeated reasoning after
the first `</think>`, so they fail the explicit exact-label task. This is an
observed output-format failure, not a proven vLLM implementation bug. The
reference is not a correctness oracle; no parser, stop list or reference text
is changed to hide these failures. Output lengths differ and answer-cohort
times must not be presented as equal-generated-work throughput. The four
one-output comparisons remain the fixed-output latency control.

## Small-row arithmetic control (not yet qualified)

The shared-scratch service's warm fixed-output vector is
1.231/11.902/32.636/43.177 seconds. The 7,973-token latency is 22.6% below
the previous55.779 seconds, but still2.45x the fresh17.642-second reference.
Its complete-answer short C16 group contains one exact-format failure:
lane12 returns the correct orchid-82 label followed by repeated reasoning.
The previous ordered service returned only the label for that prompt.
This is not yet proof that changed decode arithmetic is the cause: batching
and upstream prefill arithmetic must not be conflated.

A bounded follow-up keeps matrix query reuse for a full physical cohort,
but uses the existing ordered row program for fewer live queries than the
cohort size. This avoids one-live-row MMA work without changing the matrix
prefill algorithm, page ownership, or persistent state. Static shapes that
cannot support a reuse cohort still retain their explicit matrix alternative.
No output parser, reference or acceptance condition is changed. The candidate
must pass generated-source checks, the mixed-query analytic/sanitizer fixture,
and a fresh full-service answer/memory check before any promotion.
