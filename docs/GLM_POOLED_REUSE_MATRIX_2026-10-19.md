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
