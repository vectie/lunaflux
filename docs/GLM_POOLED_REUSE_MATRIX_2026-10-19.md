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
| v10 | Reader retirement before empty-tile reuse; parallel softmax | Normal late probe about 14.5 to 6.5 ms; sanitizer and broader fixture gates pending |

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
