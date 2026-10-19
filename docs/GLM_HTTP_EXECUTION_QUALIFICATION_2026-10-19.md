# GLM HTTP execution-path qualification

Status: integration and qualification in progress; no new production promotion.

The selected trace in `GLM_QUANTIZED_INDEX_CACHE_2026-10-19.md` proves that
the HTTP owner still selected single-partition activation maxima, packed-row
expert transport, ordered KDA and synchronous projections. Checkpoint CLI
alternatives did not imply HTTP selection.

## Contract and bounded campaign

Expose four independent startup selections as the existing immutable numerical
and physical IR values. The same `ServicePlan.stage` must feed both offline AOT
export and live preparation. No comparison, profiler, reference-engine names,
runtime compilation, payload hashes or per-request policy parsing enter the
compiler or token path. Defaults remain the currently validated choices.

Budget: one candidate per axis (32 scale partitions, bulk-tiled experts with
128 columns/two consumer teams/48 workers, chunk-triangular KDA, buffered K32
projections), plus a combined run only if the individual gates justify it.
These are explicit offline candidates, not universal optimal constants.
Keep expanded BF16 cache throughout: the latent-cache experiment lost.

For each candidate, inspect emitted symbols and launch geometry, run relevant
component/state/ownership checks, then use the same 64-row/8192-context HTTP
envelope and literal prompts. Preserve 165/2085/6181/7973-input one-output C1
timings separately from complete-answer, retrieval, concurrent-request and
cancellation tests. Compare against the matched vLLM configuration externally.
Repeat unprofiled timings; do not extrapolate the aborted native trace.

The scale and bulk paths have historical overlap-instability observations.
Component equality does not clear that whole-model boundary. Conversely,
different valid answer wording is not a correctness failure. Evaluate answers,
determinism, state isolation and actual memory behavior, retaining failures.
No tolerance or reference changes are permitted.

Stop the current GLM pair only for the qualification campaign; restore the
validated service after all candidate owners retire, including on failure.
Every run is capped at 100 GiB per Spark with no swap and a separate 16 GiB
host MemAvailable reserve. Do not repeat full-resident Nsight captures that
already exceeded this reserve; use bounded selected-source probes instead.

## Startup integration

`84dd08a1` connects four explicit startup fields to `ServicePlan.stage`, shared
by export and resident execution. `72ce7bd1` preserves the pinned ARM compiler's
`Int` spelling: the first build rejected a formatter-induced `Int32` migration;
that failed build is retained, not counted as qualification. The second clean
commit-pinned build completes (336 tasks, 1.4 GiB peak, zero swap).

The default bundle directly matches the previous index-only AOT sources on
both ranks; it reuses those binaries without payload hashing. Native service
tests: 14/14. The external finite HTTP runner performs two latency sweeps,
complete task answers, request-identity concurrency and cancellation/reuse;
its memory watcher retires both owned ranks on a resource or rank failure.
It does not insert comparisons or selection into the compiler.

At 8192 context / 64 physical rows / 8 slots / 513 history pages, export plans
102,829,853,884 bytes per rank for base/buffered, 102,829,864,636 for scales,
103,127,127,196 for bulk experts and 105,539,374,268 for chunk KDA. A plan below
100 GiB does not establish the separate 16 GiB host reserve; validate both.
Eight slots do **not** mean eight simultaneous full-8192 histories: the shared
page budget permits one full-context history, as the capacity log reports.

## Memory qualification frontier

The resident base run reaches about 17.7 GiB MemAvailable on rank 0 while
serving. The chunk candidate's additional 2,709,520,384 planned bytes would
leave approximately 15.2 GiB at that observed base footprint. This is a
projection, **not** a measured chunk-run memory violation. Do not launch the
same 8-slot/8K chunk configuration under a known insufficient host margin;
its AOT build passed, but this HTTP envelope is not qualified and has no timing
result. Fix workspace lifetimes or qualify a separately labelled capacity
before enabling it. Do not silently reduce context/concurrency for a favorable
entry in the matched table.

The allocation is explainable in the existing physical IR, not an unexplained
OOM: `ChunkDeltaPlan::new` reserves
`4 * heads * sequences * (6*R*K + 3*R*R + 2*R*V)` bytes.
Here R=64, K=V=128, TP-local heads=32 and sequences=8: 76 MiB per
recurrent layer, times 34 layers = exactly the 2,709,520,384-byte delta.
It reserves a full 64-row triangular workspace for every slot, despite the
frame having only 64 total token rows. Packed per-sequence workspace or
cross-layer transient reuse is a separate physical-plan improvement; neither
is implemented or qualified by merely exposing the HTTP selector.
