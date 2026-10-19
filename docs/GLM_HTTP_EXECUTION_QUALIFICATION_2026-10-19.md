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

The startup boundary is `JSON -> ExecutionPolicy -> ServicePlan.stage ->
existing numerical/physical IR -> private CUDA lowering`. Export and live
execution invoke that same stage builder; actual export and rank execution
logs identify the choices. No reference engine is linked into these layers.

| HTTP startup field | Existing IR value passed to the stage |
|---|---|
| `expert_scale_partitions: 32` | Distributed maximum physical partition count |
| `expert_transport: "bulk-tiles-128x2x48"` | `ExpertTransport::PackedBulkTiles` |
| `projection_staging: "buffered-k32"` | `MatrixStaging::Buffered(inner_tile=32)` for both projection families |
| `recurrent_evaluation: "chunk-triangular-f32"` | `RecurrentEvaluationLaw::ChunkTriangularFloat32` |

These are deliberate startup choices, not permission for a numerical rewrite
based on agreement with vLLM. Missing settings retain the validated defaults;
unknown settings reject instead of silently selecting another implementation.

At 8192 context / 64 physical rows / 8 slots / 513 history pages, export plans
102,829,853,884 bytes per rank for base/buffered, 102,829,864,636 for scales,
103,127,127,196 for bulk experts and 105,539,374,268 for chunk KDA. A plan below
100 GiB does not establish the separate 16 GiB host reserve; validate both.
Eight slots do **not** mean eight simultaneous full-8192 histories: the shared
page budget permits one full-context history, as the capacity log reports.

## Memory qualification frontier

The resident base run's minimum sampled MemAvailable is 17.57 GiB across
the two ranks. The chunk candidate's additional 2,709,520,384 planned bytes
would leave approximately 15.05 GiB at that observed base footprint. This is a
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
`RecurrentDeltaProgram::append_delta_launches` allocates that workspace and
retains it in the layer's owned intermediates until close; it is not a
startup-only repacking buffer that can be assumed released before serving.

## Initial HTTP results (remaining cohorts pending)

Native binary is pinned to `72ce7bd1`. All rows below use expanded BF16 cache,
row-scaled matrix index scoring, 64 physical rows and the same 8192 envelope.
Each cell lists two independent request sweeps in seconds, one output token,
C1, no cached prompt tokens. Startup and the excluded warmup are not timed.

| HTTP selection | 165 input | 2085 input | 6181 input | 7973 input |
|---|---:|---:|---:|---:|
| Control | 1.237 / 1.233 | 10.908 / 11.021 | 30.280 / 30.973 | 39.517 / 38.808 |
| 32 scale partitions only | 1.140 / 1.128 | 9.828 / 9.884 | 26.416 / 26.563 | 34.137 / 34.175 |
| Bulk experts only | 1.132 / 1.081 | 9.991 / 10.108 | 28.139 / 28.199 | 36.502 / 36.389 |
| Buffered K32 projections only | 1.214 / 1.231 | 10.884 / 11.003 | 31.581 / 30.786 | 39.038 / 39.008 |
| Fresh vLLM reference | 0.794 / 0.832 | 6.818 / 5.963 | 14.303 / 14.144 | 18.451 / 18.145 |

At 7973 tokens the scale-only two-sweep mean improves 12.78%; this is HTTP
latency, not an extrapolation from a selected kernel. Its 34.156 s is still
1.867x the fresh reference's 18.298 s; the integration has not closed the gap.
Remaining native axes are pending. The scale path preserves the observed 11/11 automatic task checks,
four concurrent distinct identifiers and six cancellation/reuse cycles.
A separately labelled mixed-load full-context request finishes `orchid-70`
exactly, with 7973 input / 10 output tokens and normal stop; its 36.188 s group
time is **not** mixed into the isolated latency table. Manual translation,
function, SQL and summary semantics are correct; both control and scale SQL
answers add an unwanted Markdown fence to the SQL-only request.

Raw roots: `/tmp/lunaflux-glm-http-policy-{base,scales}-20261019-v1`,
`/tmp/lunaflux-glm-service-benchmark-policy-{base,scales}-20261019-v1-r{1,2}`,
`/tmp/lunaflux-glm-broad-policy-{base,scales}-20261019-v1`.
Control sampled host minimum: 18,423,984 KiB; scales: 18,439,456 KiB.
Both peak GPU samples: 96,905 MiB; process swap: zero.

The scale harness verdict is retained as **FAILED**, not overwritten: its
watcher interpreted the deliberate coordinator drain as unexpected rank loss
after all request checks completed. The journal records the requested stop,
successful coordinator retirement and peer exit 0; both GPUs were verified
empty before the next run. Fix the external watcher by publishing stop intent
before either stop request; only the partial-pair-loss check consults it.
Memory-limit and failed-rank checks remain active. Three watcher regression
tests pass. Requalify final selected combinations with the corrected runner;
do not misrepresent this old verdict as a clean harness completion.

The bulk cohort was already running the old watcher when that fix landed and
retains the same requested-drain false failure. Its request checks nevertheless
complete: 11 automatic cases, full-context `orchid-70` with normal stop,
distinct concurrent identifiers and all six cancellation/reuse cycles. Both
terminal rank journals show successful retirement and no GPU owners remain.
A stop-intent marker belongs to one immutable run root; starting a new owner
in a retired root is rejected before launch, so stale intent cannot hide an
unexpected rank loss in a later test.
Bulk's sampled host minimum is 18,401,588 KiB; maximum GPU sample 96,907 MiB;
process swap remains zero. Its 7973-token two-sweep mean improves 6.94%, not
the scale-only 12.78%. These gains must not be added without a combined run.

Buffered K32 is not selected for the combined profile: its near-8K mean is
39.023 s, only 0.36% below control, while the 6181-input mean regresses 1.82%.
Two unrandomized sweeps do not establish that sub-percent difference as a
speedup. Its actual export selects buffered staging for both projection
families, so this negative result cannot be dismissed as an unselected path.
Keep the component gains in the older report, but do not substitute them for
this current whole-model HTTP result. The finite combined candidate therefore
uses distributed scales plus bulk expert transport, leaving projections
synchronous and KDA ordered.

Buffered's corrected external runner exits 0. The 11 automatic checks,
full-context retrieval, distinct-request concurrency and six cancellation/reuse
cycles pass; the four manual answers retain correct semantics and the same SQL
fence weakness. Both rank units retire successfully and both GPUs are verified
empty. Actual rank-1 execution logs select `projection_staging=buffered-k32`.
Its watcher records 342 rank samples, minimum host 17,949,588 KiB,
maximum GPU 96,907 MiB and zero process swap. This is request/ownership
qualification of the tested profile, not a reason to choose a slower schedule.

## Fresh reference qualification

`/tmp/lunaflux-reference-bench-glm-http-policies-20261019-v1` uses the same
NVFP4 checkpoint, TP2, 8192 context, maximum eight sequences, 64 batched tokens,
no speculation and no prefix caching. Actual vLLM version is
`v0.1.dev20051+g487ecf187`; the selected expert backend is
`FLASHINFER_CUTLASS`. The cached container tag contains `exl3`, but its runtime
configuration explicitly loads the NVFP4 checkpoint with `modelopt_fp4`;
do not infer checkpoint precision from the image tag. KV allocation is
256 MiB in the reference; storage representations and cache allocation sizes
are not forced to be identical. This is a matched-capacity/step-envelope
comparison, not either engine's independently tuned maximum throughput.

External comparison reports verify literal serialized prompts, actual input
counts and one-output budgets for every timing row. Repeat roots are
`/tmp/lunaflux-glm-service-benchmark-policy-vllm-20261019-v1-r{1,2}`.
The separate `-answers` root requests up to 128 output tokens and returns
`orchid-70`, normal stop, 7973 input / 10 output, in 19.449 s. Broad root
`/tmp/lunaflux-glm-broad-policy-vllm-20261019-v1` passes the same 11 automatic
checks. Translation, Python function, SQL semantics and summary were inspected;
the reference also adds an unwanted SQL Markdown fence. Agreement is comparison
evidence, not a proof that either engine is correct on arbitrary requests.

The reference watcher records 350 rank samples, minimum host MemAvailable
17,187,732 KiB (16.39 GiB), maximum GPU sample 94,546 MiB. Containers enforce
96 GiB memory with swap disabled; the independent watcher enforces the 100 GiB
GPU ceiling and 16 GiB host floor. Both reference units exit 0 and both GPUs
are verified empty before starting the next native owner. The installed Python
resource tracker warns about two shared-memory objects at shutdown; retain
that warning, not an "empty stderr" claim. No profiling overhead is included
in these unprofiled HTTP timings.
