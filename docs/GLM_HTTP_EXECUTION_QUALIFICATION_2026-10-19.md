# GLM HTTP execution-path qualification

Status: HTTP integration and finite qualification campaign complete; original
production restored and functionally checked. Chunk KDA remains memory-blocked. No new
production promotion or claim of full-gap attribution.

After all benchmark owners retired, both original `98c629fd` GLM units were
started at 20:31:09 CST on 19 October. `/health` returns healthy, and the private
chat route returns `323` for 17 times 19, with `finish_reason: stop` (26 input,
6 output tokens). This is a restoration check, not another latency benchmark.
Post-load samples show 96,881 / 96,909 MiB GPU memory and 18,658,424 /
20,319,492 KiB host MemAvailable on .178 / .179, respectively; both owned
units report zero swap. No candidate replaced the validated production bundle.
Restoration response: `/tmp/lunaflux-glm-http-policy-restoration-20261019-v1.json.reply`.

The requested execution graph and offline allocation of retained trace times
are in `GLM_EXECUTION_TIME_GRAPH_2026-10-19.md`. These historical stage weights
must not be presented as timings of the new combined candidate.

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

## Completed HTTP results

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
| Combined scales + bulk experts | 1.120 / 0.985 | 8.852 / 8.947 | 25.209 / 25.040 | 32.461 / 32.352 |
| Fresh vLLM reference | 0.794 / 0.832 | 6.818 / 5.963 | 14.303 / 14.144 | 18.451 / 18.145 |

At 7973 tokens the scale-only two-sweep mean improves 12.78%; this is HTTP
latency, not an extrapolation from a selected kernel. Its 34.156 s is still
1.867x the fresh reference's 18.298 s; the integration has not closed the gap.
The scale path preserves the observed 11/11 automatic task checks,
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

### Combined qualification

Combined 7973-token latency averages **32.4065 s**, versus control 39.1625 s:
**17.25% less time**, or 1.209x speedup. It remains **1.771x vLLM's 18.298 s**.
Do not add the two individual percentage gains; the combined run establishes
its own result. At 165/2085/6181 tokens the combined means are
1.0525/8.8995/25.1245 s, versus reference 0.813/6.3905/14.2235 s. These are
C1 one-output HTTP measurements, not decode throughput or C8 saturation tests.
All comparison reports verify literal prompts, input counts and output budgets.

The source-pinned binary is still `72ce7bd1`; combined artifact root is
`/tmp/lunaflux-glm-http-policy-combined-20261019-v1`. Both export logs and the
terminal rank execution log select `PackedBulkTiles(128,2,48)` and 32 scale
partitions, with synchronous projections and ordered KDA. Planned total per
rank is 103,127,137,948 bytes, including the existing reserve. Actual watcher:
316 rank samples, minimum host 17,921,828 KiB (17.09 GiB), maximum GPU
96,907 MiB (94.64 GiB), zero process swap. Both units retire successfully,
and independent post-stop checks find no GPU owners before restoration.

Corrected qualifier exits 0 at
`/tmp/lunaflux-glm-http-qualification-combined-20261019-v1`. All 11 automatic
task checks, HTTP JSON/SSE/Responses smoke, four distinct concurrent identifiers
and six cancellation/reuse cycles pass. Four open-ended outputs were inspected
for semantics; SQL still has the same unwanted fence seen in control/vLLM.
The full-context answer is exactly `orchid-70`, normal stop, 7973 input /
6 output tokens. Its 32.848 s is kept separate from the reference's 10-output
answer and from the one-output latency comparison. This is finite corpus and
lifecycle qualification, not universal correctness or token-by-token parity.

No CUDA/kernel numerical implementation changes were made in this serving
integration. Existing bounded component gates are retained in
`GLM_PREFILL_PROJECTIONS_AND_SCALES_2026-10-18.md` and
`GLM_BULK_EXPERT_2026-10-17.md`; their stated row/sanitizer scopes are not
expanded by this report. The current native service tests pass 14/14, targeted
warning-denied check passes with the unchanged warning exclusions, and external
runner/comparison/memory-summary regressions pass. No new full-repository suite
pass is claimed: unrelated migration changes and disk-limited fixtures remain.

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

## Source-to-execution audit: what is proved and what is not

Do not turn a source difference or component speedup into a claimed explanation
of the entire HTTP gap. The user's objection is valid: that requires matching
the source, selected artifact, actual work domain and measured execution.
The existing full vLLM trace and partial native trace have different coverage;
neither their kernel sums nor old native ratios decompose the new profile.

This campaign rechecks local `../vlm` at `10f9b5d74` against the installed
`487ecf187` source snapshots. Direct `cmp` (not hashing) proves the local
`vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py` identical
to the installed snapshot. Local GLM model files are not the installed
`glm5next` implementation; use the captured installed model/backend for those
branches. The reference runtime explicitly selects `FLASHINFER_CUTLASS`, not
the also-installed B12x expert backend.

| Boundary | Source plus execution evidence | Valid conclusion / remaining limit |
|---|---|---|
| Global activation scales | Local `modelopt.py:1509` loads input-scale parameters; `flashinfer_cutlass_moe.py:306` passes global scales to the selected backend. Native `grouped_fp4.mbt:131` builds maximum scans; combined `rank1.cu:7607` and `:7664` emit 32 producers. Earlier trace recorded the original one-CTA scans. | Native dynamically recomputes global maxima; reference uses loaded global-scale metadata, in addition to its online block quantization. Scale-only HTTP latency improves in this experiment. Removing the remaining scan would change the declared quantization program, not merely its parallel schedule. |
| Routed GEMM | Actual reference kernel symbol is `MainloopSm120ArrayTmaWarpSpecializedBlockScaled` with 128x128x256 tile, 48 CTAs, 384 threads, 168 registers and 89,088 dynamic shared bytes. Native combined `rank1.cu:7787` has 32 producer/256 consumer threads, two slots and one-dimensional bulk copies. | Bulk transport is genuinely selected, but is not the same CUTLASS tensor-map/layout/MMA schedule. Bulk-only HTTP gain is measured; no assertion that matching the tile name or instruction count closes the gap. |
| Recurrent attention | Installed `models-glm5next-nvidia-kda.py:571` selects `chunk_kda_with_fused_gate` when prefill exists; trace contains 4,250 invocations of each named KKT sub-stage. Native export/runtime still explicitly select ordered KDA. | Different executable decomposition is established. Current eight-slot chunk workspace fails the conservative host-reserve forecast; no current HTTP gain may be quoted for that alternative. |
| Sparse attention | Installed backend `forward_mqa` converts selected indices to physical indices before dispatch. Trace selects `sparse_mla_decode_dsv3_2_kernel` plus merge even during prefill. The actual decode header iterates selected entries, splits work and bulk-gathers double-buffered KV. Native combined cohort loops `base < limit` across full history before applying the membership mask, then stores scores in global scratch. | Full-history QK versus selected-key QK remains a concrete work-domain difference in the CURRENT expanded-cache artifact, not just the earlier losing latent artifact. At 8K/top2K this is roughly four times the candidate pairs, not a proved four-times wall penalty. Do not cite the unselected `prefill_kernel.cuh` as the executed reference. |
| Index bypass | Installed `sparse_attn_indexer_kpool.py:441` uses context<=topK metadata to bypass scoring. Trace records 1,023 score calls = 93 late passes x 11 layers, not all 125 passes. | Early-prefix scoring work is avoidable in the reference. The entire reference index score+selection sum is only 22.690 ms; this observation cannot explain a many-second gap by itself. |

Source snapshots and actual kernel records are the roots linked in
`GLM_QUANTIZED_INDEX_CACHE_2026-10-19.md`. Current native source is
`/tmp/lunaflux-glm-http-policy-combined-20261019-v1/rank1.cu`.
The sparse difference is an ownership/dataflow issue, not just a smaller tile:
native `kernels/selected_sparse_attention_cuda_source/query_reuse.mbt:58`
groups 16 query rows for one head; its line 89 walks the history and line 103
applies membership after the MMA. Those query rows can have different selected
lists. The executed reference assigns a query token to `blockIdx.x`, a group
of heads to `blockIdx.y`, and a selected-key split to `blockIdx.z`; all heads
in that block use that query's same selected list. These two reuse directions
are explicit in source. Neither "both use matrices" nor "both use async copy"
establishes equivalent work. A selected-domain plan must account for ownership,
gather, online softmax and publication together, preserving its numerical law;
merely changing the storage label or adding another IR layer does not do that.
The reference's recorded 0.605 s uncovered GPU intervals and native early
shared-expert overlap do not prove a current submission/overlap bottleneck.
Current full-gap attribution remains unmeasured; no fresh hardware-counter
capture is claimed by this HTTP campaign. Further compiler changes require a
bounded matched-chain experiment of the selected source, not another guess
based on an isolated source difference or extrapolation of an interrupted trace.
