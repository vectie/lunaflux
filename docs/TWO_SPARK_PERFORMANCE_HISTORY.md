# Two-Spark performance: history, evidence and current limits

Compiled 2026-10-24 from the conversation and repository reports. Latest
measurement used here: 2026-10-22. No machines were tested or services changed
while writing this document. Use the [engineering guide](TWO_SPARK_PERFORMANCE_GUIDE.md)
for the repeatable workflow; this file preserves what actually happened.

The tables are a history of **different experiments**, not a single continuous
speedup curve. Within every row, distinguish component/chain/decoder/request,
short/long, fixed/variable output work, native strategy/matched scheduling,
fresh/retained reference and selected/experimental code. Follow the cited report
for exact inputs, versions, trial counts, raw roots, failures and limits.

## 1. September: correcting the first explanations

| Workstream | Finding and consequence | Evidence |
| --- | --- | --- |
| Capacity and mixed timeline | C16 repeated prefill was corrected from 94,352 to 65,536 tokens; execution became 96 rather than 126 steps. Work-count bugs must be fixed before kernel attribution. That timeline predated the final RoPE change. | [Capacity/timeline](BENCHMARK_CAPACITY_AND_TIMELINE_2026-09-14.md) |
| Async staging | “Async adds about 33% instructions, cancelling its benefit” was not a timeless conclusion. Copy/address/packing changes and selection changed the result; preserve the exact candidate and workload. | [Async metrics](ASYNC_ATTENTION_METRICS_2026-09-14.md) |
| Framework gap | DRAM traffic, occupancy and bank-conflict counters were insufficient individually. Actual work, instruction mix and latency hiding needed source-level analysis. | [Spark counters](BENCHMARK_SPARK_COUNTER_DIAGNOSIS_2026-09-29.md) |
| Full QKV fusion | The epilogue constrained projection to 16 rows × one head per CTA: 4,096 CTAs versus 256 in the selected reference projection. Decouple matrix geometry and epilogue ownership. These are CTAs, not 4,096 separate launches. | [Three-kernel review](BENCHMARK_SPARK_THREE_KERNEL_SOURCE_REVIEW_2026-09-29.md) |
| Prefill selection | Serving selected synchronous c318; asynchronous c322 merely existed. Static equal costs and a lower-ID tie-break explained selection. An alleged async latency-hiding failure was based on the wrong executed path. | [Three-kernel review](BENCHMARK_SPARK_THREE_KERNEL_SOURCE_REVIEW_2026-09-29.md) |
| Prefill support work | Captured tensor MMA and exponent counts were similar, but register MOV was 4.19M versus 0.38M and total warp instructions 48.23M versus 19.56M. KV32/64 limits, fragment movement, addressing and scalar math warranted separate tests. | Same source review; counts apply to the selected invocations |
| Decode | GQA reuse already worked. Per-key recurrence, frequent publications, unconsumed page-hoist plans and restricted split alternatives remained. About 6.8× instructions did not imply 6.8× time: the selected replay gap was about 8%. | Same source review |
| Architecture | The missing ingredient was not simply another IR: restricted schedules, weak selection and unrealized lowering were independent problems. Physical plans must generate different executable work. | [Physical tile IR](PHYSICAL_TILE_IR_2026-09-29.md), [counter repair](COMPILER_COUNTER_REPAIR_2026-09-29.md) |

The review also distinguished the actual baseline invocation from unused paged
fallbacks. A source branch that did not execute cannot explain the recorded
stall. Strict exp/rounding versus exp2/FMA requires a numerical contract, not a
claim that an arithmetic change is just scheduling.

## 2. October 3–9: scaling, AKO and the compiler-value question

### Long context exposed prefill, not universally slow decode

The October4 Qwen3-0.6B campaign ran on one Spark, not TP2 GLM. All rows below
are C1 unprofiled median completion seconds with exact output counts:

| Input / output | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| 4,096 / 64 | 0.684 | 0.768 | 0.778 |
| 8,192 / 64 | 1.018 | 1.075 | 1.076 |
| 16,384 / 64 | 2.515 | 1.836 | 1.802 |
| 32,512 / 64 | 7.472 | 4.018 | 3.741 |
| 32,512 / 256 | 11.955 | 8.662 | 8.461 |

For 32,512/64, TTFT was 5.977/2.481/2.162 seconds, but mean post-first-token
interval was 23.41/23.95/24.75 ms. This located the large C1 gap before the
first token without proving which prefill operator caused it. Native chunk
policies differed, particularly SGLang's 8,192 versus 2,048 for the other two.
The model's 32,768 total-context envelope and memory budget did not permit a
claim of million-token serving. [Full design and results](BENCHMARK_LONG_CONTEXT_2026-10-04.md).

### The optimization loop produced both wins and losers

The AKO series varied page lookahead/readiness, query and fragment lifetime,
value windows, sibling ownership, row grouping, shape dispatch, exponential
laws and complete chains. It did not establish one universal attention schedule.
Long-history and short-input behavior often differed. Component instruction
improvements, non-bitwise arithmetic alternatives and product-level wins were
recorded separately. [AKO method and evidence index](AKO_CODEX_LUNAFLUX.md).

Important controls from this phase:

- Equal-work and partitioned-decode investigations separated token counts,
  scheduling and kernel cost: [equal-work proof](BENCHMARK_EQUAL_WORK_PROOF_2026-10-03.md),
  [partitioned trace](BENCHMARK_PARTITIONED_DECODE_TRACE_2026-10-03.md).
- Reference latency hiding and full-step attribution prevented single-counter
  explanations: [reference overlap](BENCHMARK_REFERENCE_LATENCY_HIDING_2026-10-04.md),
  [full-step attribution](BENCHMARK_FULL_STEP_ATTRIBUTION_2026-10-04.md).
- An optimized projection policy was lost across export/binding, yielding more
  CTAs despite correct local code: [propagation repair](BENCHMARK_PROJECTION_PROPAGATION_REPAIR_2026-10-06.md).
- Replacing a component and factorial controls tested causal attribution, not
  merely similarity to a vendor: [execution graph](BENCHMARK_EXECUTION_GRAPH_COMPARISON_2026-10-07.md),
  [GEMM substitution](BENCHMARK_REFERENCE_GEMM_SUBSTITUTION_2026-10-07.md),
  [factorial analysis](BENCHMARK_REFERENCE_FACTORIAL_2026-10-08.md).

### Why keep the newer compiler if the old one could emit the winner?

The equal-information ablation found the old and new selectors chose the same
kernel for all eight GPU workloads and all 64 budget/order queries. Both chose
the lower validation median among two candidates, not a global optimum. An
isolated four-line backport in the old generator emitted the same winning
schedule. Therefore that speedup did not prove a benefit from extra IR layers.

The architectural purpose remains useful: preserve compatible producer/consumer
alternatives until actual layout and memory requirements are known; model
resources and effects explicitly; apply the same decision during export and
serving. The stronger performance claim requires a connected chain that the
old locally greedy selection misses, with equal information and held-out timing.
[Ablation](BENCHMARK_COMPILER_SELECTION_ABLATION_2026-10-09.md),
[execution economics](EXECUTION_ECONOMICS_2026-10-08.md),
[connected selection](COMPILER_CONNECTED_SELECTION_2026-10-09.md).

## 3. October 10–12: DeepSeek/MiniMax and the correctness rethink

These investigations explain the numerical architecture; they are not GLM
throughput results. Original code was used for matched-input localization, not
as an infallible correctness definition.

| Observed issue | What the intervention established | What it did not establish |
| --- | --- | --- |
| DeepSeek HC projection/RMS/nonlinearity/Sinkhorn differences | Explicit fold, FMA/publication and matrix-sum laws could reproduce the observed original controls on identical inputs | Universal best reduction order or whole-model quality |
| Tiny router-weight deltas | Holding input/expert IDs/weights fixed, replacing only observed route weights reproduced an entire row's FFN mismatch through BF16/FP8 thresholds | An expert-kernel bug, or that all model error had the same cause |
| Shared FP8 down projection | Identical quantized payload/scales still differed near a BF16 midpoint; raw block matrix products plus separate scaled F32 accumulation fixed the demonstrated row | Permission to snap values or replace all math with F64 |
| Query/output FP8 projection | A common private block-matrix primitive propagated through independent immutable projection laws; actual selected prefix/output captures matched | That helper-level success alone implied owner/whole-chain propagation |
| Full-chain replay | Local repairs propagated through previous native hidden states rather than resetting every layer to the original | Head/token quality, TP2 serving parity or throughput |

The strongest counterexample to “more bitwise matches means a better model” was
the completed FP8-projection chain: final position143 mean relative L2 was
14.134457856729%, versus 13.196633236% for the preceding shared-matrix chain,
despite major local improvements. All-row aggregates were dominated by large
entries and were not quality measures. This motivated explicitly separating
finite-precision legality, causal comparison and task/distribution evaluation.
[Control-reduction evidence](DEEPSEEK_CONTROL_REDUCTION_2026-10-12.md),
[original-stream replay](DEEPSEEK_ORIGINAL_STREAM_REPLAY_2026-10-11.md),
[correctness rethink](NUMERICAL_CORRECTNESS_RETHINK_2026-10-12.md).

MiniMax required a different validation ladder: encoder/refiner, joint denoiser,
heads and scheduler updates, then video/audio VAEs and actual media. Packing,
head geometry and arithmetic fixes improved scoped comparisons; finite arrays
did not prove recognizable video. Earlier 128×128 native and original smokes
failed the desired cat visual quality. The user described an earlier native
audio clip as natural/coherent but did not understand it; that was not caption
alignment or approval of a later run. The long 256×256 experiment was later
authorized to stop to free both Sparks. Do not convert an unfinished experiment
into a quality pass. [Media geometry/quality](MINIMAX_QUALITY_GEOMETRY_2026-10-11.md).

Full-model support also required HTTP workers, GPU sampling, cancellation,
retirement, request isolation and route integration, not only decoder kernels.
SGLang image/model support and loading availability constrained the comparison;
a failed launch or an unresponsive machine yielded no valid reference score.
[Broader model comparison](BENCHMARK_SPARK_MODEL_COMPARISON_2026-10-12.md),
[numerical migration](DECODER_NUMERICAL_MIGRATION_2026-10-12.md).

## 4. October 13–18: GLM moves from serial correctness paths to useful parallelism

The early short-prompt cohorts below used six prompts, typically 22–36 input
tokens. They are not the later 7,973-token HTTP test. Some early native runs
used PP2/TP1 and different interface/warmup boundaries from vLLM; the cited
reports retain those limitations. Do not treat every row as one controlled
architectural ablation.

| Date / intervention | Recorded effect | Interpretation / source |
| --- | --- | --- |
| Oct13 sparse score reuse | Pathological selected leaf examples dropped from seconds to sub-ms by eliminating repeated score computation | Large component fix, not a thousandfold model speedup. [Sparse reuse](GLM_SPARSE_SCORE_REUSE_2026-10-13.md) |
| Parallel HC and multi-row prefill | Six-prompt native sum 323.475 → 32.981 s; selected HC 6.282304 → 0.1598 ms | Serial correctness implementations were an actual bottleneck. [Controls/prefill](GLM_PARALLEL_CONTROLS_PREFILL_2026-10-13.md) |
| Chunk KDA first attempt | Six-prompt sum 32.981 → 33.227 s; recurrence only 0.105% of the captured kernel time versus projections 40.404% | Optimized the wrong dominant phase. [Chunk KDA](GLM_CHUNK_KDA_2026-10-13.md) |
| Tiled projections/grouped experts, repaired | 32.981 → 21.016 s versus vLLM 3.144176 s, still 6.684× | Initial grouped path had regressions; realistic dimensions and packed decode fixes mattered. [Tiled/grouped](GLM_TILED_GROUPED_EXECUTION_2026-10-13.md) |
| Small-group/sparse repair | 21.016 → 16.509 s | Small expert rows and sparse projections needed distinct treatment. [Small groups](GLM_SMALL_GROUP_SPARSE_2026-10-13.md) |
| Expert TP and native low-bit execution | PP2 16.509 s; TP2 BF16 rows16 12.347 s; TP2 FP4 rows16 12.562 s; TP2 FP4 rows64 10.197 s | TP, datatype and prefill parallelism are separate factors. [Expert TP](GLM_EXPERT_TENSOR_PARALLEL_2026-10-13.md) |
| FP4 operand supply | Gate/up 401.696 → 249.056 us; down 175.264 → 142.112 us; full cohort improvement much smaller | Leaf acceleration diluted by other stages. [Operand supply](GLM_FP4_OPERAND_SUPPLY_2026-10-13.md) |
| Shape-aware compiler connection | Selected dense/router/head and decode projections improved substantially | Existing generic compiler paths had to reach GLM owners. [Shape compiler](GLM_SHAPE_COMPILER_2026-10-14.md) |
| Shared-input projection fusion | Six-projection chain benefits depended on rows; early full-model comparison was confounded/noisy | Do not claim all local fusion savings survive serving. [Region fusion](GLM_REGION_FUSION_2026-10-14.md) |
| Oct17 bulk expert transport | Rows22 chain about 6.759 → 5.214 ms; rows64 13.140 → 10.179 ms | Transfer/consumer geometry improved time even while instruction counts rose. [Bulk experts](GLM_BULK_EXPERT_2026-10-17.md) |
| HC fusion and collective graphs | Selected HC prefill 19.897 → 5.836 ms; graph capture changed decode and prefill differently | Reduced launches do not guarantee every graph path wins. [HC/graphs](GLM_HC_COLLECTIVE_REPLAY_2026-10-17.md) |
| Oct18 prefill geometry + shared/routed overlap | Equal-output five-prompt comparison gap about 21.45% → 15.99% | Four-way geometry/overlap ablation was necessary; overlap alone could regress. [Prefill/overlap](GLM_PREFILL_AND_BRANCH_OVERLAP_2026-10-18.md) |
| HC plus vocabulary TP | Combined five-prompt time 3.098 s versus reference 2.765851 s, about 12.01% higher | Complete chain including head/collectives, not isolated argmax. [HC/vocabulary](GLM_HC_AND_VOCABULARY_TP_2026-10-18.md) |
| Deferred HC post/pre | Repeated mean 3.104 → 3.0965 s, about 0.24%; shape-dependent component results | Too small to call a decisive gap fix. [Deferred HC](GLM_DEFERRED_HC_2026-10-18.md) |
| Distributed scale scans | Selected 64-row scans 163.361 → 11.146 us and 324.755 → 19.976 us | Real leaf work reduction; whole-path instability and actual selection still required checks. [Scales/projections](GLM_PREFILL_PROJECTIONS_AND_SCALES_2026-10-18.md) |

The scope expanded from resident CLI to a GPU-backed HTTP worker with sampling,
request retirement, cancellation, complete answers, wider context/concurrency
and LunaNexa routing. “Full service” required these lifecycle/API checks; a
kernel smoke returning one token did not complete the product boundary.
[Complete answers](GLM_COMPLETE_ANSWERS_AND_SERVICE_2026-10-18.md),
[service wiring](GLM_FULL_SERVICE_WIRING_2026-10-18.md).

## 5. October 19: why the long HTTP path was much slower

The matched 7,973-input/one-output control initially took **103.887 s versus
17.622 s**, or 5.90×. The earlier roughly 11% comparison had only 22–36 input
tokens and a resident diagnostic interface. This was a missed workload/selected
path coverage gap, not proof of a sixfold regression on unchanged work.

The long path performed repeated serial candidate selection and expensive
attention-side validation/copies. At history8000/full64 queries, the isolated
index leaf took about 61 ms and attention about 21 ms. Stable parallel selection,
bounded bitmap membership and cooperative key copies reduced the combined
synthetic pair from about 82.4 to 22.1 ms. Full HTTP one-output latency then
fell from 103.887 to 55.779 s. These are two separate proof levels.
[Long-prefill report](GLM_LONG_PREFILL_SELECTION_2026-10-19.md).

| Subsequent change | What was established | Limitation |
| --- | --- | --- |
| Pooled-key reuse / matrix attention | Completed pools could be retained; score/value work could use a better physical decomposition | Must preserve generation, partial pools, causal selection and actual owner propagation. [Pooled/matrix](GLM_POOLED_REUSE_MATRIX_2026-10-19.md) |
| Explicit E4M3 matrix index | Selected score-producer L2 sectors 34.085M → 1.446M; index-only near-8K service repeat 41.679 → 38.306 s | Changed numerical program; not a 10× model speedup. [Index/cache](GLM_QUANTIZED_INDEX_CACHE_2026-10-19.md) |
| Latent BF16 cache | Qualified fixtures and answer cohorts; substantially smaller attention payload | Near-8K service 42.515 versus 38.306 s index-only: 11.0% slower. Not selected as the winner. Same report |
| Connect HTTP to scale/bulk alternatives | Control two-sweep mean 39.1625 s; scales about 34.156 s; bulk about 36.4455 s; combined 32.4065 s; fresh vLLM 18.298 s | Individual savings do not add; buffered-K32 HTTP gain was inconclusive. [HTTP qualification](GLM_HTTP_EXECUTION_QUALIFICATION_2026-10-19.md) |
| Exclude unsafe chunk KDA envelope | Forecast additional workspace 2,709,520,384 bytes; estimated reserve below 16 GiB | Not a measured launch failure or missing selector; no timing for that unsafe envelope. Same report |

A full model can therefore be slow even when the desired kernels exist. Actual
HTTP selection originally retained single-partition maxima, packed-row expert
transport, ordered KDA and synchronous projections. Startup selection was
connected through the same `ServicePlan.stage` used by export and preparation;
tests then verified real symbols, resources and execution.

The reference profiler once silently substituted a short-context envelope.
That harness defect was corrected and retained in the record. Failed full
resident Nsight captures breached the host reserve; later work used lower-memory
observers and bounded leaf counters instead of repeating the unsafe setup.

## 6. October 19–22: TP causality, projection pipelines and observer cost

| Work | Finding | Reusable lesson |
| --- | --- | --- |
| Local MoE combine | Combine rank-local contributions before one reduction per MoE | Declare publication/rounding; remove a communication boundary rather than only tuning NCCL. [MoE graph](GLM_SCALES_BULK_EXECUTION_GRAPH_2026-10-19.md), [projection TP](GLM_PROJECTION_TP_2026-10-19.md) |
| BF16 projection TP | Actual trace showed 48 projection plus 42 MoE BF16 reductions and removed post-collective casts | Reduced payload is not proof that every long span is network time. [Projection TP](GLM_PROJECTION_TP_2026-10-19.md) |
| Paired dense-FFN waits | Both-rank timestamps separated early arrival from a remaining post-arrival stall | Single-rank duration cannot identify transfer latency. [Paired TP](GLM_DENSE_TP_PAIRED_2026-10-20.md), [submission](GLM_NCCL_SUBMISSION_2026-10-20.md) |
| Cold/warm proxy progress | Measured synchronous posted-operation ingestion on the NCCL progress thread, including a 6.075 ms append interval | A real software-progress stall, not demonstrated wire latency or OS descheduling. [Cold/warm](GLM_NCCL_COLD_WARM_2026-10-20.md) |
| Windowed collective graphs | Bound pending collective sites, preserve ordered effects and whole-frame retirement; no-plugin control corroborated removal of the multi-ms dense-site spike | Graph granularity is a resource choice. More capture is not automatically better. Same report |
| Projection reuse/staging/fragment work | Qualified schedules reached selected large projections; many plausible alternative schedules lost | Evaluate actual MMA issue and resources, not a feature list. [Staging](GLM_PROJECTION_STAGING_2026-10-20.md), [fragment pipeline](GLM_FRAGMENT_PIPELINE_2026-10-21.md) |
| Low-memory filtered activity tracer | Bounded phase capture and pre-allocation polling filters enabled completed two-rank native traces | A tracer must have a capacity and completeness contract. [Low-memory](GLM_LOW_MEMORY_TRACE_2026-10-21.md), [filtered trace](GLM_FILTERED_TWO_RANK_TRACE_2026-10-21.md) |
| Idle rank-1 abort investigation | Zero-timeout socket polling created two coroutine IDs per empty iteration; an isolated ARM wrap/collision reproducer aborted. Suspending the idle stage removed the busy loop | Not a demonstrated memory kill, captured natural-abort stack, repair of the external library's general ID limit, or completed multi-hour soak. [Idle abort evidence](GLM_LOW_MEMORY_TRACE_2026-10-21.md) |
| Reader allocation ownership | Each worker's anonymous PSS roughly halved with GPU allocation unchanged | Loading host allocations were real; successful weight load still was not readiness. [Loader/instructions](GLM_LOADER_ALLOCATIONS_AND_INSTRUCTIONS_2026-10-21.md) |
| Dummy profile capability | Removed 2.03125 GiB unreachable dense-MHA workspace from the selected sparse backend | Inspect actual allocation stack and backend capability, not the last warmup log. [Latest campaign](GLM_DUMMY_PROFILE_AND_FRAGMENT_TRAFFIC_2026-10-22.md) |

## 7. Negative results that must remain visible

| Experiment / tempting inference | Actual result | Decision rule |
| --- | --- | --- |
| Larger shared K tile halves loads | Small-group example occupancy fell 36.85% → 15.5%; time 1.209856 → 1.644544 ms | Include resource residency and useful parallel work. [Small groups](GLM_SMALL_GROUP_SPARSE_2026-10-13.md) |
| Register-only experts eliminate barriers | Singleton example time about 1.26 → 3.138 ms; instructions and load dependencies grew | Zero barriers is not the objective. [Expert TP](GLM_EXPERT_TENSOR_PARALLEL_2026-10-13.md) |
| Restore occupancy, recover speed | Sequential scratch restored gate/up shared storage and occupancy, but selected gate/up replay remained about 4.21 ms | Occupancy is a means, not performance. [Whole-path repair](GLM_EXECUTION_REPAIR_2026-10-17.md) |
| Fewer instructions must mean faster | Bitmap validation removed about 58% of attention instructions but barely changed time; later copies increased instructions while improving time | Use dependency/traffic/counter interactions. [Long prefill](GLM_LONG_PREFILL_SELECTION_2026-10-19.md) |
| Less cache memory must be faster | Latent-cache HTTP repeat was 11.0% slower at near-8K | Count absorption/expansion and complete numerical chain. [Index/cache](GLM_QUANTIZED_INDEX_CACHE_2026-10-19.md) |
| Address invariants explain the projection gap | Instructions 17.119M → 5.774M/5.043M, yet paired shapes mostly regressed or were inconclusive | Do not promote a counter-only win. [Loader/instructions](GLM_LOADER_ALLOCATIONS_AND_INSTRUCTIONS_2026-10-21.md) |
| Replace load waits with elected waiting | Long-scoreboard share fell to 2.815%, barrier share rose to 67.783%; no qualified universal gain | Moving the wait changes its label, not necessarily completion. [Transfer waits](GLM_GRAPH_WARMUP_AND_TRANSFER_WAITS_2026-10-22.md) |
| Double fragment ownership to match reference traffic | M64/N128 halved L2 reads, but five paired timing cells did not qualify | Withdraw; retain experiment patch. [Latest campaign](GLM_DUMMY_PROFILE_AND_FRAGMENT_TRAFFIC_2026-10-22.md) |
| Every local numerical match improves the model | DeepSeek final chain norm worsened despite local exact matches | Evaluate the complete intended output and state. [Correctness rethink](NUMERICAL_CORRECTNESS_RETHINK_2026-10-12.md) |
| Old reference timings are fresh control data | October21–22 vLLM reloads did not produce a fresh request trace | Label retained October19 reference explicitly; no invented paired TP attribution |

These failures are useful constraints on the next experiment. Repeating them
unchanged is not progress. Different input rows, cache state, observer overhead
or deployed policy must be disclosed before attributing a changed result.

## Latest verified two-rank GLM state

Evidence: [October22 complete campaign](GLM_DUMMY_PROFILE_AND_FRAGMENT_TRAFFIC_2026-10-22.md).
Native root: `/tmp/lunaflux-glm-http-dummy-traffic-20261022-v2`.
Accepted runtime `ed6be466`, projection AOT `b1788d1d`; **not** the rejected
wide-fragment experiment. Configuration retained scales+bulk experts, local MoE
combine, BF16 projection TP, 16-site collective graph windows, matrix index,
expanded BF16 cache and ordered KDA. The preceding filtered trace documents
rows64/context8192/slots8/pages513/admission16 and the selected large sparse
output's swizzled fragment pipeline.

Both requests completed: 7,973 prompt tokens, zero cached tokens, one output,
terminal `[DONE]`. Both collectors sealed two windows with dropped/invalid/
errors/refused all zero. A separate arithmetic answer returned `323` with
normal stop. The allocation below is **one complete 64-token chunk, 45 layers**,
not the whole prompt and not an overall benchmark.

### Disjoint wall-time allocation

Milliseconds; raw shared work is not added twice. The reference is explicitly
the retained **October19 rank1** database, not a fresh October22 run.

| Stage | Native cold R0 | Native cold R1 | Retained vLLM R1 | R1 difference |
| --- | ---: | ---: | ---: | ---: |
| KDA projections | 34.218433 | 34.762221 | 26.704800 | +8.057421 |
| Sparse projections | 15.860067 | 15.681542 | 9.372864 | +6.308678 |
| HC | 6.726022 | 7.054959 | 3.950080 | +3.104879 |
| KDA other | 8.850429 | 8.941144 | 4.422656 | +4.518488 |
| Dense FFNs | 5.861578 | 5.854931 | 2.281952 | +3.572979 |
| TP including waits | 18.203554 | 16.758112 | 17.468992 | −0.710880 |
| MoE union, overlap once | 183.166032 | 183.149620 | 184.459176 | −1.309556 |
| Sparse core/index/metadata | 1.816410 | 1.950577 | 2.116095 | −0.165518 |
| No recorded kernel | 0.651455 | 0.856122 | 5.255609 | −4.399487 |
| Small cross-family intersections, separately allocated | 0.007520 | 0.007520 | 0.038464 | −0.030944 |
| **Decoder wall** | **275.361500** | **275.016748** | **256.070688** | **+18.946060** |

This is 7.3988% more time against the historical reference for this chunk.
Projection excess is 14.366099 ms, or 75.83% of the net difference. That is
accounting, not proof that all 14.366 ms is recoverable. MoE is the largest
absolute interval but not the largest observed excess. Different dates/clocks
and observers prevent a fresh causal framework-speedup claim. No current
complete-request tok/s follows from this table.

### Overlap, TP and chunk handoff

| Measurement, ms | Cold R0 | Cold R1 | Warm R0 | Warm R1 |
| --- | ---: | ---: | ---: | ---: |
| Decoder wall | 275.361500 | 275.016748 | 274.223265 | 274.119178 |
| TP union | 18.203554 | 16.758112 | 17.484461 | 15.543488 |
| Shared overlapped by other work | 107.910566 | 107.240019 | 108.216471 | 109.298029 |
| Shared uncovered | 0.139808 | 0.141410 | 0.098079 | 0.096929 |
| Compute/TP overlap | 0 | 0 | 0 | 0 |
| Decoder end → next chunk | 2.203513 | 2.564608 | 2.463544 | 2.858373 |
| Handoff with no recorded device activity | 1.861595 | 2.230807 | 2.127929 | 2.523773 |

Retained vLLM shared span was 7.273535 ms, overlap 3.212895 ms, uncovered
4.060640 ms. Native shared work is much heavier but mostly overlapped; resource
contention remains a separate question. Reference handoff was 4.011232 ms with
0.224352 ms uncovered; its intermediate head work differs, so this is not an
isolated scheduler comparison.

All 90 collectives matched across native ranks. Causal clock-offset uncertainty
was 0.224261 ms cold / 0.238429 ms warm. First cold collective R0/R1 durations
were 1.247420/0.630896 ms; R0 early arrival was bounded 0.492560–0.716821 ms.
Warm first durations were 0.314303/0.205797 ms. The first two cold dense-FFN
reductions on R1 were **0.181509/0.163364 ms**, not the old 3 ms stall.

The largest post-both-start remainder was bounded 0.530599–0.754860 ms cold
on R0 and 0.630896 ms on R1; warm bounds were smaller. These are not wire-time
measurements. Same-stream producer analysis rejected graph-created streams
without visible producers; collective-only analysis was used rather than
inventing dependencies.

### Latest projection experiment: traffic fixed, timing not fixed

Baseline correction: the earlier **4.46× instruction ratio belonged to a rejected
M64/N128/K32 candidate**, not the accepted serving path. The accepted leaf's
17,119,232 versus reference 2,293,848 warp instructions is about **7.46×**.
Neither ratio is a latency multiplier. [Correction and selected source](GLM_LOADER_ALLOCATIONS_AND_INSTRUCTIONS_2026-10-21.md).

Matched leaf: M64/N4096/K8192. Accepted ownership M32/N64; experimental M64/N128
retained the ascending K16 F32 fold and BF16 publication. The common counter-only
capture reported L2 read sectors **6,294,081 / 3,149,648 / 3,149,632** for accepted,
wide and installed reference. Requested DRAM metrics were absent: no DRAM claim.

The wider candidate's main capture reduced warp instructions from retained
17,119,232 to 8,271,232, still about 3.61× the reference's 2,293,848. It used
124 registers/thread, 49,152 shared bytes, 8.6% active warps and 14.548% tensor
activity; long-scoreboard/barrier shares were 52.824%/16.843%. These metrics do
not assign causal milliseconds to a resource.

Five alternating pairs, 20 graph replays per trial, yielded median paired
candidate/baseline ratios for rows1/3/22/36/64 of
**1.02448 / 1.01089 / 1.00963 / 1.02883 / 1.00504**. Row1 used an unchanged
decode path and shows trial variation. Neither wider candidate qualified.
The code was withdrawn, with the proposal retained in
[`glm_wide_fragment_traffic_20261022.patch`](../benchmarks/experiments/glm_wide_fragment_traffic_20261022.patch).
No sanitizer qualification or serving improvement was claimed for that loser.

### Reference loading, safety and restoration

The exact dummy workspace guard removed **2,181,038,080 bytes** while live
allocated bytes at the matched boundary stayed **95,119,741,440**. A later
completed-module cache release recovered another inactive 512 MiB. Nevertheless,
all three bounded reference starts retired before graph capture at the earlier
19 GiB observer / 18.5 GiB external guards. They did not OOM and did not produce
a fresh reference request trace. Remaining inner layer7 attention allocation
was not isolated. The campaign exhausted its retry budget.

Native retrace sampled GPU maxima 94.241/94.270 GiB and host minima
17.601/18.647 GiB; owned swap was zero. After diagnostics retired, both accepted
GLM user units were restored, health passed and `323` completed normally.
Restoration sampled GPU maxima 94.239/94.268 GiB and host minima
17.563/19.099 GiB, zero owned swap. This describes **October22**, not a new
live check on October24.

## 8. Evidence map for further investigation

Use reports as entry points to their exact raw roots and command variants.
Many `/tmp` roots are historical evidence locations and may not remain present;
do not claim fresh reproducibility without checking their availability.

| Topic | Primary additional reports |
| --- | --- |
| Complete Qwen selected-path comparison | [All-new-selected](BENCHMARK_ALL_NEW_SELECTED_2026-09-30.md), [framework gap](BENCHMARK_FRAMEWORK_GAP_DEEP_DIVE_2026-09-30.md) |
| Parallel/precision architecture | [Parallel IR](PARALLEL_IR.md), [precision IR](PRECISION_IR_2026-10-04.md), [architecture completion](COMPILER_ARCHITECTURE_COMPLETION_2026-09-29.md) |
| GLM expert source/counter differences | [Expert pipeline](GLM_EXPERT_PIPELINE_2026-10-17.md), [packed transport](GLM_PACKED_EXPERT_TRANSPORT_2026-10-17.md), [work and waits](GLM_EXECUTION_WORK_AND_WAITS_2026-10-17.md) |
| Early submission/graphs | [TP submission](TP_SUBMISSION_AND_GRAPH_REPLAY_2026-10-13.md) |
| Long-path time accounting | [Execution graph](GLM_EXECUTION_TIME_GRAPH_2026-10-19.md), [scales+bulk graph](GLM_SCALES_BULK_EXECUTION_GRAPH_2026-10-19.md), [latest Oct20 graph](GLM_LATEST_EXECUTION_GRAPH_2026-10-20.md) |
| Projection decomposition | [Large reuse](GLM_LARGE_PROJECTION_REUSE_2026-10-20.md), [sparse/HC](GLM_SPARSE_HC_2026-10-20.md), [row reuse](GLM_ROW_REUSE_PIPELINE_2026-10-20.md), [tensor transfer](GLM_TENSOR_TRANSFER_PROJECTION_2026-10-20.md) |
| Observer and loading evolution | [Deferred trace](GLM_DEFERRED_TRACE_2026-10-21.md), [loading follow-up](GLM_LOADING_AND_PROJECTION_FOLLOWUP_2026-10-21.md), [loader ownership/traffic](GLM_LOADER_OWNERSHIP_AND_TRAFFIC_2026-10-21.md) |
| Independent model checks | [Checkpoint correctness](INDEPENDENT_CHECKPOINT_CORRECTNESS_2026-10-11.md), [MiniMax text diversity](MINIMAX_DIVERSE_TEXT_ENCODER_2026-10-11.md) |

The historical sequence is not “all optimizations failed.” Large serial and
repeated-work defects were removed, selected paths were connected, memory and
TP causes were isolated, and many scoped measurements improved. The repeated
mistake was allowing that evidence to imply a broader result than it measured.
Future work should retain the successful methods and narrow the claims—not
reset the diagnosis to an old bottleneck or promise parity from one counter.
