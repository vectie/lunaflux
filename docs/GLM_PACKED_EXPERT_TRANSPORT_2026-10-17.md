# Packed expert operand transport

This experiment separates two physical changes: expert-contiguous quantized
rows with the existing consumer schedule, then the same rows with independent
producer and consumer owners. At most these two schedules are evaluated.
The retained indexed 32-column schedule is the control. Wider columns and
changed numerical folds are not bundled into this experiment.

`grouped_matrix_ir` owns the stable job-to-padded-row materialization and its
bounded payload/scale regions. `physical_tile_ir` owns a finite publication/
retirement ring with disjoint producer and consumer owners. Numerical IR stays
unchanged. CUDA lowering chooses thread roles and synchronization instructions;
the model, routing, and scheduler contain no NVIDIA-specific decisions.

Quantization precedes byte-preserving row/scale packing. Gate/up reads the
packed input; its original-job-order BF16 publication is unchanged. Product
quantization then precedes packing for down. The two packed planes reuse one
startup allocation only after the earlier consumer retires. Global scale is
borrowed from the original quantizer. Invalid/padded rows are zero-filled.
No request-path allocation, host readback, comparison, JIT or hashing is added.

The CUDA experiment uses separate copy and MMA thread groups and two shared
slots. A producer waits for prior readers before overwriting a slot, completes
its asynchronous copies before publication, and proceeds to the other slot.
Consumers wait for publication, preserve ascending K arithmetic, and release
the slot after their last read. Terminal draining closes both slot epochs
before scratch reuse between projections or work tiles. This is not TMA and
must not be described as reproducing CUTLASS's implementation.

Measure the entire grouping/quantization/packing/gate-up/down/combine chain,
not matrix-only timing. Require exact retained outputs and quantized operands,
independent padded-row mapping tests, tails, sanitizer/leak checks, actual
generated-source selection, counters and unprofiled full-model timing. Compare
vLLM outside the compiler; record differing KV/timing policies and output counts.
No favorable microbenchmark alone promotes the default.

Both original GLM containers may be stopped temporarily and must be restored
unchanged afterward. Benchmark owners remain below 100 GiB per Spark, with
16 GiB independently observed host reserve and no task swap. The original
restored service already falls below that reserve; do not confuse restoration
of its unchanged configuration with a compliant benchmark. Preserve NFS and
all unrelated dirty source/toolchain files.

## Implementation and gates

Both transports are explicit, nondefault startup choices, propagated through
GLM recurrent/sparse stage construction, exported symbols, launch metadata,
workspace budgets and frame binding. `LUNA_GLM_EXPERT_TRANSPORT=packed|pipeline`
selects the same policy in the offline preparer and full-model runner. The
ordinary indexed policy is unchanged. The two extra launches materialize
quantized input and product rows; shared experts and arithmetic stay unchanged.

The first named-barrier implementation is rejected: wrapped K-tail racecheck
reported shared-memory hazards despite bitwise outputs. Its earlier timings
are not a validated winner. Phase-counted publication/retirement barriers
replace that handshake. The wrapped 704/576, three-chunk probe v2 passes
memcheck/full leak, racecheck and synccheck at R3/capacity under 2 GiB/no swap.
The final full-width producer-consumer-v3 probe passes all four gates at R1
(memcheck capacity; race/sync trimmed) under 8 GiB/no swap. No check is disabled.
Racechecker CPU workers and outstanding launches are bounded to one; this
reduces checker bookkeeping, not intra-kernel synchronization coverage.
The failed full-width 8 GiB and wrapped 2 GiB checker OOMs remain retained.

Correctness now clears candidate scratch and poisons outputs before execution,
compares retained output/quantized workspaces directly, and independently
checks both packed payload/scale planes against the stable job mapping. No
hashing is involved. CPU validation: 44 compiler/source/frame tests and 27 GLM
integration tests pass. Targeted info has 1084 pre-existing migration warnings,
zero errors; warning-denied check passes with the unchanged migration warning
exclusions. Unqualified check still fails on those existing migration warnings.

Raw roots: `/tmp/lunaflux-glm-fp4-performance-producer-consumer-20261017-v{1,2,3}`,
`/tmp/lunaflux-glm-fp4-performance-pipeline-wrap-20261017-v{1,2}` and
`/tmp/lunaflux-glm-fp4-performance-packed-20261017-v1`.
Full-model and final counter results follow below; no default promotion.

## Corrected pipeline component result

The final v3 source uses phase-counted barriers with explicit invalidation
before shared storage reuse. All eight R1/R3/R22/R64 × capacity/trimmed trials
pass bitwise chain/workspace comparison and the independent packed-row oracle.
At R22/capacity the median complete chain changes 6.796→7.040 ms (+3.58%);
R64/capacity 13.182→13.577 ms (+3.00%). The earlier unsafe named-barrier timing
is explicitly excluded. Packed-only also regresses; its R3 full-bank capacity
memcheck/leak/race/sync gates all pass with bounded checker bookkeeping.

Selected v3 Nsight replay (same synthetic 288 distinct banks and actual
generated native kernels; not a serving trace):

| Metric | Indexed gate/up | Pipeline gate/up | Indexed down | Pipeline down |
|---|---:|---:|---:|---:|
| GPU replay ms | 4.180 | 4.516 | 2.571 | 2.145 |
| Warp instructions, millions | 96.53 | 282.30 | 49.42 | 134.78 |
| L2 read sectors, millions | 36.49 | 58.44 | 18.23 | 29.22 |
| Barrier stall % | 19.23 | 2.63 | 12.09 | 8.96 |
| Long-scoreboard stall % | 76.84 | 59.39 | 60.08 | 70.03 |
| Active-warps occupancy % | 45.64 | 47.48 | 80.86 | 63.75 |
| Registers/thread | 71 | 54 | 48 | 62 |
| Static shared bytes/block | 13,824 | 27,712 | 6,912 | 13,856 |

This is not a win from lower barrier counters. The generated phase waits
execute retry loops and copy owners still issue many `cp.async` instructions;
they are not the installed CUTLASS bulk-TMA producer. Packed padding also
changes zero-fill copies into reads of materialized zero rows. The higher
instruction/read counts and doubled shared storage are measured costs; their
individual contributions to elapsed time have not been independently isolated.
Down's replay benefit is insufficient to win the unprofiled complete chain.
Raw counters: v3 `matrix.ncu-rep` and `matrix-raw.stdout`.

The retained installed FlashInfer CUTLASS R22 component capture is
`/tmp/lunaflux-glm-cutlass-component-r22-20261017-v3`: gate/up 3.247 ms,
down 1.745 ms, 79.03M/64.37M instructions. It uses 128×128×256 tiles, 48
persistent CTAs, one producer and two consumer warp groups. Native uses 16-row
groups, 512 CTA workers and one consumer group. The component shares logical
shapes/banks but has different physical padding. It is not proof of the exact
kernel invocation in vLLM serving, and cannot account for the whole-model gap.

Implementation `e461b17f` is pushed. Its clean commit-pinned ARM native build
passes with 786.5 MiB reported peak and zero swap. The full-model export logs
explicitly select `expert_operand_transport=pipeline`; unprofiled full-model
validation remains below. No experiment has been promoted to the default.

## Full-model native run

Clean pinned runtime/AOT:
`/tmp/lunaflux-glm-expert-tp-pipeline-20261017-v1`.
Matched run:
`/tmp/lunaflux-model-bench-glm-expert-pipeline-20261017-v1`.
Both export and execution select the explicit pipeline mode; recurrent and
sparse routed expert families contain the new packing symbols and phase waits.
Both TP ranks close CUDA successfully. Planned bytes are 99,441,446,188 and
100,710,859,052, below 100 GiB each; cgroup caps remain 100 GiB/no swap.
Observed host minimum is 21.8604 GiB. Startup and one warmup are excluded.

| Prompt input tokens | Output tokens | Retained completion ms | Pipeline completion ms | Pipeline TTFT ms |
|---:|---:|---:|---:|---:|
| 22 | 4 | 577 | 604 | 327 |
| 23 | 3 | 507 | 513 | 320 |
| 22 | 8 | 935 | 932 | 310 |
| 23 | 8 | 949 | 959 | 333 |
| 24 | 8 | 954 | 963 | 337 |
| 36 | 3 | 585 | 601 | 408 |

All six native output ID arrays are unchanged. Completion totals 4.572 s vs
retained 4.507 s (+1.44% in these cohorts); TTFT totals 2.035 s vs 1.992 s.
This single full-model cohort establishes no reliable win. The independent
alternating-order component trials also regress. Leave the indexed default
unchanged rather than introducing a default slowdown. These short C1 prompts
do not establish long-context or concurrent-serving performance or quality.

Full native suite: **5165/5165 pass**, with unchanged migration exclusions
`-20-25-29-35-79-92`. Warning-denied native check and targeted formatting pass.
No unrelated dirty migration/TLS files were committed. The script regression
tests pass (probe 1/1, benchmark runner 3/3).

## Fresh reference: still a regression, not a closed gap

Fresh reference root:
`/tmp/lunaflux-reference-bench-vllm-glm-transport-20261017-v1`.
All six requests complete with HTTP 200 after one excluded warmup. Times are
.450244/.390083/.652980/.668743/.671485/.373363 s. The external comparison is
`comparison-fresh-reference.json` in the native run root. Every native token
array remains identical to the retained native array.

The new reference fact answer emits five tokens, not the four in the earlier
reference cohort; arithmetic remains four versus native three. Do not reuse
the earlier five-case equal-count subset. The four equal-reported-output-count
cases (Chinese/JSON/code/retrieval) total **3.455 s native vs 2.366571 s vLLM**:
native completion is **45.99% longer**. The six-case totals are 4.572 vs
3.206898 s (42.57% longer), but contain unequal generated work. Even the
equal-count subset is not identical hidden-state work or a quality certificate.
Native timing includes CLI completion/retirement; reference uses buffered HTTP.
Native BF16 expanded KV and reference FP8 MLA KV remain different.

Reference minimum observed host available memory is 17.0544 GiB; native minimum
21.8604 GiB. Reference containers are capped at 96 GiB/no task swap, native at
100 GiB/no task swap. The query controller requests shutdown after all responses;
head exits zero and worker 137. The watcher subsequently flags the stopped
worker, with both hosts already above 117 GiB available. This is deliberate
teardown, not a demonstrated OOM or a reserve breach. Both GPU owner lists are
empty before restoration of the original containers.

## Whole-path source review, not a single-counter diagnosis

The user explicitly requested all relevant differences, not one attractive
explanation. The audit follows the current selected AOT (`stage-0.cu` in the
pipeline runtime root), the retained selected native timeline, `../vlm`, and
the **installed** reference backend. The sibling checkout does not contain
the installed GLM5-Next model class. Its common FlashInfer adapter is useful,
but GLM-specific conclusions use the installed-image source snapshots in
`/tmp/lunaflux-glm-gap-diagnosis-20261014-v1/reference-{model,glm-kda,glm-attention,moe-runner}.py`.
The running reference reports `0.1.dev20051+g487ecf187`,
`FLASHINFER_CUTLASS` and `FLASHINFER_MLA_SPARSE_SM120`. Reading an unrelated
Marlin/Machete kernel would not explain this selected path.

| Boundary | Current native implementation | Reference implementation / evidence | Consequence and remaining uncertainty |
|---|---|---|---|
| Routing/grouping | Stable ranks; no former repeated group search | Fused-MoE adapter delegates route permutation to FlashInfer | The measured native grouping cost already fell 15.896→.173 ms; it is not the remaining major prefill cost. Exact reference route timing is missing. |
| Row materialization | Writes all 16 rows per live expert group, including zeros; two additional launches | Installed `expandInputRowsKernel` iterates `num_valid_tokens`, duplicates/permutates rows, and prepares scale layout | Different packing domains and padding costs. Reference compute tiles also have padding; do not claim reference has no padding. |
| Copy instruction granularity | 128 producers issue individual 16-byte `cp.async` operations and explicit readiness/retirement waits | Installed SM120 collective elects one issuing lane and makes TMA copies for A/B/SFA/SFB | We implemented role separation, not bulk transfer. Instruction count rises despite lower barrier stalls. |
| Compute geometry / reuse | 16×32 output ownership; one consumer group; 512 workers in the component | Captured installed component: 128×128×256, two consumer groups, 48 persistent CTAs | Useful work amortized per scheduling/copy action differs. Wider geometry alone previously failed; the transport and compute geometry must be costed together. |
| Two levels of pipelining | Two shared slots, direct fragment loads followed by MMA | Installed collective has shared-memory stages **and** `copy_kblock(next)` before `gemm_kblock(current)` | Global-to-shared overlap is not the same as shared-to-register/MMA overlap. The new ring does not reproduce both. |
| Quantization | Separate dynamic global-amax and block-pack kernels before each matrix stage | Adapter passes activation global scales; FlashInfer has integrated expansion/quantization and activation-quantization alternatives | Native selected global reductions cost 8.151 ms across routed prefill layers. Scale laws are numerical choices, not interchangeable performance toggles; exact executed reference fusion remains to capture. |
| Gate/up and epilogues | Retained independent gate/up operand traversals; materialized BF16 product, second quantizer/packer, per-route down outputs | Packed FC1 weights and FlashInfer activation/finalization paths; configuration-dependent epilogue fusion | Extra storage/publication boundaries are real. Do not assume every available reference epilogue fusion is selected, or silently change BF16 round placement. |
| Shared experts | Separate BF16 program, appended after routed work on one ordered stream | Common vLLM runner has auxiliary-stream shared-expert overlap with backend/shape guards | Potential critical-path overlap, not established overlap in this measured serving invocation. Native shared prefill costs 29.179 ms in retained trace. |
| Non-expert matrices | Prefill still uses synchronous 16×64×32 WMMA, two loop barriers per K32; decode uses subgroup GEMV | Model delegates merged projections to its linear backend | Connecting the shape classifier did not bring the mature multistage prefill lowering into these operators. Exact selected vendor linear schedule is not captured. |
| KDA fusion | Six projections share one launch via disjoint virtual matrix regions; each region still owns independent operand loads | Installed GLM class uses one physically merged `in_proj_qkvbfg_a`, fused QKV convolution and in-kernel decode controls | Launch fusion is not shared-operand matrix fusion. Native recurrent family remains 34.731 ms prefill /24.567 ms decode in retained trace. |
| KDA algorithm selection | This actual AOT selects ordered-token recurrence; the slower chunk-triangular alternative remains unselected | Installed reference dispatches chunk prefill and fused recurrent decode | Chunk code existing in the repository is not selected optimized coverage. Retained native recurrence itself costs only 1.457 ms prefill /.711 ms decode; projections dominate this short case. Long-prefill conclusions need longer inputs. |
| Sparse projections and cache | Separate q-a/kv-a, norms and index projections; materializes per-head BF16 K/V history | Fused q/kv-a and QKV RMSNorm; MLA latent KV backend; FP8 cache selected | Structural work/data representation differs, not only copy speed. The indexer still recomputes head gates in F32 despite a merged weight module; do not overstate reference fusion. |
| Sparse attention itself | Retains scores once, but per-head selected-list validation/compaction, serial component dot and ordered denominator remain | Selected FlashInfer sparse MLA backend | The earlier single-thread/repeated-score defect is fixed, not every sparse-attention cost. Current 22–36-token test cannot quantify the long-context effect. |
| Hyper-connections | Explicit square/projection/control/Sinkhorn/norm/post buffers and launches per sublayer | Installed model fuses post-attention HC-post + FFN HC-pre + norm; defers FFN HC-post into the next layer | Missing cross-boundary fusion, independent of expert TMA. Retained native trace has 474 hyper calls/step, 17.306 ms prefill /12.790 ms decode. These are whole-family times, not all recoverable overhead. |
| TP reduction domain | One joined message preserves eight per-expert F32 lanes plus shared lane | Reference MoE runner can combine local routed/shared results before final all-reduce | Equal collective count does not imply equal bytes or rounding. Native joined width is **9×4096 F32 values/token**, not one hidden vector. Exact selected reference message count/dtype remains unmeasured. |
| Communication transport | NCCL with the configured Spark RoCE interface/HCA | Fresh log selects `PYNCCL`, not a custom NVLink all-reduce; launch specifies the same interconnect family | Do not attribute the gap to a hypothetical proprietary communication path. Requested HCA configuration is not proof of negotiated transport/bandwidth; collect actual NCCL transport and rank skew in a bounded diagnostic. |
| Graph / host submission | 91 compute graphs separated by 90 eager queued collectives, terminal retirement | Reference uses breakable/full CUDA graphs and backend-specific collectives | Native graph replay and queued submission are present. Larger captured regions may reduce submission costs, but no evidence of the old per-site whole-stream wait remains. |
| Text/head placement | Parallel full-vocabulary head/greedy owned only by the last TP rank | Installed model constructs `ParallelLMHead`, distributing vocabulary rows | Removing the old serial greedy did not distribute our head. Potential rank-tail imbalance remains; the reference gather/reduction cost must be included. Native text family costs 6.198/5.850 ms prefill/decode, not all head work. |
| Serving boundary | Resident diagnostic CLI, bounded fixed row capacity | HTTP scheduler and graph buckets | HTTP-vs-CLI, output counts, cache format, C1 and short generation constrain comparison; there is no current concurrency or long-context conclusion. |

This inventory is not a claim that every source difference is bad or every
reference capability executes. It identifies missing equivalences and the
measurements still needed. The existing native trace is the retained indexed
path, not a new full-model pipeline trace. Its ranked prefill families are
routed 121.275, recurrent 34.731, shared 29.179, sparse 25.283, NCCL 23.377,
hyper 17.306, text 6.198, dense 3.900, router 2.883 and combine 1.582 ms.
Decode ranks differently: recurrent 24.567, NCCL 22.093, routed 15.449,
sparse 12.900, hyper 12.790, shared 6.547 and text 5.850 ms. NCCL includes
peer waiting. Do not sum both ranks, add stall percentages, or convert these
profiled family sums into fractions of the unprofiled 45.99% gap.

### Concrete deductions and limits

1. **The new packing policy amplifies its own input work.** The recorded R22
   fixture has 176 route jobs and 155 live groups: 2,480 physical rows for
   176 useful rows (7.10% row occupancy). The old loader masks `job < 0` and
   issues zero-fill copies with no global read for those rows. Packed loading
   uses `valid = local >= 0`, so it reads the materialized zeros. This is an
   exact source/domain deduction, not a claim of 14.09× total DRAM traffic:
   weight traffic and cache reuse are different. Measured total L2 reads rise
   about 60%; attribution of that increase still needs an isolated control.
2. **Role separation alone is insufficient.** Gate/up barrier stalls fall
   19.23→2.63%, yet instructions rise 96.53M→282.30M and GPU replay time rises.
   Retained arithmetic and outputs are unchanged. We have moved costs into
   transport/control while leaving a small consumer workload. The installed
   TMA/consumer/lookahead combination addresses multiple costs jointly. That
   does not prove TMA alone would fix them.
3. **Our KDA “fusion” is weaker than the reference's merged matrix.**
   `tiled_matrix_cuda_source/segmented.mbt` calls separate device regions from
   one kernel; it saves launches but does not make their input operands one
   shared matrix tile. The 6→1 launch count is not evidence of equal reuse.
4. **Our collective coalescing is not output reduction.**
   `packed_expert_mlp/program.mbt` joins `hidden * selected` and `hidden`.
   Actual AOT `glm_moe_combine_3_collective_pack` writes `rows * 36864` words.
   At H4096/top8 this is 147,456 bytes/token/MoE site before network-protocol
   accounting. Combining to H F32 would be 16,384 bytes, but changes the
   reduction/publication order. The source gap is real; a 9× network-time or
   end-to-end speedup is **not** implied. A separate declared numerical law
   and output validation would be needed, not a hidden reassociation.
5. **No exact additive explanation of the whole cross-framework gap exists
   yet.** Native family traces, component counter controls, and full-model
   timings answer different questions. The memory-safe live reference trace
   has not succeeded. Reporting “45.99% = X% loading + Y% barriers” would be
   invented. The code audit nevertheless gives testable alternatives rather
   than another guess from a single stall counter.

Two qualifications prevent further overstatement. First, native expanded
history in the actual layer-3 AOT stores 8,192 BF16 keys and 8,192 BF16 values
per token/rank, plus indexer state. The reference wrapper passes compressed
KV to MLA. The difference is therefore representation **and** precision, not
just BF16 versus FP8 bytes; its payoff depends on history length and execution
algorithm. Second, the chosen native KDA delta body is token-sequential and
uses three convolution invocations per layer. The old chunk experiment lost
at 16/32 rows and was deliberately not promoted
([results](GLM_CHUNK_KDA_2026-10-13.md)). The current source's two largest KDA
projection roles take 22.255+9.271 ms prefill and 16.660+5.896 ms decode;
the delta update takes only 1.457/.711 ms. Treating recurrence as the dominant
short-prompt cause would contradict those observations.

These checks also rule out stale diagnoses: grouping is no longer quadratic
route rediscovery, non-expert decode no longer always computes 16 padded rows,
greedy is no longer one thread, and queued TP no longer waits for each whole
stream before enqueueing the next collective. They do **not** mean all small
reductions or metadata kernels are parallel: selected Sinkhorn, some norms,
and sparse-list compaction still contain serial sections. Their actual family
cost and launch context matter more than the presence of a serial loop.

### Compiler direction and next bounded tests

Keep comparison outside the compiler. Numerical IR owns scale, reduction and
rounding laws; region IR owns legal shared-input and deferred-publication
composition; physical plans own valid-row domains, materialization, transfer
granularity, consumer geometry, two-level operand lifetimes and persistent work
assignment. Parallel IR owns independent branches, collective domains and
capture boundaries. CUDA alone implements TMA descriptors, warp roles and
barriers. More IR names without those executable compositions would not help.

Before further optimization, test complete regions, not isolated attractive
counters: (a) separate padding traffic from phase-polling costs on identical
packed data; (b) evaluate bulk-transfer + consumer/lookahead geometry including
packing, scale conversion and epilogues; (c) independently measure real merged
KDA/BF16 matrix lowering and HC post/pre fusion; (d) measure collective bytes,
rank arrival skew and graph cuts before changing reduction semantics; (e)
confirm reference shared-stream/finalize selection rather than assuming it.
For latency attribution, use matching logical boundary timestamps and critical
paths, not sums of overlapping kernel durations. Do not launch more profiler
loads that previously breached the reserve without first reducing the capture's
memory overhead. Long history, greater prefill rows, skewed routing and mixed
batches need separate cases; six short C1 prompts cannot select a universal
winner. In particular, capture real per-layer expert occupancy and quantized
operands: the synthetic R22 155-group distribution is a useful control, not
proof of the model's actual distribution. Separate fixed-work replay from
free-running output-quality tests, and report repeated latency distributions
rather than choosing the best cohort. These are identified next experiments,
**not implemented wins**.

## Original GLM restored

After both benchmark GPU-owner lists became empty, the original worker
`785b9b6a03b9` was restarted at 15:34:23 CST and the original head
`2bd8297b9569` at 15:34:24. They were not recreated or reconfigured. At
15:42:57, `/health` has returned HTTP 200 and `/v1/models` reports
`GLM-5.3-Flash-EXL3` with its original 850,000-token maximum. Both original
containers are running with `OOMKilled=false`; `glm53-nfs` has remained
running since October 14. Only the original vLLM workers own GPU memory.

Restoration is not another benchmark: the original EXL3/DFlash2 configuration
remains unchanged. This snapshot uses 102,144/102,142 MiB GPU allocation and
has 9,549,144/11,665,040 KiB host memory available, about 9.1/11.1 GiB. Its
original configuration is below the benchmark's 16 GiB host reserve. The task
benchmarks respected that reserve; no benchmark remains running beside the
restored service. The requested default preservation/restoration does not
justify claiming that the original deployment meets the benchmark envelope.
