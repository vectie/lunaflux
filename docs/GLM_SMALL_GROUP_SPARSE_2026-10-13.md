# Small-group and sparse-chain execution

Starting point: `e5e1ce8f`, matched six-prompt completion 21.016 s versus
vLLM 3.144176 s (6.684x). This is PP2 versus TP2, not equal parallel execution.
Single-request PP2 runs successive layer ranges on different GPUs; TP2 shards
each layer and joins its partial results. Pipeline capacity does not imply
simultaneous use of both devices for one token. The current host-staged PP edge
also does not use CX7 as an intra-layer collective. Kernel and topology gaps
must be reported separately; neither is inferred as a fixed speedup multiplier.

The selected 16-input/2-output trace attributes 433.744 ms to routed experts
and 379.340 ms to the **whole sparse chain**. Its named output/Q-B projections
remain scalar; index selection still repeats scoring during each top-k pass.
The attention probability/value kernel is not synonymous with that whole sum.

Plan: at most two expert schedule variants and two sparse-chain variants,
followed by selected-route, matched full-model and hardware-counter checks.
Keep arithmetic authorization in precision IR; tile ownership in pure physical
plans; CUDA instructions and synchronization in private lowering. Comparisons
remain offline. Do not silently switch matrix arithmetic to scalar arithmetic
for small groups. Preserve tails, explicit ownership and deterministic expert
combination. No payload hashing. Probes: 2 GiB/no swap. Full model: 96 GiB per
Spark/no process swap, minimum 16 GiB host available memory, exclusive GPUs.

vLLM source inspection distinguishes packed FlashInfer CUTLASS NVFP4 execution
from our decoded-BF16 matrix alternative. Its generic grouped path also adjusts
tile geometry/work assignment for small token counts. These are implementation
ideas, not permission to change activation quantization implicitly. Sparse MLA
also uses compressed FP8 KV in the reference, unlike our expanded BF16 history.

## Local experiments

No expert schedule winner was found. All three alternatives are rejected and
the prior N64/K32 matrix schedule is retained. On the full-width synthetic
expert bank (five alternating scalar/matrix trials, same dyadic correctness
fixture), median matrix times in milliseconds were:

| schedule | one row | 16 rows |
| --- | ---: | ---: |
| previous vector-decoded N64/K32 | 1.414 | 4.588 |
| N16/K64 shared operands | 1.931 | 7.023 |
| N64/K128 shared operands | 1.867 | 7.180 |
| N16/K16 direct register gather | 3.065 | 13.166 |

Roots `/tmp/lunaflux-glm-grouped-performance-20261013-v5`, `v7`, `v8` retain
the generated sources, unsuccessful timings and dyadic comparisons. `v4`/`v6`
retain exporter-check failures. Prior timings are historical, not a newly
paired old/new comparison. The scalar controls stayed near 0.95/13.5 ms.
After the initial two non-wins, a scoped one-row NCU capture of v7 showed
109,151,744 warp instructions, zero local sectors, 2.481% tensor activity,
40,613 long-scoreboard samples versus 3,296 barrier samples. This motivated
one additional register-ownership probe, not an unlimited search. Its failure
means staging removal alone is not a demonstrated remedy.

Retained index scores are a separate measured improvement. The 16-query,
256-history, 4-head, dimension-32 probe compares the actual renderer against
the retained ordered oracle, and checks stable ascending ties independently.
Ten boundary/mask cases match bitwise; full memcheck/leak, racecheck and
synccheck pass at 2 GiB/no swap. Its all-visible timing is ~105.88 ms serial
versus ~0.19485 ms retained. This is a deliberately complete candidate-set
microbenchmark, **not a model speedup**. Root:
`/tmp/lunaflux-glm-retained-index-20261013-v1`.

The new physical index plan assigns independent queries/candidates and sizes
retained scores/availability. The ordered effect sequence permits reuse of
the later attention scratch allocation. Startup budget checks include the
maximum of both lifetimes. Stable selection, causal/tail filtering and
component/head arithmetic remain unchanged. Sparse projection precision now
explicitly selects existing matrix lowering for all eight projection leaves
and output, with launch metadata propagated into the prepared frame. Scalar
projection remains the explicit default; the `tiled-grouped` profile selects
matrix arithmetic. No reference dependency enters either compiler plan.

## Matched full-model result

Committed runtime/AOT `918c64a0`, prepared at
`/tmp/lunaflux-checkpoint-build-glm-sparse-matrix-20261013-v1`;
native run `/tmp/lunaflux-model-bench-glm-sparse-matrix-20261013-v1`.
The same-day vLLM result is **retained**, not rerun in this iteration:
`/tmp/lunaflux-reference-bench-vllm-glm-tiled-20261013-v1`.
The comparison checks literal input token IDs, maximum eight output tokens
and identical stop IDs. Each row is one timing sample, not a confidence interval.

| Prompt | Input/output | Previous s | New s | Retained vLLM s | New TTFT s |
| --- | ---: | ---: | ---: | ---: | ---: |
| fact | 22/4 | 2.632 | 2.100 | .383396 | 1.217 |
| arithmetic | 23/4 | 2.697 | 2.099 | .393215 | 1.213 |
| Chinese | 22/8 | 3.926 | 3.226 | .652706 | 1.185 |
| JSON | 23/8 | 4.066 | 3.293 | .667366 | 1.245 |
| code | 24/8 | 4.118 | 3.324 | .672706 | 1.273 |
| retrieval | 36/3 | 3.577 | 2.467 | .374787 | 1.871 |
| total | 150/35 | 21.016 | **16.509** | **3.144176** | |

Completion time falls 21.4456% (1.2730x throughput for this fixed workload).
The remaining completion-time ratio is **5.25066x**, not parity. Aggregate
output tokens per request-second including prefill: 2.12006 native versus
11.13169 reference. These are **not decode-only token rates**. Four native
token sequences are unchanged; Chinese and JSON differ after the explicit
matrix projection selection. Chinese now matches the retained reference text;
agreement is not a proof of correctness. The eight-token cap truncates some
answers, so this campaign cannot establish broader output quality.

Native minimum host available memory: **22.731735 GiB**. Both user units exit
zero, explicitly close CUDA, report zero process swap, and release GPU owners.
The comparison preserves the timing/topology caveats: native resident CLI
includes completion output/retirement and its first request is cold; vLLM
buffered HTTP excludes one warmup. Native PP2/BF16 expanded history differs
from vLLM TP2/compressed FP8 MLA. No payload hashes were computed.

## Parallel execution conclusion

**Not on par for single-request latency.** The native `TwoRankDecoderPipeline`
owns one step at a time. `ActivationLink` waits for producer completion, then
performs device-to-host, socket transfer and host-to-device publication before
the downstream stage can consume that activation. Cooperative control/retirement
overlap is not simultaneous execution of both stage ranges for the same token.
There is no selected GLM intra-layer all-reduce on this path.

vLLM's KDA source divides heads/projection width by TP size, uses merged
column-parallel input projections and row-parallel output. Its MoE backend
receives TP rank/size and packed NVFP4 weights/scales. Relevant source:
`../vlm/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`,
`../vlm/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py`.
The generic small-M heuristics in `fused_moe.py` are design references, **not
the selected NVFP4 backend's measured schedule**.

The next architectural work is executable tensor partitioning for the GLM
operator graph: immutable output/input/head partitions, quantization-block-aligned
weight slices, explicit collective effects and rank-local KV/state ownership;
then bind the existing collective backend. It must not be simulated by renaming
PP2 to TP2. Separately, native packed-quantized matrix consumption needs an
explicit activation-quantization contract; decoding every weight to BF16 is
not the same algorithm as the selected reference backend. Neither improvement
has been implemented or benchmarked here, and a guaranteed 2x TP speedup cannot
be inferred without collective/overlap measurements.

## Validation

Affected native suite: 120/120 before the IndexShare lifetime refinement;
its additional scratch-focused tests pass 6/6. Native check with the existing
warning exclusions passes. `moon info`/format performed. Unexcluded full native
check/test were attempted and remain blocked by existing migration warnings
(including state-snapshot derived methods and unqualified scheduler tests).
The broader sparse integration filter is 3/6: the same previously documented
25-vs-23, 1312-vs-1184 and 41-vs-37 fixture expectations fail. No warning
suppression, fixture waiver or unrelated migration was committed.

Final source-package follow-up tests pass 5/5. The selected native run and
profile both retire their GPU owners; both Sparks are idle after the last
counter capture. Nonzero pre-existing host swap is not process swap: all
task units retain a zero swap limit and reported zero process swap.

## Selected-route trace: where the gain actually occurred

The new profile is
`/tmp/lunaflux-model-bench-glm-sparse-profile-20261013-v1`, using the same
committed `918c64a0` runtime and AOT as the six-prompt run. The comparison is
the retained `/tmp/lunaflux-model-bench-glm-repaired-profile-20261013-v1`.
Both profiles execute 16 input tokens and two output tokens, summed over both
PP stages. These are recorded GPU kernel durations, not end-to-end wall time
or a vLLM kernel comparison. `kernel-families.txt` and `sparse-breakdown.txt`
are produced from the actual ingress/egress Nsight Systems reports.

| Selected family | Before ms | After ms |
| --- | ---: | ---: |
| Sparse input projections | 168.478912 | 31.071392 |
| Sparse output projection | 142.951744 | 20.960800 |
| Sparse index | 57.272672 | 1.387168 |
| Sparse pooling | 1.824064 | 1.815104 |
| Sparse attention core | 3.932320 | 3.889600 |
| Sparse normalization/split | .515168 | .461728 |
| Sparse history/other | 4.365408 | 4.258400 |
| **Sparse chain total** | **379.340288** | **63.844192** |
| Routed experts | 433.743904 | 431.838400 |
| KDA projections | 156.117120 | 158.824512 |
| Dense FFN | 129.339616 | 130.058464 |
| Shared experts | 78.322208 | 77.698048 |
| **All recorded kernels** | **1319.028832** | **1003.280928** |

The sparse chain saves **315.496096 ms**, accounting for almost all of the
**315.747904 ms** reduction in summed kernel time. Thus the implemented matrix
projection and retained-score plans reach the executed path; the improvement
is not inferred from source presence or the much larger index microbenchmark
speedup. The attention core itself is essentially unchanged.

Routed experts now occupy **43.04%** of recorded native GPU kernel time, followed
by KDA projections (15.83%), dense FFN (12.96%), shared experts (7.74%) and the
sparse chain (6.36%). The small expert time difference between these captures
is not evidence of an expert optimization; its implementation is unchanged.
These percentages do not allocate the full 5.25x end-to-end vLLM gap: that
requires matched reference family durations and collective/overlap accounting.

## Matched one-row expert hardware counters

The final baseline capture is
`/tmp/lunaflux-glm-grouped-performance-20261013-v9/matrix.ncu-rep`. It explicitly
executes the preserved **v3 baseline binary**, with arguments `1 matrix`.
The rejected larger-K capture is `v7/matrix.ncu-rep`, also `1 matrix`. Both
observe `matrix_gate_up` with grid 4096 and block 128 on Spark .179. The earlier
v3 counter report used 16 rows and is not the comparator here. Captures run
serially under scoped administrator Nsight, with a 2 GiB/no-swap limit; no
driver setting changes or payload hashes are involved.

| Counter | Retained N64/K32 | Rejected N64/K128 |
| --- | ---: | ---: |
| Profiled gate/up duration, ms | 1.209856 | 1.644544 |
| Warp instructions | 120,527,872 | 109,151,744 |
| Global-load sectors | 9,332,736 | 4,515,840 |
| Local-load/store sectors | 0 / 0 | 0 / 0 |
| Static shared memory, KB/block | 18.43 | 49.15 |
| Shared-memory block limit/SM | 5 | 2 |
| Registers/thread | 48 | 48 |
| Achieved occupancy | 36.85% | 15.50% |
| Tensor activity | 3.384951% | 2.481178% |
| Memory throughput | 25.57% | 24.54% |
| Long-scoreboard sampled stalls | 41,936 | 40,613 |
| Barrier sampled stalls | 13,002 | 3,296 |

The larger-K variant reduces instructions, sectors and barrier samples, yet
increases execution time. Its larger shared allocation reduces concurrent
resident work and its observed occupancy/tensor activity fall. This supports
the resource-pressure explanation for this rejected candidate; it does not
prove every expert bottleneck is occupancy. PC sample counts are not percentages
of average warp latency, and replayed kernel duration is not the unprofiled
gate/up/down/combination chain time in the earlier table.

The remaining implementation gap is substantive: packed NVFP4 matrix consumption
and useful small-group ownership, rather than another unbounded N/K knob search.
That arithmetic requires explicit activation-quantization semantics. In parallel,
true layer-wise TP needs the partition/collective/state plan described above.
Neither is claimed complete by this sparse-chain change. The finite experiment
budget is closed with all three slower expert alternatives preserved and rejected.
