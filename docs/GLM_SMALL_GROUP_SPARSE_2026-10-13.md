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
output tokens per request-second including prefill: 2.12005 native versus
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

Selected-route trace results are pending below.
