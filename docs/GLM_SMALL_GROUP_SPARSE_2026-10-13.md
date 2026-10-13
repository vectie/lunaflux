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

Full-model results and selected-route counters are pending.
