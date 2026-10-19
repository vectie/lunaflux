# GLM projection TP: deployment and bounded experiment

The qualified local-combine MoE artifact (`07d0ca98`, prepared root
`/tmp/lunaflux-glm-http-local-moe-20261019-v2`) is now installed in both durable
LunaNexa rank units. The prior `98c629fd` artifact and unit snapshots are retained.
Live backend health and a complete arithmetic answer (`323`, stop) passed.
Deployment evidence: `/tmp/lunanexa-glm-local-moe-install-20261019-v1`;
memory observation: `/tmp/lunanexa-glm-local-moe-observation-20261019-v1`.

## Measured next boundary

The retained **new MoE** trace, not the older partials path, contains 48 F32
non-MoE reductions: 34 recurrent output projections, 11 sparse output
projections and 3 dense FFN down projections. Their sum is 24.455936 ms;
the vocabulary gather adds 0.254816 ms. The reference has 48 BF16 reductions
following selected `nvjet` projection kernels totaling 9.208128 ms. Its
additional outer-window reduction (0.146944 ms) and gather (0.054368 ms) must
not be attributed to these 48 projections. The observation is a first 64-row
chunk of the literal 7,973-token request, not an overall benchmark.

Source and timeline agree: native projections publish F32 partials, reduce,
then round to BF16. vLLM's installed GLM KDA/sparse/dense owners use
`RowParallelLinear`; its `forward` reduces the local matrix output. The trace
confirms BF16 collective dispatch, not merely a source-level possibility.
Native site width is 4,096 and live rows are 64: 1 MiB per F32 site versus
0.5 MiB per BF16 site. Durations include peer-arrival wait; rank-1-only timing
does **not** identify wire time. NCCL protocol/library selection also differs.

## One explicit numerical alternative

`ProjectionPublication::ReduceBFloat16AfterRound` orders local BF16 publication
before the inter-rank sum. It is **not** a bitwise-equivalent reassociation of
`ReduceFloat32BeforeRound`. Existing defaults remain unchanged. The independent
HTTP startup option is `projection_reduction=bf16-after-round`; MoE selection
remains independent.

The immutable law flows through model builders, recurrent/sparse/dense
projection plans, owned frames and typed collective effects. Existing CUDA
lowering publishes BF16 directly; no new pack/cast kernel or vendor runtime
dependency is introduced. The borrowed output buffer is not closed by the
dense frame. Captured execution must reduce it before HC consumption. There
are no benchmark or reference-engine imports in production.

Budget: one measured TP alternative plus any necessary correctness repair.
Gates: affected tests, clean pinned ARM build, actual-source selection, complete
answers/lifecycle checks, and bounded same-workload GPU trace. No overall
benchmark, payload hashing, tolerance edits or reference-as-oracle claim.
The 100 GiB device limit, 16 GiB host reserve and zero owned swap remain active.
The candidate is not yet qualified or deployed; results will be appended.
