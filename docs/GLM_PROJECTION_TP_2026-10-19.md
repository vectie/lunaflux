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
does **not** identify wire time. The reference symbol explicitly names
`RING_LL`; the native symbol names `RING` without a protocol suffix, so that
spelling alone does not establish the native protocol or its time contribution.

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
Qualification and trace results follow. A source option alone is not a
deployment or performance claim.

## Build and source checks

- Candidate `eb152338` is pushed. Warning-denied native check passes with the
  unchanged migration-warning exclusions. Affected tests: 130/130; packed
  execution tests: 135/135. Targeted interface generation: 1,065 existing
  migration warnings, zero errors. Formatting used the pinned Spark formatter
  to avoid unrelated toolchain migrations.
- Clean pinned ARM service build passed: 1.5 GiB peak, zero owned swap. Export
  logs on both ranks select `ReduceBFloat16AfterRound` and local MoE explicitly.
  Planned bytes drop by 50,331,648 (48 F32 output arenas) to 102,680,444,572.
- Both CPU-only AOT compilation units initially reached their 2 GiB caps. With
  both resident services stopped and over 110 GiB host memory available, only
  these compilation caps were raised to 4 GiB; swap remains disabled.
- Installed `linear.py` and `pynccl.py` were captured from the retained
  reference image, CPU-only, under
  `/tmp/lunaflux-glm-installed-projection-sources-20261020-v1`. RowParallelLinear
  reduces the local matrix output; PyNccl passes its dtype/numel directly.
  The attempted separate `quantization/unquantized.py` path does not exist;
  its failure is retained. The actual unquantized implementation is already
  in `linear.py`; no runtime reference source was altered.
- The external analyzer `scripts/analyze-glm-projection-reduction.mbtx` checks
  all 48 sites in model order, rather than subtracting unrelated head collectives.
  Baseline reductions: recurrent 18.615424 ms, sparse 4.281792 ms, dense
  1.558720 ms. vLLM: 6.616512 / 2.014016 / 0.577600 ms respectively.
  Native post-reduction publication totals only 0.196960 ms.

The company credential's temporary file is no longer present. An operator
model-list request correctly returns HTTP 401 (that authority is not a company
model-list key). No policy was changed. The deployment validation above is
backend validation, not a fresh authenticated public-route acceptance claim.

## Correctness-only serving qualification (20 October)

`/tmp/lunaflux-glm-http-qualification-projection-tp-20261020-v1`
completed and retired both diagnostic owners successfully. This used the
source-pinned `eb152338` service and its actual AOT artifacts from
`/tmp/lunaflux-glm-http-projection-tp-20261020-v1`, not a standalone kernel.
Both export and execution logs select `LocalCombinedBFloat16` plus
`ReduceBFloat16AfterRound`.

- The literal 7,973-token retrieval request returned exactly `orchid-70`,
  finishing with `stop` within its 128-token output budget.
- Fifteen broader tasks all terminated with `stop`, using a 512-token budget.
  Fourteen answers are identical to the qualified local-MoE baseline. The
  changed library summary preserves all supplied facts. Translation, the
  Python function and SQL semantics were manually reviewed. The existing SQL
  Markdown fence still violates the prompt's strict “only SQL” formatting;
  this is retained as a model instruction-following failure, not marked pass.
- HTTP smoke, streaming, routing/concurrency, cancellation and retirement
  checks completed. These checks establish observed serving behavior, not
  universal numerical correctness or reference-engine infallibility.
- Minimum available host memory: 18,818,196 / 20,487,280 KiB (ranks 0 / 1).
  Maximum GPU use: 96,451 / 96,480 MiB. Owned swap peak: zero. Both limits
  hold throughout the collected observation (100 GiB GPU, 16 GiB reserve).

Answers and grades remain under
`/tmp/lunaflux-glm-broad-policy-projection-tp-20261020-v1`; the long answer is
under `/tmp/lunaflux-glm-service-benchmark-policy-projection-tp-20261020-v1-answers`.
The historical script/directory name does not imply a latency campaign: this
run explicitly used `--correctness-only`, with no overall comparison/sweep.

The full warning-denied native suite also completed: **5,330/5,330 pass**, using
the existing `-20-25-29-35-79-92` migration-warning exclusions. These exclusions
were not widened. No new CUDA source or native ABI was added: the changed
placement binds existing BF16 projection lowering and typed BF16 collectives.
