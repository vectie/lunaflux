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
  Native post-reduction publication totals only 0.216288 ms. The first
  analyzer version missed the three dense `_publish` names (reported
  0.196960 ms); their 0.019328 ms is now included, with a name regression
  test. The 48 collective durations were unaffected.

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

## First bounded node trace

Root: `/tmp/lunaflux-glm-http-projection-tp-trace-20261020-v1`.
The 500 ms collection includes one complete first 64-row chunk; anchors are
181,282,704–486,230,448 ns. The truncated final chunk is excluded. The request
drained successfully and both diagnostic owners retired before CPU-only
SQLite export. All **90 reduction kernels are now BF16** (48 projections plus
42 locally combined MoE); the one vocabulary gather remains unchanged.

| Observed GPU span, ms | New-MoE baseline / F32 projections | BF16 projections, first capture | vLLM retained trace |
|---|---:|---:|---:|
| 34 KDA projection reductions | 18.615424 | 8.489632 | 6.616512 |
| 11 sparse projection reductions | 4.281792 | 3.040992 | 2.014016 |
| 3 dense-down reductions | 1.558720 | 7.201696 | 0.577600 |
| **All 48 projection reductions** | **24.455936** | **18.732320** | **9.208128** |
| Post-reduction cast kernels | 0.216288 | 0 | 0 |
| 42 MoE reductions | 6.758272 | 7.285344 | 8.260864 |
| Vocabulary gather | 0.254816 | 0.265120 | 0.054368 |

The first-capture projection reduction span falls 23.4%, but dense layer 0/1
contain 3.788064 / 3.225632 ms waits. The next, incomplete chunk contains all
three dense sites at 0.193408 / 0.592800 / 0.168224 ms. This variability is
retained; it is not evidence that wire transfer consumes the long spans.
A repeated first-chunk capture is required before selecting the deployment.

The actual trace confirms propagation beyond export: direct BF16 projection
publication, BF16 NCCL entry points and no 48 post-collective casts. Actual
projection producers total 17.288192 ms versus 17.646976 ms before and
11.993984 ms in vLLM. Shared-expert work still overlaps routed work:
107.686464 ms overlap, only 0.110560 ms shared-only. Compute/collective
overlap is zero in this observed window. Kernel union is 297.776608 ms;
copy-inclusive coverage 297.816608 ms, uncovered 7.131136 ms, largest gap
3.147360 ms. These are execution-graph allocations, not whole-request timing
or throughput, and they do not separate peer arrival from collective execution.

Analyzer outputs:
`/tmp/lunaflux-glm-projection-after-20261020-v1.json`,
`/tmp/lunaflux-glm-projection-moe-after-20261020-v1.json`,
`/tmp/lunaflux-glm-projection-pass-20261020-v1`, and
`/tmp/lunaflux-glm-projection-overlap-20261020-v1`.

## Repeated capture and deployment decision

The identical artifact/request was captured again, without compiler or NCCL
tuning changes, at
`/tmp/lunaflux-glm-http-projection-tp-trace-20261020-v2`. One complete first
chunk lies between 132,595,552–437,838,496 ns. Both diagnostic owners retired
successfully. The 48 projection reductions again improve in total, to
**20.749760 ms** (15.2% below the retained 24.455936 ms baseline; the first
capture was 23.4% below). This is a two-capture observed range, not a confidence
interval or whole-model speedup.

| Repeated capture | ms |
|---|---:|
| KDA projection reductions, 34 | 8.954176 |
| Sparse projection reductions, 11 | 1.666592 |
| Dense-down reductions, 3 | 10.128992 |
| MoE reductions, 42 | 6.799872 |
| Vocabulary gather, 1 | 0.259840 |
| Actual projection producer kernels | 17.613920 |
| All collective spans | 27.809472 |
| Compute/collective overlap | 0 |

Dense layer 0/1 waits are 4.528736 / 5.446688 ms in this repeat; layer 2 is
0.153568 ms. The next incomplete chunk's same three sites are
0.485760 / 0.492192 / 0.494272 ms. Thus the early dense wait is repeatable
but not uniform across chunks. **It is unresolved**, not explained away as
random noise, and it must not be called a bandwidth or numerical-kernel bug
from this one-rank trace. The missing ingress GPU timeline still prevents
separating late peer arrival, peer computation and collective progress.
Inspection also confirms only decode has collective-inclusive replay; this
64-row prefill keeps compute-interval graphs and separate NCCL submissions.
No new environment forwarding or unrelated process-launcher change was made
merely to get a prettier profiling result.

Shared overlap remains 108.362912 ms, with only 0.104864 ms shared-only. The
kernel union is 300.588000 ms; copy-inclusive coverage 300.624832 ms,
uncovered 4.618112 ms, largest gap 2.967712 ms. The retained reference has
different outer anchors, so no overall/native-vLLM speedup is calculated.
Projection producer time also remains above the reference's 11.993984 ms;
halving collective payload cannot remove that compute difference.

Both captures remain within the memory envelope. Across watcher samples,
maximum GPU use is 96,489 MiB; minimum host availability is 18,264,876 KiB
(17.42 GiB). Owned swap is zero. SQLite conversion ran after GPU retirement.

The deployment decision retains this one explicit numerical alternative:
complete-answer/lifecycle checks pass without a new task failure, and both
bounded captures reduce total projection-TP span while removing 48 casts and
48 MiB of F32 arenas. This is **not a claim that all non-MoE TP overhead is
fixed or that vLLM parity is reached**. The first two dense waits remain a
specific measurement target. No overall benchmark or payload hashing ran.

## Serving restored

The source-pinned `eb152338` artifact is now installed in both durable roots:
`/home/wlc003s/.local/share/lunanexa/glm53-lunaflux-eb152338` and
`/home/wlc004s/.local/share/lunanexa/glm53-lunaflux-eb152338`. The previous
`07d0ca98` roots and exact unit snapshots are retained. Checkpoint files were
hard-linked, not copied/scanned; no payload digest was calculated.

Both `lunanexa-glm-lunaflux-rank{0,1}.service` units are active under the paired
lifecycle controller. Backend `/health` returned healthy, and a fresh live
17×19 request returned `323` with `stop`. The existing LunaNexa route and
controller deployment now name `eb152338`; controller rollout succeeded.
The existing 8,192-token shared capacity, eight resident slots and sixteen
admission lanes were preserved, not advertised as eight simultaneous 8K
requests. Private backend checks do not replace the missing fresh company-key
public-route acceptance check described above.

Deployment, rollback unit snapshots, accepted configs, live answer, route
patch and memory observation are retained under
`/tmp/lunanexa-glm-projection-release-20261020-v1`. The runtime label is an
existing declared artifact association, not a newly computed payload hash.
