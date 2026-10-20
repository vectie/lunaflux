# Large projection ownership experiment

Scope: selected KDA/projection kernels and both-rank execution graphs, not an
overall benchmark. Keep 100 GiB per Spark and 16 GiB host reserve, no hashing,
and restore the owned GLM service after the campaign. Comparisons stay outside
the compiler.

The preceding K128 campaign reduced instructions but regressed the square
matrices: lower residency, increased L2 traffic and load waits outweighed fewer
barriers. Do not repeat that blanket staging change.

The retained installed vLLM trace selects BF16 WMMA 32x32 output tiles, including
K128 single-stage kernels. Its installed KDA source merges six input projections;
LunaFlux already merges those same independent regions. Missing input fusion is
therefore not the hypothesis. The current native 16x64 tile loads each weight
tile again for each 16-token row group. A 32x32 tile halves that weight repetition
but doubles left-operand repetition. This is a testable tradeoff, not a proven
speedup. Both use four independent 16x16 accumulators and the same ascending K16
fold, with unchanged F32/BF16 publication and the existing single-token route.

Budget: two alternatives only, 32x32x32 and 32x32x64, on the existing eleven-shape
fixture, then selected counters and sanitizers for any winner. Resolve ownership
in immutable matrix IR; lowering owns warp coordinates and shared scratch.
An executable winner must reach launch metadata and fused scratch planning,
then be retraced on both ranks before any serving claim. The retained vLLM trace
is October 19, not a fresh memory-qualified reference run.

## Initial results and rejected extrapolation

Standalone root `/tmp/lunaflux-glm-projection-staging-row-reuse-20261020-v1`
uses the actual dense renderer and reserved decode grid. At 64 rows the K64
alternative improves paired medians 242.757→208.570 µs (4096×4096) and
478.608→417.400 µs (4096×8192). All eleven-shape outputs/inactive bytes match;
sampled independent double dots pass. This does not establish model quality.

The square-matrix NCU capture records 12.402M→8.549M warp instructions,
4.141M→3.443M L2 read sectors, 44.51→44.20% active-warp occupancy,
21.17→15.68% barrier stalls and 68.58→62.78% long-scoreboard stalls.
Registers remain40; total shared allocation including driver reserve rises
10,240→13,312 bytes. Profile time348.768→290.272 µs is distinct from warm
graph timing. Clock/cache control is off in both captures.

However, the actual segmented six-input body **regresses** at64 rows:
663.099→801.792 µs in
`/tmp/lunaflux-glm-projection-staging-fused-row-20261020-v2`. The first fused
root stopped at the external summary parser because the timing line omitted
`oracle_maxabs`; its GPU comparison succeeded, and its logs remain. The second
root adds that missing diagnostic field without changing generated kernels.
Do not promote the fused K64 schedule from standalone measurements. Its
resources are40 registers, zero local/stack bytes, 13,312 shared bytes including
driver reserve; old39/0/0/10,240. Whole fused-body counters are recorded below.

Short reduction4096×128 also regresses6.747→8.608 µs. A supplied minimum-inner
physical envelope now retains the default16×64×32 plan there; it is external
selection data, not a model-name test or numerical change. Tests ensure the
small KDA control projections remain byte-identical while large regions may
select row reuse. Other standalone K64 losses remain recorded, including
4096×1536,8192×1536 and16384×512. No whole-serving gain is claimed yet.

## Fused-body decision

The budgeted K32 alternative wins the complete six-projection launch:
720.282→633.515 µs median, all five paired gains6.57–14.12%. Absolute baseline
time varies between trials; retain paired gains rather than combining unrelated
medians. Both fused schedules pass exact output/inactive-byte checks at
0/1/3/22/36/64/65 rows for full and ragged geometry.

| Fused counter | Baseline16×64×32 | Reuse32×32×32 | Rejected32×32×64 |
| --- | ---: | ---: | ---: |
| NCU time µs |754.976|679.104|854.304|
| Warp instructions M |37.707|26.964|28.339|
| L2 read sectors M |14.627|11.261|10.424|
| Active warps % |71.642|71.666|51.651|
| Tensor activity, active-cycle % |7.316|8.093|6.672|
| Barrier stall % |22.255|22.230|12.129|
| Long-scoreboard stall % |66.785|70.049|64.529|
| Registers/thread |39|48|40|
| Shared bytes including driver |10240|9216|13312|
| Register-limited blocks/SM |12|10|12|
| Shared-memory-limited blocks/SM |10|11|7|

The K64 fused loss coincides with much lower residency despite fewer reads and
lower stall percentages. The winning K32 schedule retains residency, reduces
support instructions and traffic, and finishes earlier even though its load-wait
percentage rises. No spilling was recorded. This is why neither a stall
percentage nor standalone projection timing is a sufficient selection criterion.
The limiting resident-block counts are10→10→7: the K32 winner becomes
register-limited, whereas K64 is limited by shared memory. These are the
recorded launch-resource limits, separate from measured active-warp occupancy.
The serving candidate is32×32×32 for recurrent projections with K≥4096,
not K64, and not a change to sparse/expert schedules. Default selection stays
unchanged; the selected serving evidence below is not all-shape qualification.

Decode is not declared faster: fused one-row K32 trials are inconclusive, and
standalone4096×4096 one-row medians128.279999→128.472µs show a small regression
(median paired gain−0.264%). The arithmetic prefix is unchanged, but a kernel's
compiled resource allocation also applies to its decode branch. This experiment
therefore qualifies long-prefill ownership, not a universal decode replacement.

## Propagation and qualification

Implementation commits `58a2ed58` and `29b3f306` carry row ownership and the
minimum reduction extent through immutable `MatrixTilePlan`, segmented scratch,
recurrent source launch metadata, decoder owners and the explicit service
policy. Model semantics and numerical laws do not acquire GPU scheduling rules.
The six large input regions and output projection select32×32×32; the K128
control projections retain16×64×32. Sparse/expert policies are unchanged.

Eighty-eight affected native tests pass with the existing migration-warning
exclusions unchanged. Source-inspection and leaf-summary runner regressions
also pass. The warning-denied service check passes; targeted info reports1068
existing migration warnings and zero errors. This is not a clean full-suite
claim. The source-pinned ARM build at
`/tmp/lunaflux-glm-service-arm-row-reuse-20261020-v1` succeeds with zero owned
swap and approximately1.51GiB observed peak memory.

The selected standalone K32 path and both full/ragged fused alternatives pass
memcheck, full leak checks, racecheck and synccheck at rows1/3/64, in bounded
2GiB/no-swap owners. Numerical checks additionally cover0/22/36/65 rows and
inactive output bytes. The fused K64 loss remains rejected despite these passes.

The first AOT preparation root
`/tmp/lunaflux-glm-http-row-reuse-20261020-v1` stopped correctly because the
optional previous-artifact argument requires identical source for cubin reuse.
The new source differs; no old cubin was reused and no GPU serving was started.
The new v2 root compiles fresh AOT without that argument. Direct source
inspection splits only uniquely named recurrent helpers, not overloaded
precision helpers; its initial duplicate-name diagnostic is retained as a
runner issue. Rank1 has272 changed bodies (34×six private region helpers plus
34 fused entries and34 output entries), and2300 unchanged entries/preamble.
This includes68 changed GPU entry points, not272 new launches. Inspection
checks unchanged single-row prefixes and ascending K32/K16 arithmetic. Rank0
has the same counts. Both rank AOT builds exit0, peak3.38/3.34GiB respectively,
zero owned swap; no compiler process overlaps resident model loading.

## Fresh both-rank HTTP execution traces

Control: `/tmp/lunaflux-glm-http-row-control-trace-20261020-v1`.
Candidate: `/tmp/lunaflux-glm-http-row-reuse-trace-20261020-v1`.
Both use the same29b3f306 executable,64-row/eight-slot/8192-context plan,
513 pages, window16 collective replay, and all preceding scales/bulk/local-MoE,
projection-TP and bounded sparse-staging selections. Only the recurrent physical
policy/AOT changes. Each runs the same7973-input/one-output request cold and
warm with zero prefix-cache reuse, then returns final answer `323` and retires.
This is a fixed diagnostic pair, not an overall benchmark or quality campaign.

Startup logs select `recurrent_projection_tile_rows=32` and minimum inner4096;
actual CUPTI records prove selection, independently of those logs: fused input
is48 registers/8192 shared bytes, versus39/9216 in control. Output projection
is48/8192 versus40/9216. Launch grids remain3144 and1024,128 threads. The K128
control projection remains40/9216; sparse Q-A remains44/24576. Both ranks have
the intended source changes and all68 selected entries execute per chunk.

| Selected68 projection calls, kernel sum ms | Control | Candidate | Saved |
| --- | ---: | ---: | ---: |
| Cold rank0 |37.152288|33.389664|3.762624|
| Cold rank1 |46.947744|33.514528|13.433216|
| Warm rank0 |37.967232|34.052096|3.915136|
| Warm rank1 |37.130720|34.229952|2.900768|

The warm reductions are10.31%/7.81%. Input-region aggregates improve
27.362080→24.933120ms and26.999936→24.654144ms; output aggregates improve
10.605152→9.118976ms and10.130784→9.575808ms. These are aggregate leaf wins,
not a claim that every invocation is faster or that serving gains that percent.

| First HC through final HC,45 layers, ms | Control | Candidate |
| --- | ---: | ---: |
| Cold rank0 |332.585504|286.601024|
| Cold rank1 |324.198944|279.037312|
| Warm rank0 |290.406080|287.562112|
| Warm rank1 |283.824416|281.122752|

The warm wall reductions are only2.844/2.702ms (0.98%/0.95%). That is consistent
with optimizing roughly37ms of a284–290ms decoder span, not evidence that the
new path failed to execute. Warm rank0 compute union improves4.247ms while
NCCL span increases1.415ms; rank1 compute improves2.541ms and NCCL falls0.109ms.

**Do not attribute the large cold delta to the patch.** Control cold rank1 is
slower in untouched families too: sparse projections27.437→17.815ms, MoE
union199.894→184.444ms and KDA-other12.144→8.911ms. Rank0 waits for that slower
peer: its control collective union66.912ms falls to24.249ms in candidate.
Causal alignment bounds rank0 early-arrival sums at46.402–61.988ms control,
8.486–20.037ms candidate. The capture demonstrates peer progress imbalance,
but does not determine its cache/clock/host cause. Keep this variability visible;
the five alternating micro-pairs and matching warm phases support the narrower
projection benefit, not a45ms causal saving.

All90 collective nodes match on both ranks in all captures; six replay windows
remain. No observed compute/collective overlap occurs inside these chunks.
Candidate warm first collective spans7.120448ms on rank0 and0.184256ms on
rank1; rank0 early arrival is6.797792–7.051968ms. This is not a7ms network
transfer. Peer-clock uncertainty is0.254176ms; no synchronized clocks, wire-time
measurement or same-stream producer edge is invented.

Shared experts remain mostly hidden behind routed work: candidate warm rank1
overlap109.470368ms, exposed0.095072ms. Inter-chunk rank1 handoff remains
9.307776ms, of which8.978560ms is uncovered by recorded device activity;
control9.570880/9.229120ms. Rank0 handoff4.166528/3.832192ms versus control
4.148960/3.810368ms. This patch does not fix the host handoff or TP submission.

## Execution allocation against the retained vLLM trace

These disjoint buckets reconcile to the observed decoder wall span. Shared/
routed overlap is counted once in MoE; overlap between different families is
its own bucket. The native sample is the new warm rank1 capture. The reference
is the retained October19 first64-token decoder chunk, not a new vLLM run.
It has the same checkpoint/TP2/7973-token prompt, but its request output cap is2
versus native1; only this first decoder chunk is compared. Native projection
families include normalization/split/publication leaves; reference matrix
families follow selected GEMM symbols. They are not matched single-GEMM costs.

| Disjoint execution time, ms | Latest native | Retained vLLM |
| --- | ---: | ---: |
| HC |9.813023|3.950080|
| KDA projections |35.076800|26.704800|
| KDA other |8.884736|4.422656|
| Sparse projections |18.110240|9.372864|
| Sparse non-projection |1.846048|2.116095|
| Dense FFN |6.009312|2.281952|
| MoE including shared/routed overlap |184.631744|184.459176|
| TP collective span |16.279936|17.468992|
| Kernel-free trace time |0.466305|5.255609|
| Cross-family overlap |0.004608|0.038464|
| Total decoder span |281.122752|256.070688|

The remaining warm first-chunk difference is25.052064ms (9.78%), not a fresh
request-latency or throughput result. Projections/HC/KDA-other/dense work remain
larger, while the MoE critical-path union is close. More shared-expert kernel
time alone would not identify the critical-path gap. No claim of full parity,
long-history acceleration, or a solved network stall follows from this table.

Raw graph reports:
`/tmp/lunaflux-glm-row-{control,candidate}-{cold,warm}-rank{0,1}-20261020-v1`.
Leaf comparisons: `/tmp/lunaflux-glm-row-effect-{cold,warm}-rank{0,1}-20261020-v1.json`.
Paired causal bounds: `/tmp/lunaflux-glm-row-{control,candidate}-{cold,warm}-tp-20261020-v1.json`.
All source, negative timings, raw NCU and Nsight reports remain retained.

Each run has178 memory-watch rank samples. Maximum observed GPU use is96,533MiB
(94.270508GiB). Fast watcher host minima are18,208,848KiB control and17,898,792KiB
candidate, both above the17GiB early guard and16GiB requirement. Available owned
swap readings remain zero. These are sampled bounds, not proof of every instant.
Diagnostic processes retire before report conversion and production restoration.

## Restoration and completion

Both original `lunanexa-glm-lunaflux-rank{0,1}.service` units are restored with
their previous qualified configuration, not silently switched to this explicit
experimental selection. Both are active with successful unit results and zero
owned swap. `/health` returns healthy; the restored `glm-5.3` backend returns
final content `323`, normal stop, on the arithmetic check. Response evidence is
`/tmp/lunaflux-glm-row-reuse-restored-answer-20261020-v1.json.reply`;
restoration monitoring is
`/tmp/lunaflux-glm-row-reuse-restoration-20261020-v1`. The observer was retired
only after these checks. This is not a new public-route qualification claim.

The task's implementation and selected both-rank retrace are complete. The
remaining performance gaps, small-row caveats and cold-run variability above
are not declared fixed. No overall benchmark, payload hashing, numerical-policy
change, reference substitution, or new runtime vendor dependency was introduced.
Unrelated toolchain/worktree changes are preserved. External source, counter,
execution-graph, handoff and paired-TP analyzers pass their affected tests; no
full-repository clean-suite claim is made.
