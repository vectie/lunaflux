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
driver reserve; old39/0/0/10,240. Counters on the whole fused body are pending.

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
| Barrier stall % |22.255|22.230|12.129|
| Long-scoreboard stall % |66.785|70.049|64.529|
| Registers/thread |39|48|40|
| Shared bytes including driver |10240|9216|13312|

The K64 fused loss coincides with much lower residency despite fewer reads and
lower stall percentages. The winning K32 schedule retains residency, reduces
support instructions and traffic, and finishes earlier even though its load-wait
percentage rises. No spilling was recorded. This is why neither a stall
percentage nor standalone projection timing is a sufficient selection criterion.
The serving candidate is32×32×32 for recurrent projections with K≥4096,
not K64, and not a change to sparse/expert schedules. Default selection stays
unchanged pending actual serving qualification.
