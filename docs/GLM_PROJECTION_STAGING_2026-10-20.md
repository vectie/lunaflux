# GLM projection staging: bounded follow-up

Scope: fix one measured execution-graph gap at a time. No overall benchmark,
payload hashing, numerical-policy change, or comparison in the compiler.
Both Sparks retain the 100 GiB ceiling and 16 GiB host reserve. Restore the
owned GLM service after isolated GPU qualification.

The latest complete first-64-token decoder trace (6c10bf37 runtime) attributes
23.455 ms of a 34.020 ms rank-1 excess to KDA/sparse projections. The selected
physical body is 16x64x32 synchronous WMMA, with two CTA barriers per K32.
Buffered staging already has losing full-model results; it is not enabled by
this experiment. The retained vLLM trace is differential evidence, not a
numerical oracle or a guarantee that its schedule wins every shape.

First campaign: capture counters on the actual shared dense lowering at
64 rows, then test at most two synchronous staging-window alternatives if
the counters support that hypothesis. Keep the ascending K16 matrix fold,
16x64 result ownership, decode route, grid and publication unchanged. Test
4096x4096, 4096x8192, 1536x4096 and 128x4096, including live-row boundaries.
Five alternating paired CUDA-graph trials include the complete projection.
No component timing is presented as a whole-service gain.

The offline fixture lives in `tests/packed_moe_execution/projection_staging_probe.cu`.
It checks full F32 output/inactive-byte equality and sampled independent double
dots, with bounded memory and deterministic CUDA cleanup. Source export,
sanitizer and actual serving-selection evidence remain required before promotion.

## First isolated campaign

Baseline root: `/tmp/lunaflux-glm-projection-staging-baseline-20261020-v1`.
Candidate roots: `/tmp/lunaflux-glm-projection-staging-candidates-20261020-v2`
and `/tmp/lunaflux-glm-projection-staging-coverage-20261020-v1`. The earlier
candidate-v1 controller failed before GPU launch because the remote MoonBit
path did not exist; preserved logs are not timing evidence. Remote execution
now uses the bounded native probe; local AKO summarizes retained five-pair
stdout. The helper no longer computes payload digests.

At64 rows, median unprofiled K32→K128 microseconds:

| N×K | Baseline | K128 | Decision |
| --- | ---: | ---: | --- |
|4096×4096|233.584|332.720|Reject|
|4096×8192|466.475|685.506|Reject|
|1536×4096|103.728|79.970|Candidate|
|128×4096|93.410|65.659|Candidate|
|512×4096|92.867|65.554|Candidate|
|16384×512|68.907|98.139|Reject|
|4096×1536|61.248|79.894|Reject|
|32×4096|77.560|68.216|Candidate|
|8192×1536|141.216|191.997|Reject|

K64 is not a robust universal alternative either. All raw five-pair results,
decode rows and tail shapes remain in the directories. Whole F32 outputs and
inactive bytes match for all eleven shapes at0/1/3/22/36/64/65 live rows;
sampled double-dot checks pass. Neither microtiming nor equality establishes
whole-model output quality or a serving speedup.

For4096×4096, K128 reduces warp instructions12.402M→7.231M and barrier
stall share20.90%→12.75%, but shared bytes rise9,216→24,576, active-warps
occupancy falls44.72%→27.52%, long-scoreboard share rises63.17%→72.12%,
and L2 read sectors rise4.121M→5.149M. Fewer instructions/barriers alone are
not the objective. For1536×4096, instructions4.652M→2.713M and barriers
14.79%→6.72%; measured profile time303.36→220.32µs, distinct from the
unprofiled warm graph medians above. Counter captures do not fix cache/clocks.

The explicit physical candidate therefore bounds **both** weight workset and
active matrix tile count, and requires a divisible K window (no extra padded
MMA). Its supplied envelope is16MiB weights and192 active tiles: at most
48 SMs × four K128-resident workgroups, not a universal hardware constant.
The generic IR receives these bounds as data and resolves a concrete staging
plan before lowering. The service opts in only for standalone sparse
projections; fused recurrent segments remain unchanged because their shared
scratch high-water mark would penalize the large regions too.

Default service policy is unchanged. Source regressions verify that Q-A/KV-A
select K128 while Q-B/KV-B/output and the fused KDA body remain K32; global
workspace bytes and numerical laws are unchanged. Native affected tests78/78
pass with the existing migration-warning exclusions. The unfiltered warning-
denied check still rejects pre-existing deprecated APIs; no new suppression
was added. Targeted `moon info` reports121 existing warnings, zero errors.

The first committed ARM build (`6dd2dee2`, staging-v1) stopped before GPU
execution: the newer local formatter introduced `Int32`, unavailable in the
pinned ARM toolchain. Keep the existing cross-toolchain `Int` spelling; do not
change the experiment's compiler toolchain. Its failed journal is preserved.
