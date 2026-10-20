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

## Actual serving selection and bounded qualification

Corrected source `bbd0d33b` builds with the same pinned ARM compiler in
`/tmp/lunaflux-glm-service-arm-staging-20261020-v2`; the journal reports
1.5 GiB peak CPU build memory and zero swap. Both-rank AOT export/compilation
is `/tmp/lunaflux-glm-http-sparse-staging-20261020-v1`.

Direct comparison must use the **latest projection-TP source**, not the older
policy-combined source (which has 132 more publication entry points). The
external `inspect-glm-staging-source.mbtx` checks the actual exported rank1
source against `/tmp/lunaflux-glm-http-projection-tp-20261020-v1/rank1.cu`:
55 changed kernels, 2,312 byte-identical kernels and an unchanged preamble.
Only Q-A, KV-A, index K, pooling gate and index weights change, in 11 sparse
layers. The compiler contains no reference-engine comparison or timing report.

Actual CUPTI records independently prove the new binary executes: Q-A uses
24,576 shared bytes/44 registers instead of 9,216/40; its 384-CTA/128-thread
launch is unchanged. Q-B stays 9,216/40, and fused KDA stays 9,216/39. The
reserved decode grid is not the count of active prefill matrix tiles.

The first candidate trace root is
`/tmp/lunaflux-glm-http-sparse-staging-trace-20261020-v1`. It selects the same
64-row, eight-slot, 8192-context, 513-page, eight-admission-lane window-16 plan as
the latest diagnostic, with only the explicit sparse staging policy added.
Two literal 7973-input/one-output requests finish with zero cached tokens;
the subsequent answer check returns `323`, normal stop. This is an answer
and retirement smoke, not broad model-quality certification.

All selected narrow shapes 2/3/4/7 pass memcheck/full leak, racecheck and
synccheck at 1/3/64 rows under 2 GiB/no-swap owners. Wider shapes 5/6 also pass
those gates but remain unselected because they lose performance. Tail shapes
retain exact output checks but are excluded by the divisible-window policy.
Independent all-11-shape F32/byte comparisons and sampled double checks remain
separate from the full-service smoke.

Rechecked affected native packages pass 78/78 with the unchanged migration
warning exclusions. Clean ARM build passes. External source/leaf analyzers
pass 3/3; execution-graph and paired-TP analyzers pass 4/4. No full native-suite
pass or new ABI qualification is claimed.

## Fresh control versus candidate: execution graph, not overall benchmark

The subsequent fresh control is
`/tmp/lunaflux-glm-http-staging-control-trace-20261020-v1`. It uses the **same
bbd0d33b executable**, same pinned compiler, same literal prompt, same window-16
plan and memory admission. Only the sparse staging selection/AOT differs.
Both runs capture both ranks cold and warm, with zero prompt reuse. This is
one fixed cold/warm pair per policy, not a statistical throughput campaign.

The 55 actually executed, changed leaves improve on every rank/phase aggregate:

| First 64-token chunk, kernel sum ms | Control | Candidate | Reduction |
| --- | ---: | ---: | ---: |
| Cold rank0 |10.156352|6.640448|34.62%|
| Cold rank1 |9.822463|6.319808|35.66%|
| Warm rank0 |10.107488|6.316576|37.51%|
| Warm rank1 |9.693728|6.936000|28.45%|

This is 2.758–3.791 ms saved in the selected leaves, not 28–38% faster serving.
The smallest index-weight leaf is weak: warm rank 1 is 1.325888 vs 1.287456 ms
(38.432 microseconds worse over 11 calls). Keep this negative result; do not
claim every individual selected kernel improves. The other four leaf families
carry the measured gain. One-token microtimings remain approximately flat;
some small shapes have sub-1% negative samples. No decode speedup is claimed.

| Complete decoder boundary, ms | Control | Candidate |
| --- | ---: | ---: |
| Cold rank0 |299.203520|295.313472|
| Cold rank1 |291.305408|285.593376|
| Warm rank0 |296.294848|295.463936|
| Warm rank1 |290.144096|288.219328|

These are first-HC through final-HC spans for 45 layers, not request latency.
Selected work is only about 10 ms of a 290–299 ms boundary. Changes in unchanged
kernels and TP arrival also matter: warm rank 0 compute union falls
270.370271→265.699424 ms, but NCCL union rises 25.430400→29.283104 ms,
leaving only 0.831 ms less wall time. Do not credit all compute variation to the
new staging, or label the extra collective span wire time.

All 90 collective nodes are matched on both ranks in each capture. No recorded
compute/collective overlap exists in any of these four candidate/control
chunks; shared experts remain almost entirely hidden behind routed work.
Candidate cold rank 0's first collective is 9.925376 ms versus rank 1's
0.190048 ms; causal bounds assign 9.604352–9.852672 ms to early arrival.
Candidate warm first collective is 7.448192/0.181760 ms, with
7.127040–7.374208 ms rank 0 early arrival. Thus this is not a new 7–10 ms wire
stall. Some graph collective nodes execute on internal stream 583 while their
producers execute on 13. The same-stream producer analyzer correctly rejects
that capture; the explicit `--collectives-only` mode retains causal bounds
without inventing a producer edge. Graph correlation remains six windows.

Inter-chunk wall time is still visible. Candidate cold/warm rank 1 is
8.853824/7.996800 ms, with 8.503584/7.660448 ms uncovered by recorded device
activity. Fresh control is 9.746368/10.707808 ms, uncovered 9.406016/10.370432.
The patch does not modify host submission. This cross-run change is **not**
evidence that the host handoff was fixed; CPU/transport attribution remains.

## Remaining reference gap and next work

The retained, matched vLLM first decoder boundary is 256.070688 ms (October 19,
same checkpoint/TP2/7973-input prompt/first 64-token chunk). It is not a new
vLLM run: the previous fresh-load attempt hit its safe early memory guard.
Candidate rank 1 remains 11.53%/12.55% longer cold/warm against that retained
boundary. No full-request or tokens/s comparison follows from these traces.

On cold rank 1, sparse projection disjoint time decreases
20.918623→17.518976 ms; vLLM's retained matrix category is 9.372864 ms.
These categories follow the existing graph analyzer: the native projection
family also includes its normalization/split/publication leaves, whereas the
reference matrix category is identified by selected GEMM symbols. They are
not a one-to-one matched GEMM microbenchmark; use the full attention chain
and the recorded leaves when attributing reference differences.
KDA projection source remains unchanged and takes 37.695488 ms against the
retained 26.704800 ms. These are the next large projection frontiers, not
resolved by a bigger synchronous K tile. The rejected square-matrix K128
results show why another blanket tile-size change is unjustified. Next work
must address the large-matrix load/reuse/residency schedule, then HC and the
host completion/submission boundary separately. Do not multiply this first
chunk's saving by 125 to predict long-history request time.

Raw decoder reports are
`/tmp/lunaflux-glm-staging-{candidate,control}-{cold,warm}-rank{0,1}-20261020-v1`.
Both raw roots retain `paired-{cold,warm}-90.json`, handoff reports,
requests/responses and SQLite. The candidate root also retains all four
`versus-control-*.json` leaf-level comparisons; `summarize-glm-staging-effect.mbtx`
rejects unmatched call counts and requires all 55 selected calls to execute.

Memory-watch observations cover 178 rank samples per run. Worst sampled
host availability is 18,258,668 KiB candidate /18,316,940 KiB control (both
above 16 GiB), maximum GPU use 96,533 MiB (94.270508 GiB), observed owned-process
swap zero. The faster 250 ms admission watcher also remained above its 17 GiB
early-stop threshold. Samples are not proof of every unobserved instant.
Both diagnostic owners retired and both arithmetic/retirement smokes passed.
Production restoration is a separate check; no experimental path is silently
promoted by these results.

## Restoration and handoff

The previous qualified production configuration is restored unchanged on both
Sparks. `/health` is healthy; the backend model `glm-5.3` returns `323`, normal
stop (26 input/four output tokens). The first restoration client request used
the public-route model alias `glm-5.3.flash` against the private backend and
was rejected with HTTP400; the corrected request is preserved separately at
`/tmp/lunaflux-glm-staging-restored-answer-20261020-v2.json.reply`. This is not
an engine correctness failure or a new public-route qualification claim.

Restoration evidence: `/tmp/lunaflux-glm-projection-restoration-20261020-v1`.
Observed GPU maxima are 94.190430/94.218750 GiB; host minima
17.782738/19.460430 GiB. Available owned-unit swap readings are zero; five
startup peak readings were unavailable and are recorded as such. The watcher
retired after health/answer verification. No GPU experiment remains active.

The physical staging implementation and source tests are committed, but the
new policy remains explicit/nondefault. This phase qualifies a narrow prefill
improvement, not all projection shapes, broad concurrency or decode gains.
The AKO-style gate kept the large-matrix losses out of the selected path and
required both actual serving selection and a fresh control before reporting
the improvement. Comparisons remain external to the functional compiler.
