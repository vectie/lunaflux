# Priority repairs: compiler decisions reaching serving

The first integrated, strict-arithmetic retest improves every median in the
nine-cell serving vector. This is a positive result, not parity with the
references. It supersedes the throughput portion of the
[remaining-gap investigation](BENCHMARK_FINAL_GAP_INVESTIGATION_2026-09-30.md).

## Matched end-to-end results

Qwen3-0.6B BF16 on one GB10 Spark; input/output lengths and concurrency are
independent benchmark axes. Each cell has one warmup and three measured
trials. Rates are output tokens/second. Reference results are the previously
completed matched September 30 campaign, not newly rerun references.

| Input/output | C | Prior LunaFlux | Strict repaired | vLLM | SGLang |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128/32 | 1 | 131.1 | 132.2 | 107.0 | 110.7 |
| 128/32 | 8 | 752.9 | 792.6 | 898.2 | 892.0 |
| 128/32 | 16 | 1312.8 | 1343.8 | 1590.1 | 1546.8 |
| 4096/64 | 1 | 84.1 | 86.4 | 82.6 | 81.8 |
| 4096/64 | 8 | 175.6 | 180.7 | 225.1 | 219.1 |
| 4096/64 | 16 | 200.3 | 204.6 | 243.2 | 241.7 |
| 4096/256 | 1 | 98.7 | 102.3 | 93.7 | 92.8 |
| 4096/256 | 8 | 251.0 | 258.8 | 307.3 | 299.2 |
| 4096/256 | 16 | 298.8 | 305.4 | 345.5 | 340.9 |

Strict repaired C16 long-generation completion is 13,411 ms versus prior
13,708 ms and the references' 11,855 / 12,015 ms: approximately 13.1% / 11.6%
longer. Improvements across cells are roughly 0.9–5.3%; a few short-cell trial
ranges overlap, so this is not a claim of statistically significant gains in
every short workload. All 225 measured output sequences match prior
LunaFlux exactly. All 150 long-input sequences match both references. The
pre-existing short-case disagreements remain (222/225 overall vLLM,
216/225 SGLang); all first tokens match both references.

## What changed and what actually propagated

1. A pure value-component ownership map selects contiguous packed fragments
   where legal. CUDA lowering consumes it in ordinary and partitioned decode;
   fragment widths unsupported by the renderer retain the masked interleaved
   map. BF16 conversion and ordered strict key arithmetic remain unchanged.
   Decode copy retention selects 16-byte vectors for compatible dimensions,
   with a legal smaller-vector fallback.
2. Projection fold records V3 carry consumer cooperation and a bounded row
   domain. The narrowest applicable domain wins independent of record order.
   The selected small-row output and sibling gate/up kernels use 64-thread
   cooperative groups through rows16; wider shapes retain their measured
   global schedule. A source-byte comparison verifies that the exported
   selected sources are the measured ones, after removing probe-only comments.
3. The exporter/build/materializer/runtime chain supports an explicit
   `blockwise-fma-f32-probability-v3` alternative. It emits scoped `fmaf`, not
   global fast-math. Bundle schema V8 and its decode ABI V9 propagate through
   assembly, launch materialization, bootstrap classification and executable
   device admission. Strict bundles remain on their existing schema/law.
4. Ordinary and 2/4/8-way split alternatives are measured including merge.
   The serving bundle contains the resulting shape/bucket route table, rather
   than promoting one aggregate winner across all workloads.
5. A request-epoch graph correlation separates gaps before, during and after
   the next graph submission. This is diagnosis; no unmeasured host-runtime
   rewrite is claimed as a performance improvement.

Target propagation also needed repair: attention recipe rendering used a
literal sm120, and fused bundle export separately defaulted to sm120. Recipes
now render the compiler plan's target; the candidate exporter accepts explicit
device capability; the fused builder passes the compiled capability and
rejects mixed-target module sets. Both fused and unfused bundle export preserve
that capability. Legacy direct CLI invocations retain their sm120 default for
compatibility, while the recipe builder supplies an explicit target. Target
and numeric-law checks are startup/offline work, not token-step cryptography
or filesystem checks.

## Exact-kernel improvements and limits

The corrected-target explicit FMA bundle also completed the nine-cell retest
(`serving-fma5`). Its rates, in the table's order, are 135.6, 815.3, 1372.7,
87.7, 179.2, 201.0, 104.5, 260.7 and 305.6 tok/s. It is faster than the prior
runtime throughout, but **not uniformly faster than the strict repair**:
4096/64 C16 regresses from 5006 to 5095 ms; 4096/256 C16 is effectively tied
(13,411 versus 13,404 ms). These campaigns also differ in ingress cut (strict
uses full fusion; this FMA experiment uses producer-separated ingress),
corrected recipe target and measured split selection, so the difference is not an isolated
FMA causal estimate. All 150 long-input sequences remain exactly equal to
strict/prior LunaFlux, but only 220/225 total sequences do; all first tokens
match. The explicit alternative is implemented and executable, not a
semantics-preserving default substitution. Strict arithmetic remains the
default. The FMA campaign passes deterministic compilation, sanitizer checks,
startup, bounded-memory serving and successful child drain with empty stderr.

Five unprofiled paired replay trials at 16 real rows, using actual dynamic
shared allocations, measured:

| Chain | Prior median | Selected median | Interpretation |
| --- | ---: | ---: | --- |
| Output projection | 12.970 us | 12.323 us | 5.0% shorter; grid16/block256 becomes grid32/block64 |
| Whole MLP | 58.932 us | 47.660 us | 19.1% shorter; gate/up grid192/block512 becomes grid96/block64 |
| Gate/up alone | 18.466 us | 12.330 us | Approximately 33% shorter; down schedule unchanged |

Synthetic chain comparisons are bitwise equal with zero maximum error on
these operands. Memcheck (including leak checking), racecheck and synccheck
passed. Replays which accidentally omitted dynamic shared allocation are
retained but excluded. Several output/down alternatives were slower; down
has no demonstrated new winner and is not silently replaced.

A paired NCU replay of exact packaged strict split decode, 16 rows and 4095
history, finds warp instructions **74.05M → 61.29M** (−17.2%), registers
**118 → 80**, shared wavefront sectors **12.19M → 10.09M**, no local spills and
zero source-correlated excessive shared wavefronts in both variants. Its
cold-cache profiled invocation changes **1.270 → 1.238 ms** (−2.5%). These
are synthetic selected-cubin counters, not a live-serving counter capture or
cross-framework throughput result. The replay uses a 16-row launch bucket;
the fresh serving trace retains a 32-row attention grid with inactive slots.
Thus its instruction/timing deltas are not a capture of the exact live graph
launch geometry. The trace independently verifies serving improvement.
Dependency/issue ratios remain high and
some increase as the issue denominator changes; they are not latency
percentages. Reduced instructions did not eliminate the critical waits.

A second same-geometry eight-partition replay (`counters-fma1`) compares the
strict repair with FMA: **61.29M → 52.81M** warp instructions (−13.8%), but
**1.250 → 1.241 ms** (−0.7%). Both use 80 registers, zero local spilling,
zero excessive shared wavefronts and 10.09M shared sectors. Eligible warps
remain low (0.107 → 0.093); dependency/issue ratios remain high. Thus arithmetic
compression does not remove the shared-load/reduction dependency chain or
increase ready-to-issue work. This capture deliberately holds eight-way
geometry fixed; it is not the FMA serving route selector's four-way invocation
and cannot assign the entire end-to-end difference to FMA. It supports keeping
the non-uniform alternative optional rather than treating fewer instructions
as sufficient proof of faster serving.

The prior exact 4096/256 C16 trace contains 288 graph submissions. Correlated
kernel gaps total 432.13 ms between graphs: 378.38 ms before the next submit,
53.69 ms inside its API span, 0.062 ms after it; no unmatched gap. Intra-graph
gaps are 4.67 ms. This is not proof of an idle GPU (copies are not included),
nor permission to sum overlapping event waits into a CPU bubble budget.

The fresh strict trace (`trace-strict1`) verifies propagation in actual serving:
rows16 output launches grid32/block64, and rows16 gate/up launches
grid96/block64. Its matched measured request has **65,456 / 65,456 graph
kernel calls**; split decode remains selected and down retains its prior
geometry. Summed family milliseconds are:

| Family | Prior trace | Strict repaired trace |
| --- | ---: | ---: |
| Decode attention plus merge | 9171.149 | 8948.214 |
| Gate/up plus activation | 1010.799 | 951.621 |
| Output plus down | 947.755 | 943.010 |
| Prefill attention | 660.618 | 665.020 |
| QKV ingress | 870.310 | 874.699 |
| All kernels | 13229.522 | 12951.360 |

The large synthetic MLP gain is diluted by unchanged larger-row/down work;
output/down's aggregate change is small. Decode supplies most of the actual
trace reduction. This explains the modest serving gain without claiming the
compiler patch failed to execute. The new correlated inter-graph gap is
385.51 ms: 332.49 ms before submit, 52.88 ms within it, 0.135 ms after it,
with zero unmatched gaps and 4.53 ms intra-graph gaps. No host-runtime change
was made, so the difference from the prior trace is not attributed to a new
scheduler optimization. These are profiled diagnostic windows, not another
throughput measurement.

## Reproduction and validation

Remote campaign root:
`/home/wlc004s/lunaflux-positive-repair-20260930.nHyfgw10`.
`projection-round5`, `decode-round2`, `decode-round5`, `integrated-round3`,
`serving-round1`, and `counters-strict2` retain exact sources, recipes, cubins,
commands, sanitizer logs, client sequences and resource reports. Failed
campaigns remain separate and are never relabelled as successes. In particular,
the first FMA serving attempts exposed schema-reader and target propagation
defects; an isolated stderr-enabled worker diagnosed `Invalid(DeviceTarget)`.
That diagnostic launcher/worker is not a production or timed-benchmark binary.

GPU work is serialized. Serving has a 64 GiB cgroup ceiling and no additional
swap; counter replays use 8 GiB. Campaign admission requires at least 32 GiB
available memory. This validates a local model/runtime benchmark, not fleet
deployment, other GPUs/models, all numerical inputs or production promotion.

The compiler remains layered: pure semantic laws → immutable ownership and
schedule selection → explicit storage/effect plans → CUDA lowering. No
model-family or scheduler CUDA special case was added. Performance choices
are portable descriptions; selected records remain device/workload scoped.

Local native validation: **4263/4263 tests passed**. `moon info` completed;
warning-denied check/test use `--warn-list -79-20-29-25-92-14` for existing
toolchain-migration and unrelated TLS/multimodal warnings. This is not a claim
that the whole dirty repository passes the default unsuppressed warning policy.
Affected exporters also pass focused warning-denied checks/tests. Formatting
and standalone checks cover the changed MoonBit campaign helpers; Git whitespace
checks pass. Unrelated tensor-parallel/remote/multimodal edits are preserved.

The final diagnostic archive includes exact strict/FMA serving source snapshots,
successful and failed campaign logs, selected AOT modules, raw client outputs,
NCU reports and the fresh Nsight trace. Model-root copies and build caches are
listed as exclusions, not deleted. Remote archive:
`archive-final3/gap-repairs.tar.gz`, SHA-256
`b676e3fe96e06bf565aa5fa134ca8219e0ea080619a25424cd889f16e3577605`.
Exact strict source snapshot SHA-256:
`c5d5be2d7a8f277c77433dabfed92f073a87506254aa9e5e931cf5972807fafc`;
FMA source snapshot:
`dba616651995012940eded59cb93f9d52eb53579b5e5384759e5d9fa4aa7ab85`.

## October 1 decode-pipeline follow-up

Decode is the largest remaining measured family, not the only remaining gap.
Output/down is still separate. Several pipeline alternatives were tested before
changing the selected production source; none reliably improved the C16/H4095
replay. Their rejected production edits were removed, not left as inactive
compiler options:

| Experiment | C16/H4095 median change | Outcome |
| --- | ---: | --- |
| Larger ready fragment window | approximately −0.3% | Not a reliable gain |
| Three-slot dead-K reuse | approximately +0.7% | Reject; instructions +17.3% |
| Register-owned probabilities with three slots | approximately +1.0% | Reject |
| Streaming rather than cache-all async copies | approximately +0.6% | Reject |
| Prefetch before QK, with separate V readiness | approximately +1.1% | Reject |
| Smaller KV tiles | roughly flat at C16 | Reject; C1 long history substantially regresses |

These are paired synthetic chain replays, not new framework comparisons. The
three-slot counter experiment improves residency from two to three blocks and
reduces short-scoreboard latency, but increases instructions, long-scoreboard
latency and barrier latency. More occupancy or overlap is not itself a speedup.

A concrete lowering defect was found independently of those experiments:
`CompleteTileReaders` is a terminal effect, but the blockwise CUDA renderer
emitted it after every KV tile. The fix emits `ReleaseTileReaders` inside the
loop and `CompleteTileReaders` once after the loop, matching the immutable
effect plan. The pipelined loop now has two workgroup publications instead of
three; synchronous single-slot behavior is unchanged. The lowering identity
records the corrected terminal placement. No numerical law, tile geometry,
copy width, stage allocation or scheduler behavior changes.

Regression coverage checks all eight synchronous/asynchronous strict,
dual-score and contracting blockwise candidates. Local native check and
**4264/4264 tests pass**, with the same documented warning exclusions. The
same exported strict c452 kernels pass bitwise comparison and memcheck,
racecheck and synccheck on the Spark. Source and sanitizer correctness do not
prove a speedup.

Campaign `/home/wlc004s/lunaflux-decode-handoff-20261001.zzsKyyQ7` uses the
traced 32-row serving launch envelope, including inactive rows. Both old and
new p8 use the same geometry and five alternating trials per cell. Representative
medians in microseconds:

| Active rows | History | Prior p8 | Corrected p8 | Change |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4095 | 92.124 | 92.189 | +0.07% |
| 8 | 4095 | 628.226 | 640.269 | +1.92% |
| 16 | 4095 | 1189.644 | 1194.671 | +0.42% |
| 16 | 8191 | 2353.508 | 2351.440 | −0.09% |

Thus the bug is fixed, but isolated timing does not show a reliable performance
gain. The p4 trial also changes the partition count and is not a same-geometry
attribution to the handoff fix. A new exact-overlay serving rebuild/retest is
required before revising the end-to-end table above. Its first rebuild hit its
8 GiB cgroup ceiling during parallel compilation; the retry limits compilation
to two jobs. The host retained its memory reserve; no failed run is promoted.

The completed serving overlay is
`/home/wlc004s/lunaflux-decode-serving-20261001.T34eKMDF`. It rebuilds the
required native executables from the archived strict source plus the handoff
fix, re-exports the selected kernels, rebinds the release and regenerates the
decode route package. It is not a build of the unrelated dirty working tree.
The standalone-selection guard verifies the final publication in the exported
source; the packaged direct kernels and split chain pass sanitizer checks.
All **225/225 output sequences and first tokens match** the prior strict run.
The runtime drains successfully, reports `child_closed=1`, and has empty stderr.

| Input/output | C | Prior tok/s | Corrected tok/s |
| --- | ---: | ---: | ---: |
| 128/32 | 1 | 132.23 | 131.69 |
| 128/32 | 8 | 792.57 | 792.57 |
| 128/32 | 16 | 1343.83 | 1340.31 |
| 4096/64 | 1 | 86.37 | 87.43 |
| 4096/64 | 8 | 180.66 | 181.17 |
| 4096/64 | 16 | 204.55 | 204.72 |
| 4096/256 | 1 | 102.28 | 101.91 |
| 4096/256 | 8 | 258.85 | 260.03 |
| 4096/256 | 16 | 305.42 | 306.08 |

C16 long completion time is **13,411 → 13,382 ms** (−0.22%); the trial ranges
overlap. This does **not** establish a meaningful speedup. Against the unchanged
September 30 reference measurements, completion time remains about **12.9%
above vLLM and 11.4% above SGLang**. Those frameworks were not rerun for this
follow-up. The earlier throughput table is not silently replaced with a new
cross-framework campaign.

Paired C16/H4095/p8 cold-cache counters in
`/home/wlc004s/lunaflux-decode-counters-20261001.vXe6FQnj` confirm the change
executes, rather than merely existing in an unused compiler plan:

| Counter | Prior | Corrected |
| --- | ---: | ---: |
| Warp instructions | 61,353,984 | 61,323,264 |
| Barrier stall / issued-instruction ratio | 0.679740 | 0.544329 |
| Short-scoreboard stall / issued-instruction ratio | 3.189743 | 3.198981 |
| Long-scoreboard stall / issued-instruction ratio | 2.690638 | 2.928409 |
| Eligible warps / active cycle | 0.121867 | 0.123502 |
| Source-correlated excessive shared wavefronts | 0 | 0 |
| Profiled partial-kernel time | 1.240640 ms | 1.250656 ms |

The instruction reduction is exactly **30,720**, only **0.05%** of the kernel.
There are 1024 active CTAs, two warps each, and 16 KV tiles per partition;
replacing 16 terminal publications with one removes
`1024 × 2 × (16 − 1) = 30,720` warp barrier instructions. Barrier waits fall,
but the load/reduction dependency chain and almost all supporting arithmetic
remain. The stall ratios are normalized counters, not percentages of wall
time. This is a synthetic cold-cache partial replay, not a new live-serving
counter trace. It explains why the correct lowering repair is not a cure for
the remaining throughput gap; no bandwidth-roofline or framework parity claim
is justified by these measurements.

The successful scoped rebuild peaks at **4.8 GiB** with zero cgroup swap. The
initial full-module build retry also encountered an unrelated standalone
link target without `main`; the reproducible helper now builds only the eight
executables needed by this campaign. Failed attempts are retained separately.

The final handoff replay, exact serving overlay and paired counters are archived
at `lunaflux-decode-archive-20261001.6B1pDSaH/decode-handoff.tar.gz`, SHA-256
`e5fd794562f5f276a727c14d70db7ab26cbb96e94f0c713c33f336af598ef146`.
Build caches, toolchain installations and duplicate model-root copies are
explicitly excluded, not deleted. Earlier rejected experiments remain in their
separate remote campaign directories. The source fix is committed as `e57fe109`
on `parallel`; it does not assert that the remaining performance gap is fixed.
