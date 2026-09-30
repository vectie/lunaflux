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
