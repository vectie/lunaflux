# Sparse projection and HC ownership

Scope: latest row-reuse/scales/bulk/local-MoE/exact-window candidate,
fixed first 64-token decoder chunk, both ranks, cold and warm. No overall
benchmark or payload hashing. Comparison stays outside the compiler.
Stop only the owned GLM service; restore its prior configuration afterward.
Limits: 100 GiB/device, 16 GiB host reserve, no owned-process swap.

The preceding warm rank-1 trace attributes 18.110240 ms to sparse projection
and 9.813023 ms to HC. Within those families, sparse output alone is
6.097984 ms and the 89 deferred post/pre calls sum 8.278496 ms.
These are selected kernel durations, not an additive whole-request estimate.

Finite campaign: at most two alternatives per target before reconsidering
the hypothesis. Preserve negative trials. Validate generated source, actual
launch ownership, numerical frontiers and sanitizers before both-rank replay.

Initial hypotheses to test:

1. Sparse output/query-B still use column-heavy 16x64 ownership despite
   multiple independent token rows. Reuse rows only where it does not undo
   the already-qualified wider-K staging selection. Keep the same ascending
   matrix accumulation and BF16 publication.
2. Deferred HC still publishes every level of the same reduction tree
   through workgroup shared memory. A subgroup-local suffix may preserve
   exactly the same addition tree while reducing barriers and shared traffic.
   Test the complete row-dispatched entry, not merely a standalone bucket.

The installed vLLM source and retained selected trace are performance
references, not correctness oracles. Its HC uses a matrix prenorm path;
changing our scalar arithmetic into that path is not assumed equivalent.
The local `../vlm` tree may be newer than the installed reference.

## Isolated evidence

HC source references: installed model snapshot
`/tmp/lunaflux-glm-selected-vllm-source-20261019-v3`, model.py
`hc_fused_post_pre`; local `../vlm/vllm/model_executor/kernels/mhc/tilelang.py`
433–634 distinguishes the small FMA route from larger-row post + matrix
prenorm. `tilelang_kernels.py` 525–640 uses subgroup reductions in the
small route. Our retained 64-row vLLM trace actually selects
`sm120_tf32_hc_prenorm_gemm_impl`; do not mislabel the small FMA source as
the executed prefill kernel. Neither its reassociation nor TF32 arithmetic
is silently substituted for our declared normalized scalar folds.

HC root `/tmp/lunaflux-glm-hc-vocab-reduction-local-20261020-v1` preserves
the first subgroup-only trial. At 64 rows it saves only a few percent; the
22/36-row trials are noisy. Counters still show dominant global-load waits.
Second root `/tmp/lunaflux-glm-hc-vocab-reduction-wide-20261020-v2` retains
the full dispatched baseline (widths1/2/4) and subgroup+reuse alternative
(widths1/2/8). All seven 64-row pairs improve: baseline56.20–56.77 us,
candidate36.64–38.98 us. Both preserve every tested publication, inverse
and control projection bit, including inactive bytes.
The failed v1 preparation is retained: SSH control masters expired; no
candidate GPU execution occurred there. Both masters were re-established.

Selected 64-row counters, same complete dispatch (not isolated width arms):

| Metric | Shared-tree / width4 | Subgroup suffix / width8 |
| --- | ---: | ---: |
| Warp instructions | 9,042,304 | 6,096,832 |
| L2 read sectors | 1,591,894 | 1,477,788 |
| Registers/thread | 96 | 104 |
| Register-limited resident blocks | 2 | 2 |
| Active warps | 32.08% | 32.75% |
| Long-scoreboard share | 69.87% | 68.73% |
| Barrier share | 4.94% | 5.02% |

Fewer instructions and repeated producers matter, not a lower stall
percentage. NCU replay durations are not substituted for warm graph trials.

Sparse output root
`/tmp/lunaflux-glm-projection-staging-sparse-reuse-20261020-v1` tests the
actual dense lowering, 64 rows, K8192/N4096, reserved1024-CTA grid.
Its fixture publishes F32 partial outputs; serving publishes BF16 after
rounding. Thus these counters establish the dense-body hypothesis, not
literal BF16-serving-kernel counters. A subsequent source-extracted probe
and complete serving traces must establish that distinction explicitly.
M32/N32/K32 is inconclusive (one pair loses); M32/N32/K64 wins all five
pairs, median paired gain15.408%, minimum3.233%. Median time483.443→417.394 us.
NCU instructions24,591,360→16,872,448; L2 sectors8,212,913→6,527,076;
active warps44.41→44.15%; tensor activity4.99→7.20%; registers40→40.
The prior KDA fused-K64 loss is not contradicted: it has a different
resource envelope. This alternative is selected only by an explicit
independent sparse projection ownership envelope.

The pure `MatrixRowReuse` pass preserves previously selected K128 or buffered
staging. The supplied test envelope uses minimum capacity32, minimum K1536,
and K64 only from K8192; other qualifying reductions use K32. It reaches
sparse output, Q-B and index-Q through actual source and launch construction.
Q-A, KV-A, KV-B, index-K/weights and pooling, recurrent kernels, numerical
laws and device allocations remain unchanged. These bounds are explicit
offline configuration, not model-name branching or runtime tuning.

Physical `ReductionTreePartition` preserves every descending addition level,
moving only the suffix into one subgroup. The eight-control candidate changes
producer reuse, not control ordering. Comparison remains external.

For the observed four-stream HC, there are24 controls. Four controls/group
reconstruct the published residual and its RMS six times per row; eight
controls/group reconstruct them three times. The128-CTA launch budget is
unchanged: logical work iterations fall384→192 at64 rows, not a claim that
the launched grid halves. Control weights and each control's arithmetic
remain unchanged. Sparse M32 ownership similarly halves weight staging
across token tiles, while the narrower N32 tile repeats input staging more
often. This trade is favorable for the measured weight-heavy shape;
it does not imply half the DRAM bytes or universally faster GEMM.

Both-rank traced results and final source-exact counters follow below.

The width8 trial loses at some36-row pairs. Final catalog selection restricts
width8 to live64, retaining widths1/2/4 elsewhere. Final complete-wrapper
root `/tmp/lunaflux-glm-hc-vocab-reduction-bucketed-20261020-v1` retains a
small36-row regression even with four-control ownership (typically
27.8→28.6 us): adding a bucket changes the shared compiled resource envelope.
This is visible, not claimed solved by the branch alone. All seven64-row
pairs still improve (53.84–58.30→36.48–40.01 us). No production promotion
is inferred from this long-prefill qualification.

The intermediate whole-AOT v1 compilation was explicitly stopped after
refining this row domain; no GPU model run used it. Its source/logs remain.

Final four-bucket dispatch counters are retained separately in that final
root: instructions9,042,304→6,137,984; L2 sectors1,699,672→1,489,006;
registers96→108; shared memory8,192→16,384 bytes including driver reserve.
Register residency remains two blocks; active warps29.46→32.90%. Earlier
three-bucket counters above are not relabelled as this final binary.

Why K64 helps sparse output but previously hurt fused KDA: the output fixture
keeps measured active warps about44% even when the shared-memory block limit
falls10→7. The preceding fused KDA fixture used about72% active warps before
that resource increase and fell to about52%. Thus the larger staging tile
reduces loop/transfer instructions here without the measured residency loss
seen in the fused body. This is a shape/resource-envelope explanation from
two different fixtures, not a claim that one stall percentage proves speed.

Validation so far: 111 affected native tests, warning-denied service check
with unchanged migration exclusions; targeted info1074 existing warnings,
zero errors. Final HC and selected sparse K32/K64 shapes pass memcheck with
full leak checks, racecheck and synccheck under2GiB/no-swap limits. The full
strict repository check/test were attempted and stop on pre-existing
toolchain deprecations (e.g. parallel_ir/collective_windows and
internal/monotonic_clock); no full-suite pass is claimed. Owned sources only
were formatted; unrelated migration files remain untouched.

## Exact reference dispatch and campaign provenance

The retained installed vLLM rank-1 first64 trace selects
`nvjet_sm121_tst_mma_128x64x64_4_32x64x64_tmaAB_bz_TNNN` for sparse output,
not the CUTLASS KDA projection kernel discussed in earlier work. Its11
calls sum4.099424ms; CUPTI records grid256, block128,255registers/thread
and99,328bytes dynamic shared memory. The executed HC prenorm is
`sm120_tf32_hc_prenorm_gemm_impl`:90calls/1.800544ms, grid48,
block384,168registers/thread,98,368bytes dynamic shared memory.
HC post and pre-big-fuse add1.623840ms and0.550560ms. These are trace
durations/resource metadata, not new hardware-counter captures.

Heavy per-block resources do not by themselves explain slow execution:
the reference uses TMA/matrix reuse at lower possible occupancy. Native
output still uses synchronous global-load/shared-store staging. Native HC
retains normalized F32 scalar folds, unlike the reference TF32 prenorm.
The two remaining implementation differences must not be confused with
the measured ownership improvements.

Reference database:
`/tmp/lunaflux-reference-bench-glm-long-egress-trace-20261019-v2/rank1.sqlite`.
It is the retained October19 installed reference (revision487ecf187),
not a fresh two-rank vLLM run. Latest LunaFlux traces cover both ranks.
No fresh unsafe full-reference load is justified by this component test.

The final source-pinned7cffe534 artifact is
`/tmp/lunaflux-glm-http-sparse-hc-20261020-v3`.
Its predecessor v2 hit the **4GiB CPU compiler cgroup limit**;
both failures are preserved. The v3 retry uses8GiB CPU compilation,
checks that this cap plus16GiB host reserve is available, and disables
owned-process swap. It completed on both ranks. This did not exhaust the
100GiB device budget or launch GPU work. The source audit on each rank
records393 changed entries,90 added HC bucket helpers and2524 unchanged
entries; only the specified sparse leaves and HC dispatch/helpers differ.

The first control trace root v3 failed before model loading because the
local18889 SSH forward had not survived the control-master replacement.
Both owned ranks were stopped. The corrected control uses a new v4 root,
preserving that failure. No model or kernel correctness claim is drawn
from the failed startup.

## Literal BF16 output source and counters

`/tmp/lunaflux-glm-sparse-selected-output-20261020-v1` extracts
`glm_sparse_3_output` literally from both rank1 AOT source files, changing
only the entry names. Launch1024x128 matches retained serving metadata;
M64/N4096/K8192 and BF16 publication are unchanged. The complete output
arrays agree bitwise. Five rotating, graph-replayed pairs all improve;
median old488.390us, new411.373us. This is a synthetic component test,
not a full-model answer-quality or overall throughput result.

| Source-exact counter | Before | After |
| --- | ---: | ---: |
| Warp instructions | 24,625,152 | 16,906,240 |
| L2 read sectors | 8,165,450 | 6,407,606 |
| Registers/thread | 40 | 40 |
| Shared bytes/block including driver reserve | 10,240 | 13,312 |
| Active warps | 44.59% | 44.24% |
| Tensor activity | 5.33% | 7.34% |
| Barrier share | 17.15% | 13.82% |
| Long-scoreboard share | 56.18% | 57.12% |

The long-scoreboard percentage does **not** decrease in this literal-source
capture, despite shorter completion time. It is a fraction of active warp
cycles, not an additive millisecond attribution. NCU replay times
667.648→494.944us are kept separate from the graph trials above.
Both literal sources pass memcheck/full leak, racecheck and synccheck,
2GiB/no-swap, with explicit release.

A bounded cuBLAS component comparator chose a **different** implementation
from the retained vLLM output: `cutlass_80_wmma_tensorop_bf16...32x32_128x1`,
grid16x16x4,128threads,72registers,18,432shared bytes including reserve.
Its graph-call median326.253us includes the GEMM call; its first-kernel
counter capture alone reports9,348,096instructions,8,081,460L2 sectors,
39.82%active warps and9.55%tensor activity. This is not the vLLM TMA
kernel, not a complete split-K counter sum, and not a correctness oracle.
It reinforces that lower byte traffic/higher occupancy alone are insufficient,
but cannot supply fresh vLLM-service counter attribution.

## Both-rank fixed-chunk retrace

Control: `/tmp/lunaflux-glm-http-sparse-hc-control-trace-20261020-v4`.
Candidate: `/tmp/lunaflux-glm-http-sparse-hc-candidate-trace-20261020-v3`.
Both use the same clean7cffe534 ARM executable, latest previously qualified
options, and the same7973-token request with0cached tokens. Only the AOT
ownership paths differ. Each capture analyzes the first64-token,
45-layer decoder chunk; embedding/head/sampling/HTTP are excluded.
Both complete cold/warm requests, then answer the separate17x19 correctness
request with323. All122 changed leaves execute in each captured chunk:
89deferred HC and11each sparse output/query-B/index-Q.

| Changed-leaf sum, ms | Cold rank0 | Cold rank1 | Warm rank0 | Warm rank1 |
| --- | ---: | ---: | ---: | ---: |
| Deferred HC before→after | 8.050→5.303 | 8.609→5.496 | 7.939→5.351 | 8.549→5.999 |
| Sparse output | 5.717→4.752 | 6.056→5.262 | 6.040→4.879 | 5.921→5.022 |
| Query-B | 2.839→2.297 | 2.369→2.090 | 2.539→2.124 | 2.360→2.143 |
| Index-Q | 1.162→0.919 | 1.142→0.919 | **1.061→1.132** | 1.074→0.929 |
| Total selected saving | 4.498 | 4.408 | 4.093 | 3.811 |

Index-Q's warm-rank0 regression is retained. Neither it nor the small-row
HC resource penalty is waived by the other improvements.

### Time and overlap, not overall throughput

| First decoder chunk, ms | Control cold | Candidate cold | Control warm | Candidate warm |
| --- | ---: | ---: | ---: | ---: |
| Rank0 wall | 286.038 | 284.986 | 287.231 | **288.974** |
| Rank1 wall | 279.324 | 277.364 | 281.212 | **282.440** |
| Rank0 compute union | 261.685 | 257.147 | 262.144 | 258.334 |
| Rank1 compute union | 263.238 | 260.354 | 264.183 | 266.047 |
| Rank0 TP union | 23.303 | 27.264 | 24.541 | 30.105 |
| Rank1 TP union | 15.477 | 16.486 | 16.495 | 15.694 |

The execution graph retains the same ordering and overlap:

```text
HC → KDA-or-sparse projection/attention → TP → HC → dense-or-MoE → TP → …
                                                     ├─ routed path ─┤
                                                     └─ shared path ─┘
```

Shared/routed overlap on warm rank1 is108.246→112.424ms; only
0.091→0.095ms of shared work is exposed. Do not add shared duration to the
routed critical path. Compute/TP overlap is zero inside this observed
decoder window. Candidate warm rank1's **unchanged** routed gate/up,
down and KDA projections take respectively2.809ms,1.290ms and0.898ms
longer. The disjoint MoE union increases4.575ms. Together with other
small differences this cancels the changed-leaf gains. The traces do
not establish whether clock/thermal variation, cache effects or another
cause explains those unchanged-kernel changes; none is declared proven.
**No consistent whole-chunk gain or production promotion is claimed.**

The following warm rank1 allocation reconciles exactly to each wall total.
Reference is the retained October19 vLLM rank1 capture, not a fresh paired
reference run; native columns are this turn's paired traces.

| Disjoint allocation, ms | vLLM retained | Native control | Native candidate |
| --- | ---: | ---: | ---: |
| HC | 3.950 | 10.087 | 7.690 |
| KDA projections | 26.705 | 34.656 | 35.554 |
| KDA other | 4.423 | 8.951 | 9.145 |
| Sparse projections | 9.373 | 17.720 | 16.307 |
| Sparse other | 2.116 | 1.851 | 1.888 |
| Dense FFN | 2.282 | 5.902 | 5.871 |
| MoE union, overlap counted once | 184.459 | 185.012 | 189.587 |
| TP | 17.469 | 16.495 | 15.694 |
| Kernel-free, not necessarily idle | 5.256 | 0.534 | 0.700 |
| Cross-family overlap counted once | 0.038 | 0.005 | 0.005 |
| Total | 256.071 | 281.212 | 282.440 |

### Both-rank TP bounds

Separate capture clocks are aligned only by the causal collective bounds;
uncertainty is0.245–0.256ms. No cross-host clock equality or wire-time
claim is made. Warm rank0 accumulated early-arrival lower/upper bounds
are8.118–20.899ms(control),10.666–25.572ms(candidate).
For the candidate's first collective, rank0duration6.790ms versus
rank1duration0.196ms gives rank0early-arrival6.460–6.706ms. This first
long interval is mostly waiting for its peer, not6.790ms of network
transfer. The observed rank0 TP increase must not be attributed solely
to NCCL transport or to the changed kernels.

Step-boundary handoff is outside the decoder totals: warm rank0
3.420→5.000ms, rank1 7.664→9.352ms. It is separately retained in
each `handoff.json`, not hidden inside a projection/HC kernel.

Reports: `/tmp/lunaflux-glm-shc-{control,candidate}-{cold,warm}-rank{0,1}-20261020-v1`;
exact changed-leaf reports
`/tmp/lunaflux-glm-shc-effect-{cold,warm}-rank{0,1}-20261020-v1.json`;
paired clocks/TP bounds
`/tmp/lunaflux-glm-shc-{control,candidate}-{cold,warm}-tp-20261020-v1.json`.

Fast monitor minima: control18,020,040KiB, candidate17,918,236KiB
(17.09GiB), both above16GiB. Maximum observed device allocation:
96,533/96,539MiB respectively, below102,400MiB. Owned-process swap
remains zero. All diagnostic ranks and component probes retire.

Local disk exhaustion briefly prevented creating the restoration monitor.
Only regenerable `_build/native/debug/test` and `check` caches were removed
(about397MiB); no source, model or measurement evidence was removed.
The prior qualified GLM service was restored, not replaced by this
unproven full-path candidate. Restoration root:
`/tmp/lunaflux-glm-sparse-hc-restoration-20261020-v1`.
Both USER services are active, Result=success, ExecMainStatus=0,
MemorySwapPeak=0. The private backend reports healthy and answers323,
finish_reason=stop. The bounded restoration observer has completed.
No diagnostic GPU owner or monitor remains active.

### Decision

Keep the ownership improvements explicit and unpromoted. The component
and leaf gains are demonstrated; consistent decoder-wall gain is not.
The next qualification must distinguish unchanged-kernel run variation
from cache/resource interaction rather than assign either cause by guess.
HC also needs separate compiled resource envelopes if small-row performance
is to be protected from the64-row bucket's resource peak.
Remaining TMA/matrix-HC alternatives require their own physical and numerical
contracts; external vLLM/cuBLAS comparisons do not belong in those compiler
contracts. No tolerance, model reference, or production numerical law was
changed by this campaign.
