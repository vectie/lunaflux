# Balanced score maximum — dual Spark, 2026-10-05

## Decision

Reject this production change. Balancing the maximum dependency reaches the
selected executable, but does not improve the measured kernel materially.
Six paired medians range from −0.44% to +0.39%; none passes the unchanged
five-pair, 3%-per-pair gate. Restore owned production IR, renderer, fixtures
and generated interface exactly. Keep the external experiment, a reusable
offline numeric probe and the SASS dependency analyzer.

This rules out another specific hypothesis. It does **not** prove attention is
optimal or that the remaining framework gap is inevitable. No serving bundle
was rebound and no new end-to-end/vLLM/SGLang comparison is claimed.

## Budget and functional boundary

One maximum-tree alternative, three Q2048 workloads, five alternating pairs
each on both Sparks, 30 CUDA-event repetitions per member. .178 and .179 ran
timings concurrently; afterwards .178 ran sanitizers and .179 captured counters.
There was only one GPU job per host. CPU builds did not overlap unprofiled timings.

An experimental immutable `RowScoreMaximum` plan groups score elements by
their existing probability-row ownership and constructs a balanced maximum
tree. Pure construction covers irregular and power-of-two key-fragment counts;
leaf order and negative-infinity identity are explicit. CUDA rendering consumes
that plan. No model name, machine name, new runtime scan or request JIT appears.

This is maximum selection, not floating-point sum reassociation. QK/PV product
order, probability/denominator sum order, BF16 rounding, score scaling and
`approx-base2-f32-v1` remain unchanged. Maximum-number nonfinite and signed-zero
behavior was separately tested on the target device. The implementation also
compiled for the grouped matrix-decode caller of this shared renderer, but the
timing/correctness campaign here covers the named prefill kernel only. Rejection
restores both callers; no unqualified grouped-decode change remains enabled.

## Propagation beyond source text

Symbol: `lunaflux_attention_prefill_tile_compiler_exp2_v1`.
Q64/K64/D128, one stage, accepted page-batch addressing, block128, runtime row
envelope32 and grid63×16×1 remain fixed. Both artifacts use234 registers/thread
(240 allocated), 49,168 dynamic shared bytes, two register/shared-limited
resident CTAs, zero stack frame and zero spills. Two candidate compiles match.

Executable text stays78,208 bytes but its SHA changes:

- Baseline: `62e5de8dd753c52009f7b9385a6b14a2350024fbd422cadc684edbc75719f65c`.
- Tree: `c3a05d0df9f244d0a9aaa753b5dc32e968a9b33b3524fbea507ec8647e65cc72`.

Different executable bytes alone do not prove a useful dependency change.
`ako_maximum_chain.mbtx` therefore tracks register versions and FMNMX dependency
depth through the section before the first exponential, propagating depth
through butterfly copies and resetting it on other register definitions.
Regression tests cover register reuse, butterfly propagation and CSV quoting.

The observed maximum-selection dependency prefix is **19 → 8**, not an elapsed
cycle estimate. The baseline includes both row roots before its first
exponential; the tree overlaps that first exponential with the remaining
row's final maximum operation. Its prefix has38 versus37 maximum instructions,
but complete executed maximum totals are identical. Do not present the prefix
count as an instruction-count saving or a complete cross-iteration critical path.

## Paired timings

Reduction is the median of paired ratios, not the ratio of separately
summarized medians. Negative reduction is slower.

| Host | Rows / history | Baseline µs | Tree µs | Paired reduction | Worst pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| .178 | 1 / 28,672 | 6375.966 | 6355.958 | +0.212% | −1.281% |
| .178 | 2 / 28,672 | 6406.858 | 6374.867 | +0.390% | −1.287% |
| .178 | 2 / 8192 | 1913.732 | 1909.926 | +0.171% | −0.862% |
| .179 | 1 / 28,672 | 6603.842 | 6615.836 | +0.005% | −4.553% |
| .179 | 2 / 28,672 | 6601.001 | 6629.953 | −0.439% | −1.522% |
| .179 | 2 / 8192 | 1985.436 | 1988.346 | −0.147% | −4.552% |

No losing pair is discarded. The .179 two-row cells are classified as
regressions by the unchanged driver. Cross-host absolute differences are not
framework effects; decisions compare artifacts on the same host.

## Matched counters and next-action evidence

.179 Q2048/R2/H28672, two instrumented launches, baseline then tree.

| Metric | Baseline | Tree |
| --- | ---: | ---: |
| Total warp instructions | 891,530,112 | 891,530,112 |
| FMNMX | 35,526,656 | 35,526,656 |
| HMMA | 119,668,736 | 119,668,736 |
| MOV | 97,258,496 | 97,258,496 |
| FMUL | 123,408,384 | 123,408,384 |
| NOP | 25,242,624 | 25,242,624 |
| CTA barriers | 2,808,832 | 2,808,832 |
| Active warps, percent of peak | 16.023% | 16.039% |
| Average warp latency / issued instruction | 6.583 | 6.132 |
| Wait / warp latency | 33.26% | 35.09% |
| Long scoreboard / warp latency | 11.19% | 18.04% |
| Math-pipe throttle / warp latency | 14.96% | 15.64% |
| Barrier / warp latency | 5.27% | 3.26% |

Sampled not-issued wait counts at FMNMX fall4,484→1,453. But HMMA wait samples
are106,258→106,747, HMMA math-throttle samples83,575→83,984, and the hot integer
page-bound comparison's long-scoreboard samples47,040→45,857. These observations
show that the targeted maximum chain changes while substantial tensor and
load-dependent paths remain. Samples are neither elapsed cycles nor additive
allocations of a serving gap. Profile durations7.169/7.168ms are not used for
acceptance; stall-fraction changes alone are not performance wins.

Together with the [row-mask trial](BENCHMARK_AKO_PROBABILITY_MASK_2026-10-05.md)
and earlier unit-rescale trial, this argues against another round of maximum
or guard cleanup as the main speed strategy. Next isolate the **MMA issue /
operand-consumer pipeline or unresolved metadata-load dependency**, verifying
that its executed dependencies change. Do not infer a complete solution from
source-level tree depth, occupancy or instruction totals alone.

## Correctness and memory

All30 timing pairs are bitwise equal to baseline, maxabs0 and sampled BF16
oracle maxabs≤0.003. Q129/R2/H128 memcheck/racecheck/synccheck pass with zero
errors/hazards and oracle maxabs0.000330008. Q2048/R2/H28672 memcheck passes,
oracle maxabs0.000377474. This is not whole-model quality admission.

The standalone `score_maximum_numeric.cu` probe compares linear and balanced
CUDA fmaxf over4,096 deterministic16-element cases: all signed-zero/nonfinite
edges, mixed edges and random float bit patterns. It passes bitwise on both
Sparks. .178 also passes full-leak-check memcheck; the probe checks each CUDA
status and deterministically releases both allocations. It is offline-only,
not linked into production. This numeric primitive check does not qualify
unmeasured kernel families or shapes.

Minimum observed MemAvailable121,585,444KiB, reserve33,554,432KiB. CPU units:
8GiB/no swap/TasksMax128/300s. GPU units:16GiB/no swap/TasksMax64/600s; counter
capture uses authorized root access. GPU jobs are serialized per host; both
are idle at terminal checks. Unified-memory GPU usage is not invented from
unavailable nvidia-smi fields.

Experimental native tests: physical IR33, renderer88, lowering7, all passed.
Restored production:32+88+7, all passed. Scoped checks deny warnings with the
existing migration exclusions20/79/29/25. Generated owned interfaces match
after scoped `moon info`. Offline schedule/report/maximum-chain tests pass4/4.
Unrelated working-tree changes are preserved; no whole-tree-clean claim.

## Preserved reproduction

CUDA13.0.88 nvcc SHA-256:
`fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.
GB10/sm121 UUIDs: .178 `GPU-9c3d3cf0-439a-5da2-67e9-20414255879f`,
.179 `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.

- Local root: `/tmp/lunaflux-ako-maximum-tree-20261005.Dgrhg2uJ`.
- .178 root: `/home/wlc003s/lunaflux-ako-maximum-tree-20261005.wPW05bS9`.
- .179 root: `/home/wlc004s/lunaflux-ako-maximum-tree-20261005.472AGLbr`.
- Inputs: `206dee4eff42086bc74100c5067cef2af07194ed7c45522d67103142f8e3a09f`.
- Candidate CUDA source: `48c91eb172ed5443d6c2bcff6051c4b6b79e50d7773b08f2ddc7e1324904d971`.
- Candidate cubin: `43ca1fde0cc2a274c297515c2898880949b00684b3d360cbdf9ccfa5d1c0979e`.
- Final source snapshot: `ae28d38afb4b51fa16751fa51971bd3c1356bb524b9f35f74c25dad5df692690`.
- .178 downloaded archive: `eeac88fa109aa24f8cb585ce7978ed2eae1ea59683043764765170bbc2b7fda5`.
- .179 downloaded archive: `0514dda1b94ac4d0b92c9d1334be4346c873f12d86db90f85134aa80bcb15bc8`.

Both archive hashes and every measurement-manifest entry verify locally using
new non-overwriting paths. The local root includes executable identity,
complete-prefix dependency analysis, raw SASS/counters, paired summaries,
numeric/sanitizer output, extracted campaigns and the final experimental source.
The offline `maximum-tree-v1` driver requires that archived source snapshot,
not restored production, to regenerate the rejected tree implementation.
