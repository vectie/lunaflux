# Row-owned probability visibility — dual Spark, 2026-10-05

## Decision

Do not enable this experiment in production. It changes executed code and
removes supporting instructions, but six matched timing cells remain
inconclusive. Median paired reductions are +0.02% to +1.40%; each cell contains
at least one non-improving pair. None meets the unchanged five-pair, 3%-per-pair
acceptance gate. Restore the owned production IR, lowering, source fixtures and
generated interface exactly. Preserve the experiment externally for reproduction.

Both Sparks ran timings concurrently. Afterwards .179 collected hardware
counters while .178 ran sanitizers. One GPU workload per host; CPU builds did
not overlap unprofiled timings. No serving rebinding, end-to-end improvement or
fresh vLLM/SGLang comparison is claimed. The accepted page-batch implementation
remains the production route.

## One bounded hypothesis

Keep Q64/K64, D128, one pipeline stage, accepted page-batch addressing, CTA
geometry, ordered sums, BF16 conversion and `approx-base2-f32-v1` fixed. Factor
probability validity through the already merged row maximum rather than check
each score independently.

The experimental pure `OrderedProbabilityPacking` plan deduplicates immutable
row owners. CUDA terminal lowering uses one finite-maximum mask per owner for
the same exponential and bit-mask operation. Prepared scores are finite or
masked negative infinity. A finite merged maximum makes masked scores
exponentiate to positive zero; an infinite maximum makes admitted probabilities
zero under the existing contract. Scalar tests cover finite and both infinite
score/max boundaries. No reassociation, fast-math change, model-name branch,
runtime compilation or token-path validation is introduced.

This is a general physical-domain factoring experiment, not a new mandatory IR
layer. Its unused public methods were removed after rejection. The final source
snapshot includes the experiment and regression tests.

## Executable propagation

Unlike the preceding [PV-window experiment](BENCHMARK_AKO_VALUE_WINDOW_2026-10-05.md),
this changes the selected executable text. Offline identity extraction compares
the unique executable ELF section, not container/debug digests.

Symbol: `lunaflux_attention_prefill_tile_compiler_exp2_v1`.

| Identity | Baseline | Row mask |
| --- | --- | --- |
| Executable bytes | 78,208 | 77,824 |
| Executable SHA-256 | `62e5de8dd753c52009f7b9385a6b14a2350024fbd422cadc684edbc75719f65c` | `015c121752b20c0e7543aa37ee134d4633617d47898a066140af544d8f394ba6` |
| Cubin SHA-256 | `7b42b3531466f5fe22c59e5aac2883db68984907c49a5637ac5f1358ad2eab57` | `85d940c4b2462eb0c7f787adc57f0adcc273ba411fedd620b89118d17250676f` |

Two independent candidate compilations match. Registers remain 234/thread
(240 allocated), with zero stack frame/spills. Both use block128, runtime row
envelope32 and grid63×16×1. The candidate artifacts, not an old serving bundle,
were executed in these kernel-only probes.

## Unprofiled results

Q2048; five alternating pairs per cell and 30 CUDA-event repeats per pair.
Reduction is the median of paired ratios, not the ratio of independent medians.

| Host | Rows / history | Baseline µs | Row mask µs | Paired reduction | Worst pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| .178 | 1 / 28,672 | 6392.920 | 6333.135 | +0.023% | −0.724% |
| .178 | 2 / 28,672 | 6381.980 | 6354.439 | +0.522% | −0.798% |
| .178 | 2 / 8192 | 1911.622 | 1896.004 | +0.817% | −0.022% |
| .179 | 1 / 28,672 | 6619.702 | 6554.811 | +0.778% | −1.131% |
| .179 | 2 / 28,672 | 6580.563 | 6525.262 | +1.106% | −0.620% |
| .179 | 2 / 8192 | 1988.463 | 1960.691 | +1.397% | −0.644% |

Comparisons are paired within one host. Different absolute host timings are
not a framework effect. These narrow gains are not a measured serving win.

## What changed and what did not

.179, two matched Q2048/R2/H28672 instrumented launches, baseline then candidate.

| Counter | Baseline | Row mask |
| --- | ---: | ---: |
| Total warp instructions | 891,530,112 | 866,287,488 |
| FSETP.NEU.AND | 31,787,008 | 3,739,648 |
| NOP | 25,242,624 | 28,047,360 |
| MOV | 97,258,496 | 97,258,496 |
| FMUL | 123,408,384 | 123,408,384 |
| FADD | 67,313,664 | 67,313,664 |
| MUFU.EX2 | 31,787,008 | 31,787,008 |
| HMMA.16816.F32.BF16 | 119,668,736 | 119,668,736 |
| LDSM.16.MT88.4 | 29,917,184 | 29,917,184 |
| LDGSTS.E.BYPASS.128 | 14,958,592 | 14,958,592 |
| CTA barriers | 2,808,832 | 2,808,832 |
| Register/shared-limited resident CTAs | 2 / 2 | 2 / 2 |
| Dynamic shared bytes | 49,168 | 49,168 |
| Active warps, percent of peak | 16.026% | 16.023% |
| Average warp latency / issued instruction | 6.145 | 6.197 |
| Wait / warp latency | 35.59% | 36.03% |
| Long scoreboard / warp latency | 11.22% | 11.25% |
| Math-pipe throttle / warp latency | 16.11% | 16.04% |
| Barrier / warp latency | 3.13% | 2.91% |

The executable change propagates: 28.05M fewer score comparisons, partly offset
by 2.80M additional NOPs, leaving 2.83% fewer total warp instructions. Loads,
tensor products, exponentials, output sums, register rearrangement and barrier
counts do not change. The hypothesis removes repeated predicate work, not the
dominant unchanged arithmetic/movement or copy-consumer dependencies.

Highest sampled not-issued dependency sites remain the same kinds of
instructions: an integer bounds comparison, completion/publication barrier,
warp synchronization and an integer predicate. Their addresses move in the
new binary. Samples describe blocked instruction sites, not independently
additive elapsed-time causes; they do not prove the comparison itself caused
the preceding load wait. Profile durations (7.251/7.149 ms) are excluded from
the timing decision. Small stall-fraction changes cannot establish a causal
pipeline improvement.

## Correctness, memory and local validation

All 30 pairs passed bitwise equality to baseline, maxabs zero and sampled BF16
oracle ceiling0.003. .178 Q129/R2/H128 memcheck, racecheck and synccheck passed
with zero errors/hazards; oracle maxabs0.000330008. Q2048/R2/H28672 memcheck
passed with zero errors; oracle maxabs0.000377474. This is kernel correctness,
not whole-model quality or runtime-leak qualification.

Minimum observed MemAvailable121,658,820 KiB, reserve33,554,432 KiB. GPU units
used MemoryMax16GiB/no swap/TasksMax64/600s. CPU units used8GiB/no swap/
TasksMax128/300s. The counter unit ran as root for authorized performance-counter
access; the other units were user units. An initial user-unit sudo attempt
failed before any workload because its credential cache was TTY-scoped; its
journal is preserved and it was not relabeled as a successful capture. GPUs
were idle at terminal checks. Unified-memory GPU usage is unavailable in
nvidia-smi here; no fabricated GPU-memory figure is reported.

Experimental affected tests passed128/128: attention physical IR33, CUDA
source88 and lowering7. Restored production passed127/127 (physical IR32).
Offline schedule/report/executable-identity helpers passed5/5. Scoped native
checks deny warnings with existing migration exclusions20/79/29/25. Generated
owned interfaces match after scoped `moon info`; this is not a whole-tree-clean
claim. Unrelated working-tree changes were preserved.

## Reproduction

CUDA13.0.88 nvcc SHA-256:
`fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.
GB10/sm121: .178 `GPU-9c3d3cf0-439a-5da2-67e9-20414255879f`,
.179 `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.

- .178 root: `/home/wlc003s/lunaflux-ako-row-mask-20261005.15MGyvN2`.
- .179 root: `/home/wlc004s/lunaflux-ako-row-mask-20261005.cseyDAda`.
- Local root: `/tmp/lunaflux-ako-row-mask-20261005.si3Ua8d6`.
- Input archive SHA-256: `58348358c24eedb3b6e13c8f96726d30a3a6a8b40027361814fd0195fae105de`.
- CUDA source SHA-256: `19a18b3ca6a257df078990a9a4b17a9df4d01aa657b830ab5373473ca1007e60`.
- Final experimental source SHA-256: `bb84916476fdce3d7c38624e88f145c7643107eb7c197d709103abd1ef0f342a`.
- .178 downloaded archive SHA-256: `5e9882728c500638ec5b3c1a868a825bf8ff904379986eace9435da36a66cc3a`.
- .179 downloaded archive SHA-256: `2ed82fbfcca6429ef6744dcf2ae2cbb1df9951be73cdecde4e130519156d5b56`.

Downloads/extraction use new paths; archive hashes and every measurement
manifest entry verify locally. The local root retains both extracted campaigns,
raw counters, SASS, paired summaries, executable identity and the full source
snapshot. The offline `probability-mask-v1` driver mode reproduces the archived
experiment when used with that snapshot, not restored production source.

## Next bounded direction

Do not spend another round merely removing finite-score guards. The measured
change already removes most of that repeated work without a robust gain.
Target an unchanged dataflow group—fragment rearrangement, scale arithmetic or
copy-consumer dependencies—and verify executable propagation before timing.
Keep numerical law explicit and CTA residency fixed for the first isolation;
do not change geometry and arithmetic together or infer a serving gap from
instrumented warp-latency percentages.
