# Remaining framework gap: exact-wave traces, counters and source

## Conclusion

The largest remaining tested gap, Qwen3-0.6B BF16 at **32512 input / 64 output
tokens / concurrency two**, is primarily GPU work, not unexplained host bubbles.
Against vLLM, decode attention adds approximately 916 ms in the fresh profiled
wave; prefill attention offsets about 333 ms of that. Against SGLang, both
prefill and decode attention are slower. This is not a universal diagnosis for
every model, shape or machine.

The selected long-context decode has two concrete disadvantages:

1. It remains unsplit at C2 because measured decode-route coverage ends at 8K,
   and the blockwise split fallback is restricted to one request.
2. It executes **2.41 times as many warp instructions** as the selected vLLM
   decode invocation: scalar FP32 arithmetic and operand/index operations,
   instead of the reference's tensor-core algorithm. Async copying is already
   present. Global-load and CTA-barrier waits are not its dominant latency
   contributions in this capture.

There is also a calibration-envelope mismatch and a scheduler fairness
difference. These require targeted fixes, not another IR layer or a guessed
pipeline rewrite. No inference source or serving selection changed in this
investigation; only diagnostic automation and this report were added.

## Workload and version control

Spark .179, GB10 sm121, 48 SMs, UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, driver 580.178.04. Same token-ID
generator, greedy decoding, ignored EOS, prefix reuse disabled. CUDA work was
serialized. The unchanged repaired worker is
`dd4e8e2b5ddfb66cc6c901cf96b5ca14f12f9138841644c4fc2c4b4480928622`.
Fused bundle: `54753db487c343443226c97937efb0571ef7176212843fac03d8495960b2b726`.
Routes: `7b3dda884f123c2eff8b8ab26304fe1bd3a1bf147e079abf003fb2a1c054dbb0`.
Launch: `a601ea713669ee2ed9fd6cd6ca69a08c3fce176be04cc4e2252de55403d03555`.

Reference containers are the same pinned versions as the preceding benchmark:
vLLM 0.13.0+faa43dbf.nv26.01, FlashAttention, 2048-token chunks;
SGLang 0.5.7+nv26.1 (banner 0.5.7+31b61bbe), FlashInfer, 8192-token chunks.
This is a comparison of these configurations, not a claim that they use the
same chunk policy. Exact executed reference source was copied from the stopped
containers, rather than treating newer sibling checkouts as executed code.

The [unprofiled repair benchmark](BENCHMARK_PROJECTION_PROPAGATION_REPAIR_2026-10-06.md)
remains the throughput authority. These medians are **not** replaced by Nsight
or counter-replay timings:

| Input/output/concurrency | LunaFlux ms | vLLM ms | SGLang ms | Luna completion-time gap |
| --- | ---: | ---: | ---: | ---: |
| 128/256/C1 | 1726.5 | 2129.0 | 2115.5 | −18.9% / −18.4% |
| 4096/64/C1 | 689.0 | 770.5 | 779.5 | −10.6% / −11.6% |
| 4096/64/C16 | 4511.5 | 4197.5 | 4231.5 | +7.5% / +6.6% |
| 32512/64/C1 | 3939.5 | 3948.5 | 3718.5 | −0.2% / +5.9% |
| 32512/64/C2 | 7981.0 | 7255.5 | 6756.5 | +10.0% / +18.1% |

Negative means LunaFlux finishes sooner. The previous roughly 20% C16
regression was a lost projection policy and has been repaired. It must not be
reused as the explanation for these remaining gaps.

## Exact measured-wave GPU attribution

Client epoch bounds select only the measured wave, excluding warmup and
startup. Kernel families are classified by executed attention symbols; generic
GEMMs remain aggregated instead of guessing their operation from grid sizes.
GPU active time is the union of kernels, copies and memsets, not their sum.

| 32512/64/C2, fresh profiled wave | LunaFlux ms | vLLM ms | SGLang ms |
| --- | ---: | ---: | ---: |
| Client wall time | 7891 | 7195 | 6669 |
| GPU activity union | 7786.82 | 7088.82 | 6627.65 |
| No GPU activity within trace window | 106.46 | 106.52 | 43.91 |
| Prefill attention, summed kernels | 3490.26 | 3822.77 | 2812.90 |
| Decode attention + merge, summed kernels | 2619.71 | 1703.56 | 2066.45 |
| All other kernels, summed | 1677.91 | 1562.59 | 1764.97 |

LunaFlux minus vLLM: decode **+916.16 ms**, prefill **−332.50 ms**, other kernels
**+115.31 ms**. This accounts for approximately 699 ms of summed kernel time;
the GPU-union difference is 698.00 ms, with essentially identical inactivity.
The reference executes different mixtures of prefill and decode, so the
prefill total is not proof that our isolated prefill kernel is universally faster.

Against SGLang, prefill adds **677.36 ms**, decode **553.26 ms**, and other
kernels save **87.06 ms**. SGLang has some overlap; these sums are not an exact
additive wall-time decomposition. Its GPU-union advantage is 1159.17 ms;
inactivity differs by 62.55 ms. Host API durations overlap GPU work and are not
added to either table.

LunaFlux executes 95 graph steps: 32 prefill/mixed and 63 pure decode. The heavy
cached QKV kernel has the repaired grid 32×32 and 116 registers. Ordinary
steady-C2 decode executes 1736 times (62 steps × 28 layers), grid 8×8×1,
64 threads, 148 registers. The eight-part decode appears only in one graph,
not throughout steady C2. There is no evidence of a return to the old QKV
geometry or of graph fallback as the main gap.

## Token timeline and scheduler source

| Request order within each captured wave | TTFT ms | Median inter-token gap ms | Largest gap ms |
| --- | ---: | ---: | ---: |
| LunaFlux request 0 | 4816 | 49 | 53 |
| LunaFlux request 1 | 2404 | 49 | **2412** |
| vLLM request 0 | 5041 | 37 | 41 |
| vLLM request 1 | 2424 | 38 | **258** |
| SGLang request 0 | 2114 | 39 | **2117** |
| SGLang request 1 | 4190 | 39 | 41 |

LunaFlux's first-prefilled request pauses while the second prompt prefills.
[Scheduler selection](../scheduler/core/planning_selection.mbt:97) gives an
eligible aged prefill the remaining token budget before scanning runnable
decode rows. A full 2048-token chunk leaves no token for decode until the mixed
tail. The captured vLLM scheduler schedules running requests before waiting
requests, and its trace interleaves decode during the second prefill.

**SGLang also pauses decode in this configuration.** Therefore interleaving
cannot explain its advantage. Its larger chunks and ragged-current/paged-history
attention implementation are relevant: pinned `flashinfer_backend.py:825–841`
executes causal current attention and noncausal historical attention separately,
then merges state. Fresh SGLang hardware-counter capture failed; this report
does not invent an instruction-level SGLang attribution from its trace.

The pause is mainly time spent doing prefill, not an extra idle period. Its
duration and the decode-kernel disadvantage cannot be added as independent
predicted speedups. Changing fairness changes the mixture and batch sizes;
the whole chain must be remeasured.

## Why the selected decode remains expensive

vLLM counters are from the actual selected C2 serving invocation, grid
1×6×16, after skipping the preceding single-request decode calls. LunaFlux
serving-counter interception failed at its worker launch seam. The successful
LunaFlux counter probe therefore uses **unchanged AOT cubins and exact traced
8-row launch geometry, two live requests, history 32511, synthetic immutable
KV**. This is an explicit limitation, not a live-request counter claim.

| Counter | Luna ordinary | Luna split partial | Selected vLLM decode |
| --- | ---: | ---: | ---: |
| Replay GPU duration ms | 1.5207 | 1.2396 | 1.2131 |
| Registers/thread | 148 | 146 | 210 |
| Warp instructions | **40,973,152** | **41,328,896** | **17,006,400** |
| Achieved occupancy | 5.69% | 7.85% | 8.33% |
| Tensor-active, elapsed | 0% | 0% | 11.57% |
| Warp cycles / issued instruction | 2.76 | 11.36 | 33.66 |
| Long-scoreboard contribution, cycles | 0.20 | 2.83 | 23.68 |
| CTA-barrier contribution, cycles | 0.05 | 0.43 | 3.97 |
| Theoretical global L2 sectors | 8,344,000 | 8,387,968 | 8,369,472 |
| Register spills | 0 | 0 | 0 |

These are counter-replay timings, not throughput or the unprofiled probe's
event medians. The split-partial counter excludes its merge kernel.
Theoretical L2 sectors are **not actual DRAM bytes**; the report lacks that
GB10 byte metric. Occupancy and warp-latency averages are not stand-alone
performance rankings: vLLM waits more per issued instruction but issues much
less work and uses tensor operations.

SASS identifies the excess work, rather than merely suggesting “latency hiding”:

| Dynamic warp opcode | Luna ordinary | vLLM selected |
| --- | ---: | ---: |
| FADD + FMUL, non-FTZ scalar operations | **18,011,552** | reference uses different FTZ/tensor operations |
| IMAD.U32 | **5,853,120** | different addressing sequence |
| LDS.U16 | **4,161,536** | operand matrices primarily use LDSM |
| HMMA.16816.F32.BF16 | 0 | **4,177,920** |
| LDSM.16.M88.4 / LDSM.16.MT88.4 | 0 | **1,175,040 / 1,044,480** |
| LDGSTS 128-bit async copies | 520,192 | 525,312 |
| BAR.SYNC | 130,112 | 33,408 |
| MUFU.EX2 | 97,504 | 1,076,352 |

FADD+FMUL alone account for 44.0% of Luna's instructions. The selected
[blockwise fold lowering](../kernels/luna_cuda_attention_tile_source/source_blockwise_fold.mbt:63)
computes scalar QK dots, subgroup reductions and scalar weighted V sums.
Its ordered FP32 law does not contract multiply-add; explicit contraction is
already implemented for compatible fold contracts at line 144. It is wrong to
say that all FMA or matrix alternatives need to be implemented from scratch.

Luna launches 64 CTAs, but only two live rows × eight head groups = **16 useful
CTAs on 48 SMs**; inactive rows return early. Splitting launches 512 CTAs with
128 useful CTAs and improves parallelism. It hardly reduces total instructions.
The reference uses BF16 probability packing and tensor arithmetic: replacing
our ordered FP32 law with that algorithm is **not bitwise-equivalent**. Selection
must obey an explicit numerical contract and independent accuracy checks.

## Selection and calibration defects

1. [Decode calibration](../benchmarks/gpu_pipeline/measure_decode_routes.mbtx:236)
   tests histories 127/255/4095/8191. Its `--context-limit 32768` expansion covers
   prefill/mixed measurements, not this decode loop. The repaired route table
   therefore does not contain a measured 32K C2 decode winner.
2. [Blockwise split fallback](../engine/device_step/paged_decode_split_prepare.mbt:122)
   permits only one request. [Graph dispatch](../engine/device_step/graph_bucket.mbt:67)
   first uses a measured owner, then fallback thresholds. With neither measured
   32K C2 coverage nor an eligible split fallback, ordinary decode executes.
3. The frozen calibration probe defaults to a compact two-row envelope;
   [serving capture bounds](../engine/device_step/graph_bucket.mbt:217) are
   1/8/16/32, so C2 actually launches eight rows. A diagnostic compact replay
   passed numerical checks but was rejected as evidence of the serving geometry.
   The corrected replay explicitly uses the traced eight-row envelope without
   changing cubin workspace indexing. Logical shape alone is not sufficient to
   reproduce a graph launch's physical domain.

Five paired, alternating **unprofiled isolated** timings at this corrected
geometry give ordinary median **1444.02 µs**, split+merge **1180.37 µs**:
**18.26% less chain time**, bitwise equality, max pairwise error zero, sampled
FP64-oracle error 0.0002436, KV unchanged. This proves a useful existing route
is being missed for this probe. It does **not** prove an 18% whole-serving gain,
nor replace a full workload/calibration/sanitizer campaign.

## Recommended next implementation, in order

1. Carry the immutable physical launch/capture envelope into every calibration
   probe. Extend measured long-history decode coverage within bounded KV memory,
   bind records to exact artifacts, and verify executed graph owners. Test
   existing split, explicit-FMA and matrix alternatives before adding kernels.
2. Address the selected decode's scalar arithmetic/operand/index instruction
   budget through pure numerical and ownership plans, then device lowering.
   Keep strict and explicitly relaxed laws distinct. Benchmark the complete
   partial+merge or matrix chain, not only its fastest component.
3. Evaluate decode reservation during prefill and chunk-size alternatives as
   whole-wave policies. Compare TTFT, every token interval, completion time,
   actual batch/capture envelope and numerical outputs. Include both references;
   SGLang demonstrates that interleaving is not automatically necessary to win.
4. Reprofile the 4K C16 cell separately. Its remaining 7.5%/6.6% gap is not
   causally apportioned by a 32K C2 capture. The short-control 1.1% change also
   remains unattributed; no old/new paired short trace was collected here.

This retains functional layering: semantic/numerical contract → pure schedule
and physical-domain selection → explicit effect/ownership plan → CUDA lowering.
The measured deficiencies are coverage and executable choices, not proof of too
few compiler levels. Measurements must select among legal alternatives, not
silently weaken arithmetic or insert profiling/JIT into the token path.

## Evidence, safety and validation

Successful traces:
`/home/wlc004s/lunaflux-remaining-trace-20261006.0nD9XgJI` (Luna/vLLM), and
`/home/wlc004s/lunaflux-remaining-counters-20261006.ZeAoNRCq/sglang-trace/capture`.
Successful counters: the same counter root's `vllm/capture` and
`luna-isolated-v3`. The AOT module shared by ordinary and split entries hashes
to `318d74ff5b8ebfb5e0a97ea6daea16501a98a5ef1b67116350cf8b737266c185`;
frozen probe `014a0eeac3b27a3877eac139a0037e6615775dd0e75f477c41d442c7beccd66d`.

Pinned reference-source hashes, included in the archive:

- vLLM scheduler: `11c760f230d4e353900124c26dcde2ff678f8febbee0ea5f637c9b9f353eca8a`.
- vLLM FlashAttention backend: `31fdae4cdefa6a7459cb4a26270d2a8f2c5b7761ee204f48eb26faaf667a81d0`.
- SGLang FlashInfer backend: `02d7cb128a7c1e40913379921dce6861f464f3b09d38468c3e231fbc11578df8`.

Failures are retained, not relabeled: initial Luna NVTX interception and live
NCU worker interception exited before listening; initial SGLang trace omitted
the measured tail; SGLang NCU returned profiling-resource-unavailable, exit nine,
not an OOM; obsolete probe flags failed before execution; compact-envelope
replay was rejected for geometry mismatch. CUDA-only Luna tracing, explicit
SGLang start/stop profiling and direct traced-envelope AOT replay recovered the
scoped evidence above. The reason for the profiler resource failure is unproven.

Serving limits were 64 GiB, bridge 2 GiB, controllers/probes 8 GiB, no swap;
available memory reserve ≥32 GiB checked every 500 ms. GPU is idle at sealing.
No production services, model data, AOT inputs or inference source were altered.

[Downloaded raw evidence](/tmp/lunaflux-remaining-diagnosis-20261006.DjaF8C41/verified-diagnosis.tar.gz)
contains 483 readable files plus sorted inventories/digests and an explicit
exclusion list. Build/dependency trees and unreadable privileged failure files
are excluded, not deleted; copies of failed Luna logs are retained separately.
Archive SHA-256:
`21b0ceb8dd3f84f0e9c56029563af9527d30b6c1d55a47591bf74b72ca7edd7b`.

The AKO skill bounded the serial experiments and required selected-route,
counter, numerical and complete-chain distinctions. Validation is scoped to
the six new `.mbtx` diagnostic helpers: formatting and warning-denied native
checks passed; four focused tests passed (two helpers have no test entries).
The analyzer test checks interval overlap and exact nanosecond parsing. A first
test compilation incorrectly constructed a read-only JSON variant; it was
corrected before the passing run. These are not full inference-suite or fresh
kernel sanitizer results, and do not validate the unrelated dirty source tree.
