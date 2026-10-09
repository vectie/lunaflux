# Selected-kernel framework gap: service trace, counters, and source

## Conclusion

The remaining long-C16 deficit is principally GPU work, not a missing compiler
layer or excessive host launch gaps. The largest measured family deficits are
attention and output/down projection, followed by gate/up. Full ingress is
slower than vLLM's complete ingress chain, but is **faster than SGLang's chain**
in this capture. The vocabulary head is not the dominant bottleneck here.

This is a diagnostic change only. Production kernels, selection, binaries, and
deployment were not modified. The working tree contains unrelated changes,
which were preserved.

## Workload and measurement boundaries

- One GB10/sm121, 48 SMs, CUDA 13.0.88; GPU UUID
  `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.
- Qwen3-0.6B BF16, identical token-ID requests, greedy decoding, ignore EOS,
  prefix reuse disabled. Fresh service traces: input4096/output64/concurrency16.
- LunaFlux uses the frozen final v6 selected worker/AOT package described in
  [the current-source reproduction](COMPILER_MEASURED_RUNTIME_JOIN_2026-09-30.md).
  Kernel source commit is `730378cc`; source archive SHA-256 is
  `61420f9f21bd1cfd24c68fb6ae1e59767c19ec4b7bbf452d0c17b3125672b906`.
- References use the saved, image-ID-pinned NVIDIA 26.01 containers, serially,
  with their original serving configurations. These are not assumed identical
  to the newer sibling `../vlm` and `../sgl` checkouts.
- Each reference trace has an unprofiled warmup and one measured request batch.
  Luna's measured batch is selected using client epoch timestamps against the
  Nsight session epoch. Tables describe the retained finite trace window.
- Kernel replay is separate from serving timing. Reference replay KV allocation
  fraction was reduced from 0.5 to 0.15 to preserve memory headroom; shape limits
  and prefill chunk settings were retained. Replay durations are **not** serving
  results, and allocator/cache/clock effects prevent treating them as speedup
  estimates.

The fresh trace batches complete in **5135/4190/4226 ms**, respectively. They
reproduce the previous unprofiled medians **5175/4201/4244 ms**. The prior
4096/256/C16 medians remain **14081/11861/12003 ms**: Luna takes 18.7%/17.3%
longer, at 290.89 versus 345.33/341.25 output token/s. No new unprofiled campaign
or performance improvement is claimed in this investigation.

## 1. Whole-request GPU attribution

Sum of kernel durations, milliseconds; equivalent request batches, not identical
per-launch shapes. Output/down includes the references' combined decode GEMM
family. Ingress includes QKV, QKNorm, rotary, and KV-cache writes. Gate/up
includes activation. Attention includes partial/merge kernels.

| Family | LunaFlux | vLLM | SGLang | Luna minus vLLM | Luna minus SGLang |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attention, prefill + decode + merge | 2988.43 | 2530.50 | 2474.71 | +457.93 | +513.73 |
| Output + down | 637.13 | 340.84 | 332.61 | +296.29 | +304.52 |
| Gate/up + activation | 632.17 | 516.09 | 533.65 | +116.09 | +98.53 |
| Complete ingress | 545.27 | 454.64 | 577.23 | +90.62 | -31.96 |
| Norm/embedding | 103.73 | 90.39 | 128.09 | +13.34 | -24.36 |
| Head/sampling | 117.98 | 126.81 | 97.62 | -8.84 | +20.36 |
| Other | 0.00 | 32.59 | 88.36 | -32.59 | -88.36 |
| All kernel durations | 5024.71 | 4091.87 | 4232.26 | +932.85 | +792.46 |

These are accounting differences, not independently additive causal speedups.
References overlap some kernels and distribute mixed steps differently.
In particular, vLLM's causal/mixed prefill kernel includes decode rows that
Luna processes in a separate decode kernel. Comparing only the separately
labelled decode totals exaggerates the deficit. Comparing only prefill totals
can misleadingly make Luna look faster. **Combined attention is the defensible
whole-request comparison.**

The v3 mapping corrects two earlier analysis omissions: SGLang's tail
128x256 CUTLASS QKV/gate kernels and both references' attention merge kernels.
The raw names/configurations are retained, so the classification is auditable.
Generic reference copy/index kernels remain documented in ingress/other rather
than being silently discarded.

### Are bubbles the main problem?

| Trace metric | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| Kernel calls | 19836 | 35658 | 29969 |
| Kernel calls tagged as graph nodes | 19836 | 24647 | 25344 |
| Kernel+copy+memset activity span, ms | 5117.14 | 4171.13 | 4227.09 |
| Union of those activity intervals, ms | 5023.45 | 4091.74 | 4205.68 |
| Internal intervals without those activities, ms | 93.69 | 79.40 | 21.41 |

All observed Luna kernel calls in this window are graph nodes. Its excess
no-activity interval is about **14.29 ms versus vLLM**, or **72.28 ms versus
SGLang**, much smaller than the completion gap. Zeroing those gaps cannot
remove the measured kernel deficit. These intervals are not proof of a
particular CPU cause; CPU scheduling and CUDA API correlation would be needed
to attribute each gap. They exclude activity outside the selected window.

## 2. Output/down: too little useful reuse per physical tile

Actual dominant prefill configurations in the service traces:

| Kernel | Luna CTAs / threads | vLLM CTAs / threads | Luna regs / shared bytes | vLLM regs / shared bytes |
| --- | ---: | ---: | ---: | ---: |
| Output | 1024 / 256 | 80 / 256 | 74 / 32768 | 255 / 87040 |
| Down | 2048 / 128 | 128 / 256 | 92 / 24576 | 255 / 78848 |
| Gate/up | 1536 / 512 | 384 / 256 | 64 / 40960 | 232 / 73728 |
| Full ingress / reference QKV | 1024 / 128 | 256 / 256 | 118 / 32768 | 232 / 73728 |

More CTAs are not automatically bad. Here they demonstrate that Luna decomposes
the same large projection into many smaller reuse regions. More output tiles
repeat input staging; more row tiles repeat weight staging. Both also repeat
address/control/publication work. This is a source-supported mechanism; the
fresh trace establishes the total cost, not a fresh Luna DRAM amplification
counter.

Fresh vLLM prefill replays show the selected output/down kernels:

| Selected reference kernel | Warp instructions | Tensor-active % of elapsed | Achieved occupancy % | Long-scoreboard cycles / issued instruction | Barrier cycles / issued instruction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Output, NVJET 128x208x64 | 4307944 | 50.56 | 16.01 | 1.28 | 0.06 |
| Down, NVJET 128x176x64 | 6712608 | 49.67 | 16.13 | 0.99 | 0.04 |
| QKV, CUTLASS 256x128x32 stage3 | 13875200 | 52.30 | 16.58 | 4.61 | 4.21 |
| Gate/up, same CUTLASS family | 20812800 | 55.92 | 16.63 | 4.13 | 3.39 |

Actual NVJET SASS contains **UTMALDG.3D**, LDSM, and HMMA. Down has 3.24M HMMA
instructions out of 6.71M total; output has 2.13M out of 4.31M. Its instruction
mix spends much more of its work on tensor computation than a pipeline that
must explicitly distribute every vector copy. The low barrier/long-scoreboard
ratios show that the reference output/down pipeline is not predominantly
waiting at CTA publication or for ordinary load results. Math-pipeline and
sleeping waits also exist: this is not a stall-free reference.

Luna's source has correct double buffering, fragment sharing, and direct
accumulator stores already. The remaining limitation is the executable
transport/reuse schedule:

- [source_matrix_pipeline.mbt](../kernels/luna_cuda_projection_aot/source_matrix_pipeline.mbt):
  lines46–69 distribute vector input/weight transfers and calculate addresses;
  lines89–103 repeat that program per row/output tile.
- [source_fold_lifetime.mbt](../kernels/luna_cuda_projection_aot/source_fold_lifetime.mbt):
  lines36–57 realize the two-slot ring; lines119–138 lower typed publication to
  CTA-wide synchronization. Asynchrony overlaps copies, but does not remove
  their address/control work or change the reuse geometry.
- [source_sibling_reuse.mbt](../kernels/luna_cuda_projection_aot/source_sibling_reuse.mbt):
  lines51–73 publish two F32 result planes, change ownership, apply SiLU, and
  retire shared storage. Fusion saves an intermediate global round trip but
  still incurs shared staging and synchronization.

The reference gate/up cost must include its separate activation: its complete
chain remains faster here despite more kernels. Reference output/down retains
very high register use and only one resident CTA/SM, yet wins. Raising Luna
occupancy or removing bank conflicts alone is not the solution.

**Limit:** the fresh isolated Luna serving-counter wrapper exited6 before any
CUDA kernel. Thus this turn does not provide fresh Luna output/down/gate opcode
or DRAM counts. The attribution above combines actual service timings, fresh
reference counters/SASS, and Luna source/launch geometry. A precise quantitative
breakdown of Luna's extra copy/address instructions remains unmeasured—not
something to invent from tile size.

## 3. Decode: expensive scalar state transitions, not an absent feature

The selected current c441 report has 16 tokens/16 rows/history4095, grid16x8,
256 threads, 71 registers, two resident CTAs/SM, zero measured local-load/store
sectors. The retained selected replay executes **120696064 warp instructions**;
long-scoreboard contributes 31.23% and barriers 17.82% of active-warp stall
accounting in its selected observation.

Fresh reference serving replays include:

| Decode implementation | Configuration | Warp instructions | Tensor active % | Replay microseconds |
| --- | --- | ---: | ---: | ---: |
| Luna c441, retained exact-selected probe | grid16x8, block256 | 120696064 | 0 | 1358.14 |
| vLLM FlashAttention Q64/KV128 | grid1x16x8, block128 | 17788544 | 12.18 | 1192.64 |
| SGLang FlashInfer | grid54x8, block16x2x4 | 42206504 | 0 | 1198.11 |

The reference replays have different KV partitioning, an approximately4096-key
history, and a different profiling/cache environment from the retained Luna
probe. Treat these as instruction/schedule evidence, **not a matched latency
benchmark or a promise of 6.8×/2.9× acceleration**. SGLang's separate merge
kernel is included in whole-request attention but not in its decode row above.

Concrete source mechanisms:

- [source_grouped_split.mbt](../kernels/luna_cuda_attention_tile_source/source_grouped_split.mbt):
  lines88–107 calculate scalar dot/reduction results for each assigned key.
- [source_physical_program.mbt](../kernels/luna_cuda_attention_tile_source/source_physical_program.mbt):
  lines76–105 preserve the per-key state law: maximum/denominator update,
  exponentials, publication of two scales, and repeated output rescaling.
- Grouped-query reuse, async operand staging, and page-address hoisting **are
  already implemented and consumed**. The old diagnosis that hoisting was only
  metadata is no longer correct.

The pinned FlashInfer source is more specific than the earlier generic
“KV128 blockwise decode” explanation. Its selected template uses vec8,
16-lane reductions, two scores per local state update, two pipeline stages,
predicated 16-byte asynchronous loads, and `ptx_exp2`. Its `compute_qk`,
lines64–115, updates the maximum over the small score group **before** rescaling
the denominator/output. This is not the same algorithm as vLLM's tensor-based
FlashAttention decode. The two references should not be conflated.

Our blockwise/partitioned alternatives already exist. In the previous
unprofiled exact-C16 cell, partial+merge was approximately1294us versus1263us
for c441, so rejection was appropriate. The problem is an alternative that
does not yet win, not a missing selector. Another forced switch to a losing
schedule would recreate the previous regression.

## 4. Prefill and ingress: correct the stale diagnosis

### Prefill

Serving selects **async c322**, not sync c318. Query-owned fragments and direct
fragment forwarding are present. The current selected mixed probe
(2048 tokens,8 rows,history2048, padded row bucket32) has 230 registers,
95.08M warp instructions, 43.63% tensor activity, zero local-sector traffic,
21.46% long-scoreboard and 5.46% barrier stall accounting. These are retained
current-code probe counters, not a new same-shape reference pair.

The reference prefill/mixed launches have different query/history distributions.
Do **not** divide that 95.08M count by vLLM's first causal-prefill count and
call the result supporting-instruction overhead. Useful attention work differs.
The fresh complete-attention deficit is real; exact prefill instruction
amplification requires a matched position/mask/history replay.

Source-supported targets remain query-fragment lifetime/reloads, score/PV
ownership transitions, and the strict exponential/rounding contract in
[source_query_owned.mbt](../kernels/luna_cuda_attention_tile_source/source_query_owned.mbt),
[source_online_fold.mbt](../kernels/luna_cuda_attention_tile_source/source_online_fold.mbt),
and [source_query_numeric.mbt](../kernels/luna_cuda_attention_tile_source/source_query_numeric.mbt).
Our selected KV64 schedule is narrower than vLLM's KV128, but a legal KV128
alternative was already measured and rejected after mixed-phase spilling.
“Make KV128 the default” is therefore not a demonstrated fix.

### Ingress

The old hard16-row/4096-CTA claim is stale. The 16x16x16 guard in
[qwen_source.mbt](../kernels/luna_cuda_fused_parallel_aot/qwen_source.mbt)
describes a matrix microtile; the executable physical row tile is64, with
one head/CTA and1024 CTAs in the dominant prefill launch.

That still constrains the reuse domain. Projected accumulators cross shared
storage for the headwise epilogue, and mapping/rotary preparation is repeated
across head groups. Larger row/multi-head choices are legal and measured, but
lose after their increased live state/resource demand; merely enlarging them
is not enough. The selected ingress probe executes67.46M instructions versus
13.88M for the reference **projection alone**. This is not a whole-chain ratio:
normalization, rotary, and cache-write work is also inside ours.

Pinned reference Qwen source retains an independently scheduled QKV projection
followed by QKNorm/rotary/attention. The observed whole chain costs454.64ms
for vLLM,577.23ms for SGLang, and545.27ms for Luna. Fusion is therefore not
universally bad—or universally beneficial.

## 5. General functional-compiler remedies and falsifiable tests

| Priority | Compiler change, not a model special case | Decisive measurement |
| --- | --- | --- |
| P0 output/down | Joint reuse + producer/consumer plan: model repeated operand bytes/address work, permit larger output/row ownership and backend bulk transfer; express publication/retirement per consumer group instead of defaulting every transition to whole-CTA barriers | Same M/N/K/input-layout replay; source-correlated copy/address/HMMA counts, L2 sectors, stall ratios, then full-chain service time |
| P0 decode | Blockwise softmax state algebra with an explicit merge tree; tune score ownership, subgroup width, partition overhead and rescale frequency jointly | Same lengths/positions/pages/GQA; compare complete partial+merge, not partial alone; measure state updates, reduction/shuffle instructions and eligible warps |
| P1 prefill | Joint Q/KV/fragment lifetime plan; reduce reloads/remapping before increasing KV width; separate interior/tail arithmetic only where the current plan still emits it | Matched causal/mixed masks, ragged histories and useful-QK/PV work; normalize instruction categories by that work; reject spilling or end-to-end regressions |
| P1 ingress/gate | Keep independent GEMM geometry and epilogue ownership as executable alternatives; fuse only when saved traffic exceeds staging/repartition costs | Entire projection+norm+rotary+write or GEMM+activation chain, not a fused kernel versus only the reference GEMM |
| P1 scheduling | Derive prefill token/chunk and graph bucket choice from measured useful work, padding, KV capacity and latency policy | Equal-work mixed-arrival tests plus native-best-policy tests; actual useful/padded rows, histories, admissions and graph steps |

The architecture remains **pure semantic plan → immutable ownership/layout/
pipeline plan → explicit effect schedule → device lowering**. Tensor-memory
accelerator descriptors and warp instructions belong only to CUDA lowering;
bulk transfer capability and producer/consumer dependencies can be general IR.
More IR layers alone cannot improve a winning kernel whose executable work
remains unchanged.

Changing per-key folds to associative blockwise merges, using approximate exp2,
or changing probability precision is not bitwise-equivalent floating-point
optimization. Such alternatives need an explicit numerical contract with error
bounds, deterministic correctness and model-output checks. Copy/address/lifetime
changes can preserve the existing ordered fold and should be tested separately.

The prior compiler integration did not materially accelerate these winners:
the retained instruction pairs are ingress67.59M→67.46M,
prefill95.08M→95.08M, decode120.70M→120.70M. It correctly avoids bad choices;
it does not yet provide a better physical program for every dominant shape.

## 6. Is the test workload too uniform?

It is sufficiently revealing to establish the long-C16 GPU deficit, but not
sufficient for a universal policy. References have different chunk/mixed-phase
decompositions: Luna/vLLM use a2048 token budget; SGLang's retained configuration
uses8192. Compare both their native-best policies and separately matched work;
do not degrade a reference simply to obtain a favorable number.

The next diagnostic vector should vary input length1024/4096/8192/16384,
output32/64/256, concurrency1/2/4/8/16, staggered arrivals, ragged history and
chunk-boundary remainders. Replays must record actual Q/K lengths, causal offsets,
KV layout, GQA, padding and partial/merge scope—not just the launch grid.
Memory admission must use model/precision/KV bytes plus profiler replay reserve.
Do not run the entire Cartesian product simultaneously.

## 7. Reproduction, provenance, and limitations

Remote diagnostic root:
`/home/wlc004s/lunaflux-framework-gap-20260930.Lw5LblTl`.
Local download root:
`/private/tmp/lunaflux-framework-gap.Ra3cRz`.

Downloaded archive: `framework-gap-v3.tar.gz`, verified local SHA-256
`13b669559185e8710bbc6ef607be396bc5744f003d6230c44460d34be62d4919`.
The archive contains `FILES.v3.sha256`. It excludes source/build/runtime copies
and unreadable files; stopped-container logs/inspection retain failed-capture
state. Original remote files remain preserved. It includes the scripts as run;
subsequent local formatting and missing-metric handling do not alter that
immutable snapshot.

Relevant retained files: each trace's `trace.sqlite`, `gap-summary-v3.json`,
`gap-configs.json`, `gap-activity.json`, request bodies/results; reference
`counter.ncu-rep` and raw metrics; `existing-selected-counters-v2.json`;
pinned reference model/FlashInfer sources; memory samples; commands and failed
capture logs. Offline automation lives under `benchmarks/gpu_pipeline/*.mbtx`.
Raw units distinguish percentage-of-active-warps from cycles-per-issued-
instruction; they must not be compared as interchangeable percentages.

Pinned source SHA-256:

- vLLM Qwen: `53ff557561948f8bdfeb26172b876e0bccc524d974f489e6f18ddd4fd3c6ecb2`.
- SGLang Qwen: `fb2909e786d8a0c848659350ec573f5dba377621ad63efaee1b32dc5156ab5e1`.
- FlashInfer decode: `019d673aa848a938798a2c58b34b9b5813a3f137962cbbd90ef7bd71f636f373`.

Successful reference campaigns retained at least49.75GiB available system
memory (trace) and76.59GiB (counter replay), above the32GiB reserve. An initial
broad vLLM replay hit the reserve guard and was stopped; it is not a performance
result. Three isolated Luna serving-replay attempts exited6 before CUDA work,
without observed OOM; their cause remains unresolved. Thus Luna's existing
current selected counters are reused transparently, not relabelled fresh.
Those older reports did not collect per-PC execution metrics; their static
SASS exports cannot establish executed opcode counts. GB10 did not provide
usable DRAM byte metrics in these captures; use recorded L2/source metrics and
do not fabricate bandwidth utilization.

The NVJET implementation is not inferred from an unavailable proprietary source:
its observed geometry/counters and disassembled transport instructions support
the statements here. The current diagnosis identifies executable scheduling
deficits and next isolating experiments; it does not claim every stall or the
entire throughput gap is already causally explained.

## 8. Fresh blockwise decode counter correction and scalar-copy ablation

The new 23-variant decoder sweep is retained at
`/home/wlc004s/lunaflux-gap-decode-20260930.qsFvUfWA`. It varies C1/C8/C16 and
history127/4095 with the actual Q16/KV8/D128/page8 ABI. The blockwise/dual-score
frontier passes the independent attention oracle, but is not universally faster:
C1/history4095 c454-p8 is114.7µs versus315.6µs for c441, whereas C16/history4095
c454-p8 remains approximately5% slower including its merge. Neither result is
an end-to-end throughput or a matched vLLM/SGLang replay.

The paired C16/history4095 instruction capture isolates ordinary c441 and the
c454-p8 partial kernel. The partial duration below excludes its separate merge:

| Metric | c441 | c454-p8 partial |
| --- | ---: | ---: |
| Nsight replay duration | 1.307264ms | 1.389184ms |
| CTA count / threads per CTA | 128 / 256 | 1024 / 64 |
| Registers per thread | 71 | 115 |
| Dynamic shared memory per CTA | 33.82KB | 33.04KB |
| Shared-memory resident CTA limit | 2 | 2 |
| Achieved resident warps per SM | 15.50 | 4.13 |
| Achieved occupancy | 32.29% | 8.61% |
| Warp instructions | 120,696,064 | 79,890,432 |
| Issue activity, elapsed-cycle basis | 19.43% | 12.07% |
| Long-scoreboard cycles per issued instruction | 5.824 | 1.736 |
| Short-scoreboard cycles per issued instruction | 2.696 | 2.359 |
| Barrier cycles per issued instruction | 3.194 | 0.143 |
| Fixed-wait cycles per issued instruction | 1.925 | 2.226 |

The algorithmic reduction is real: executed SHFL.DOWN decreases5.243M→2.425M,
SHFL.IDX4.719M→1.638M and EX2 instructions2.101M→0.096M. Nevertheless the same
shared allocation with four times fewer threads permits only four resident
warps rather than sixteen. Extra grid partitions add waves, not resident warps.
Register count is not the measured residency limiter: the register limits are
three versus eight CTAs, both above the two-CTA shared-memory limit.

**Correction: zero register-spill counters do not mean zero local memory.**
c454-p8 executes LDL/STL from automatic retained-address arrays. Its source CSV
explicitly reports786,432 theoretical local sectors, not measured DRAM bytes:
STL.64 contributes262,144, STL131,072, LDL131,072 and LDL.64 262,144.
Each opcode variant executes32,768 times, totaling65,536 local loads and65,536
local stores. The CUDA stage producer stores `addresses[copies]`
and `sizes[copies]` between its key-copy and value-copy loops. The actual CTA
geometry changes copies per thread from four to sixteen, allowing those arrays
to materialize in local memory. Two IADD3 PCs immediately consuming LDL size
results account for19,790 of39,329 long-scoreboard samples, approximately50.3%.
Those PCs are `0x325b70d20` and `0x325b713f0` in the retained new capture.

Both kernels execute8,388,608 LDS.U16 instructions and have zero source-correlated
excessive shared wavefronts. Major new short-scoreboard sites are the BF16 unpack
instructions immediately following LDS.U16 in the QK dot loop. Thus this capture
does not justify returning to a bank-conflict-only diagnosis.

The isolated scalar-copy ablation keeps numerical law, score ownership, CTA geometry,
shared size, and key/value commit ordering fixed. A pure `RetainedCopyOwnership`
program derives finite vector/owner/slot bindings; CUDA lowering emits named
scalar address/size values instead of indexed arrays. It applies to async
decode generically, not to Qwen or this GPU by model/shape special case. Compare
old and new **c454-p8**, not old c454 against a newly modified c441. The scalar
representation may increase registers or compiler spills; the fresh results
below separately establish its effect. Consumer-geometry changes are excluded.

Paired downloaded raw/source CSVs are under
`/private/tmp/lunaflux-gap-fix-sources.JCW6KUwk/decode-c454-p8-isa-{0,1}-{raw,source}.stdout`.
The new scalar-copy export is
`/tmp/lunaflux-decode-scalar-copy-export.Hri14BRT/export.stdout`, SHA-256
`02aad76bdfc8e5b84d720f01d95efbea448094090f78a5d0234945fa71af7056`.
This hash identifies a source export, not a serving campaign. The prior export
and remote measured kernels remain unchanged for the A/B comparison.

### 8.1. Confirmed scalar-copy result, without changing geometry or law

All six C1/C8/C16 × history127/4095 physical cases pass the sampled independent
attention oracle, old/new comparison and KV-preservation checks. Six bounded
memcheck/racecheck/synccheck runs also pass. For history4095, paired timings of
the entire partial-plus-merge chain are:

| Concurrency | Indexed retained addresses | Scalar retained addresses | Time reduction |
| --- | ---: | ---: | ---: |
| C1 | 112.660µs | 110.613µs | 1.82% |
| C8 | 730.245µs | 663.802µs | 9.10% |
| C16 | 1319.347µs | 1256.525µs | 4.76% |

The fresh old/new C16 partial-only Nsight replay confirms the specific cause:
duration1.377216→1.288512ms; registers115→120; executed warp instructions
79.890M→78.447M. Executed local load/store instructions fall 131,072→0;
the old source CSV's theoretical local sector sum is 786,432. The new source
CSV omits the local-sector column and has no executed Local-addressed access
rows; therefore no new aggregate local-sector counter is asserted. The zero
executed LDL/STL counts confirm elimination of those accesses. The new compiled
kernel has zero stack/spill bytes. Long-scoreboard cycles per issued instruction
fall 1.738→1.230. The producer's indexed lifetime was therefore a real, fixed
source-level problem, despite zero register-spill counters in the old version.

This does **not** fix all decode constraints. Shared arena,1024 CTAs and64
threads per CTA are unchanged; occupancy remains8.26%→8.23%. Short-scoreboard
cycles per issued instruction remain2.361→2.464 and fixed waits2.226→2.228.
Both paths still execute8,388,608 LDS.U16 and have zero excessive shared
wavefronts. The remaining problem is not a local-array or bank-conflict claim;
it is the measured shared-load dependency/consumer-residency schedule.

A useful next resource hypothesis is storage plus ownership, not another blind
stage-count change. The allocated34,176 bytes per CTA leaves three such CTAs
just above the device's102,400-byte shared budget. A separately declared
register-owned score exchange or different cooperative consumer geometry may
improve residency, but must account for its added register/shuffle/lifetime
costs. Neither is part of this measured scalarization, and neither is yet
claimed faster. Earlier c441 timings are not a concurrently paired baseline
for the new source; no full-service or framework-parity gain is inferred.

The confirming paired captures are
`/private/tmp/lunaflux-gap-fix-sources.JCW6KUwk/scalar-copy-isa-{0,1}-{raw,source}.stdout`.
The ten affected attention/compiler/CLI suites pass190/190 with warning73
enabled (excluding preexisting toolchain warnings20/79/29). Software test counts
are separate from the physical and sanitizer results above.

## 9. Retained-query prefill: joint binding is not the missing speedup

The bounded joint-consumer campaign completed at
`/home/wlc004s/lunaflux-direct-ncu-20260930.K9JVg22n/prepared-joint`:
**72 timing runs, 18 sanitizer runs and six counter captures**. It compares
baseline c322 against c20644/c20645 using the actual Q16/KV8/D128/page8 ABI,
128 threads and 49,168 shared bytes. Twenty-four pure/mixed/ragged cells cover
129/512/2048 query tokens, one/16 rows and history0/4096; 32-row cells use
history0 only to stay inside the actual 16,384 page-entry bound. The independent
oracle, same-law bitwise checks, memcheck, racecheck and synccheck pass. These
are isolated AOT replay results, not new serving or matched framework results.

c10644/c10645 retain respectively 64/128 query dimensions across KV epochs.
c20644/c20645 preserve those lifetimes but consume each retained Q fragment's
four independent right-column products in one PTX region. The generic
`QueryOperandLifetime` records that consumption choice; IR, schedule and CUDA
lowering carry its canonical identity. Baseline and previous retained source
bytes remain unchanged for A/B comparison. No model-name branch, approximate
exponential, changed reduction order or forced production selection was added.

The long mixed2048-query/16-row/history4096 timing cell is near-neutral. Each
candidate below has its own paired baseline; do not treat the two baseline
samples as one common denominator.

| Candidate | Paired c322, microseconds | Candidate, microseconds | Change |
| --- | ---: | ---: | ---: |
| c20644, half retention + joint consumers | 2483.441 | 2469.064 | -0.58% |
| c20645, full retention + joint consumers | 2472.147 | 2480.682 | +0.35% |

The matched half-retention counter pair still shows **217.562M versus 211.482M
warp instructions (+2.87%)**, **235 versus229 registers**, no local-sector
traffic, long-scoreboard accounting39.34% versus39.92%, and barrier
accounting18.28% versus18.89%. These fractions describe the captured active-warp
stall accounting, not directly additive portions of elapsed time. A sub-1%
timing difference here is not evidence of an end-to-end improvement or of a
general winning schedule. **Keep c322 selected; do not promote either joint
variant on this result.**

### Why the shared PTX binding barely helps

The previous matched old-half capture already corrected the aggregate-copy
hypothesis: of6.556M additional warp instructions, register-source MOV accounts
for only0.306M. NOP, IMAD and LOP3 increases were approximately1.715M,1.565M
and1.296M respectively; HMMA and asynchronous global-copy counts were unchanged.
Named retained operands are bootstrapped once, and the existing matrix helper
takes its Q operand by const reference. Thus a returned C++ Q aggregate copied
on every key product was not an established dominant cause. The source-level
joint region removes repeated bindings/shared-base calculations, but does not
change the major persistent state or transport dependency chain.

Concrete unchanged source mechanisms:

- [source_query_lifetime.mbt](../kernels/luna_cuda_attention_tile_source/source_query_lifetime.mbt)
  retains Q fragments through **both QK and PV**, not only while QK consumes Q.
  [source_joint_query_consumer.mbt](../kernels/luna_cuda_attention_tile_source/source_joint_query_consumer.mbt)
  shares the consumer address/input region but cannot shorten that lifetime.
- [source_query_stage.mbt](../kernels/luna_cuda_attention_tile_source/source_query_stage.mbt)
  retains eight future-V 64-bit pointer values per thread for this geometry.
  The next K producer creates them before current softmax/PV; the V producer
  consumes them after current V readers retire. This is executable source
  lifetime evidence, not a claim that every pointer maps to a separate hardware
  register in the compiler's final allocation.
- [query_owned_effects.mbt](../kernels/luna_attention_tile_schedule/query_owned_effects.mbt)
  still orders K readiness/publication, V readiness/publication, and V-reader
  retirement with three CTA-wide publications per 64-key epoch. Joint Q binding
  changes neither those ownership edges nor async-copy waits.
- [source_online_fold.mbt](../kernels/luna_cuda_attention_tile_source/source_online_fold.mbt)
  keeps a 64-word/thread F32 output fragment and a 32-word/thread F32 score
  fragment at this D128/KV64 geometry. Probability conversion is per key fragment,
  so later score fragments remain live while earlier PV products execute.
  Strict `expf`, BF16 probability rounding, output rescaling and denominator
  updates are unchanged. Retained Q adds16/32 representation words, rather than
  removing these costs.

No spilling is necessary for this to lose: longer live ranges can induce
rematerialization/issue-spacing overhead even below the register limit. The
new counters do not identify the exact new per-PC cause without a joint-source
SASS comparison; they do show that joint binding has not eliminated the measured
supporting work or waits. The previous KV128 spilling result therefore cannot
be remedied simply by keeping more Q state or increasing the KV width.

### Remaining transport/lifetime direction, not another unmeasured default

The next executable alternative should plan Q/KV/score/probability/PV together:

1. Explicitly convert and retire the complete epoch-local F32 score into compact
   BF16 probability ownership before PV, preserving ascending exp/sum and each
   accumulator's MMA order. Compare that lower-live-state schedule with today's
   interleaving; extra materialization is not assumed free.
2. Represent future-KV addressing as compact page/segment affine products or
   producer-private state, instead of retaining full vector pointers in every
   compute thread. Preserve ragged, causal and mixed-history mapping; measure
   any recomputation/metadata-load tradeoff.
3. Choose producer/consumer roles, storage slots, issue/readiness/retirement and
   layout together. Derive barriers from the actual readers before reuse;
   deleting a barrier without changing ownership is invalid. Hardware descriptor
   transfers or warp-specialization belong to capability-gated CUDA lowering,
   not model semantics or a universal NVIDIA-only requirement.

This is a pure lifetime/transport refinement followed by explicit effects and
device lowering, not a request-path compiler or a model-specific optimization.
The current fragment-only peak estimate explicitly excludes address/control
temporaries; a usable resource model must include those live products and the
real ownership phases, then consume measured compiler resources. Only after
that should larger KV epochs be retested to amortize loops without spilling.
The pinned reference's selected transport/ownership must be compared on the
same useful-QK/PV workload; sibling source examples are architectural evidence,
not proof that the container used that exact implementation. No vLLM/SGLang
speedup, restored parity, or full-service performance gain is claimed here.

Offline reproducibility: `benchmarks/gpu_pipeline/run_prefill_lifetime.mbtx`
has separate `prepare-joint`/`run-joint` modes; the old modes remain unchanged.
The actual joint export is `/tmp/lunaflux-query-joint-export.fp7DQC`, and prepared
local inputs are `/tmp/lunaflux-prefill-joint-prepare.WrAaGqE7/prepared`.
`summarize_query_retention_sass.mbtx` parses executed per-PC counts, rather than
counting static assembly text. Nine affected attention package suites pass
183/183; physical results above are separate from those software tests.

## 10. Four-agent implementation and isolated GPU outcomes

The four workstreams now provide executable compiler alternatives, rather than
only selector scaffolding: projection producer/reuse geometry, prefill query
lifetime/consumer binding, blockwise decode/copy lifetime, and sibling MLP
ownership/transport. Plans remain immutable and backend-neutral; instruction
realization remains in CUDA lowering. No model-family branch, request-path JIT,
or token-step identity check was added by these workstreams.

All times below are five-trial unprofiled event medians on GB10, comparing
LunaFlux's own matched alternatives. They are **not new vLLM/SGLang ratios**,
and the percentages must not be added to predict whole-model speed.

| Matched isolated comparison | Before us | After us | Interpretation |
| --- | ---: | ---: | --- |
| Output, 2048 tokens, async 64x64 + producer address reuse | 211.947 | 157.814 | 25.54% lower kernel time |
| Down, 2048 tokens, 128-row reuse | 312.490 | 227.707 | 27.13% lower kernel time |
| Complete MLP with only that down change | 872.002 | 782.450 | 10.27% lower chain time |
| Decode C8/history4095, same c454-p8 geometry, scalar copy lifetime | 730.245 | 663.802 | 9.10% lower partial-plus-merge time |
| Decode C16/history4095, same scalarization ablation | 1319.347 | 1256.525 | 4.76% lower partial-plus-merge time |
| Gate, same coowned geometry, independent to segmented copies | 792.818 | 671.189 | 15.34% improvement versus the experimental control, still loses ordinary baseline |
| Gate, ordinary baseline versus guided segmented 64x64 | 511.354 | 544.450 | 6.47% slower; do not replace baseline |
| Prefill, half-retained joint consumer | 2483.441 | 2469.064 | Near-neutral; do not replace c322 |

Each listed campaign passed its deterministic probe/oracle checks and bounded
memcheck/racecheck/synccheck checks. The attention workload matrix includes
ragged/mixed histories and obeys the real ABI's page capacity. Tests are
serialized; the attention probe budget is below512MiB with32GiB system reserve.
These are not full-model numerical or full-serving qualification claims.

Winning projection source-probe arguments:

```text
output 2 4 1 4 4 1 255
gate-shared intermediate 2 4 1 8 1 255
```

The actual recipe-derived launch envelopes are used, not a probe override.
Compilation is nvcc13.0.88, sm_121, `-O3 --fmad=false --maxrregcount=255
-lineinfo -Xptxas=-v -cubin`. The production candidate-set API can select the
geometry through source-bound offline fold records; a diagnostic regression
checks that actual primary/companion launches and bounded row variants inherit
it. Its fixture latencies are explicitly synthetic, not physical observations.
The candidate export CLI forwards those fold records. The subsequent
[matched framework benchmark](BENCHMARK_FRAMEWORK_COMPARISON_2026-09-30.md)
also connects them to release binding and validates the compiler's selected
launch geometry instead of rebuilding an untuned launch. Fresh output/down
whole-chain measurements supply the records used by that serving bundle.
The isolated improvements above must still not be added together or substituted
for the measured full-service results in that report.

Final combined affected compiler/kernel/export suites passed415/415 native tests
with warning73 enabled and existing migration warnings79/20/29/25 excluded.
Scoped native check passed. Scoped `moon info` regenerated interfaces but reports
1179 existing dependency migration warnings; this is not a repository-wide
warning-clean claim. Formatting is restricted to affected source files; the
unrelated dirty working tree is preserved.

Remote repair roots are
`/home/wlc004s/lunaflux-direct-ncu-20260930.K9JVg22n` and
`/home/wlc004s/lunaflux-gap-decode-20260930.qsFvUfWA`.
The snapshot archive is
`/home/wlc004s/lunaflux-gap-repairs-export-20260930.FDD2PLpd/snapshot/gap-repairs.tar.gz`,
SHA-256 `2e67b7b894bdfbdf305f6705ebfe11a084569ff593bec64ccfb4aa6d28f1bf1c`.
It contains1864 readable artifact/log/source files, per-file hashes and an
explicit exclusion list; source/build/toolchain trees and unreadable failed
privileged-capture files are excluded without modifying originals.
The downloaded archive at
`/private/tmp/lunaflux-gap-repairs-download.9s7iA8Ag/gap-repairs.tar.gz`
has the same locally verified SHA-256; it does not overwrite an earlier download.

The gate/up remaining counter analysis is in
[the ownership/transport report](BENCHMARK_GATE_OWNERSHIP_TRANSPORT_2026-09-30.md).
Prefill's next substantive remedy is the joint phase-lifetime/transport program
in section9, not another query-retention-only patch. These unresolved paths
remain explicit: the four-agent source work does not establish framework parity.
