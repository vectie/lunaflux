# Selected partitioned decode trace

The selected LunaFlux decode partial has substantial scalar reduction and load
dependency costs. Its merge is small. However, pure C16 serving attention is much
closer to the references than its large share of LunaFlux's request window
suggested. Partitioned decode is an optimization opportunity, not an explanation
of the entire remaining framework gap. This investigation changes diagnostic
tools only; it does not implement or deploy a faster kernel.

## Workload and selected execution

Qwen3 0.6B BF16 runs on the 48-SM DGX Spark GB10 with CUDA 13.0.88 and Nsight
Compute 2025.3.1. References use the pinned NVIDIA 26.01 images: vLLM
0.13.0+faa43dbf and SGLang 0.5.7+31b61bbe. The serving traces are the existing
4096-input, 64-output, C16 captures, after the affine prefill repair.

The qualified serving bundle selects c452, eight partitions, query tile 1,
KV tile 32, D128 and the `blockwise-f32-probability-v1` arithmetic law. The partial
uses grid `(32,8,8)`, block 64 and 33,040 bytes of dynamic shared memory. The merge
uses grid `(32,16,1)` and block 64. The capacity grid includes inactive rows:
at C16, 1,024 of the partial's 2,048 launched CTAs have active rows. Grouped Q/KV
head reuse is already present: two query heads share one KV head.

The exact selected cubin SHA-256 is
`bf663e1055588f47ff62c733714df00baae8e9999b281ef052a243c25f5bbd5e`.
The trace names the partial and merge entry points ending in `ep_3903` and
`ep_3904`; the diagnostic verifies this recipe and module before replay.

The first pure C16 step is not identical across engines. Sorted past lengths are:

- LunaFlux: 4096, 4097, 4098, 4099, 4100, 4101, 4102, 4103, 4104, 4105,
  4106, 4107, 4108, 4110, 4112, 4115.
- vLLM: 4096, 4098, 4100, 4102, 4104, 4106, 4108, 4110, 4112, 4114,
  4116, 4118, 4120, 4122, 4124, 4127.
- SGLang: sixteen rows at 4096.

The new offline probe replays each retained vector through the same unchanged
LunaFlux module with synthetic operands. The live reference counter runs cover
the C16 decode envelope, but do not collect new row markers, so they are not an
exact vector match. Paging layouts, operands and arithmetic contracts also
differ. Neither these counters nor instrumented traces are ordinary throughput
measurements.

## Pure decode serving attribution

Restricting the existing work traces to sixteen query rows of one token each
avoids conflating prefill, mixed steps and decode. Kernel activity means the sum
of CUDA kernel durations, not a reconstructed critical path.

| Pure C16 attention activity | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| Observed pure steps | 44 | 32 | 64 |
| Main attention calls | 1,232 | 896 | 1,789 |
| Main attention mean | 1,192.197 us | 1,157.115 us | 1,175.938 us |
| Merge calls | 1,232 | 0 | 1,788 |
| Merge mean | 6.360 us | none | 3.127 us |
| Main plus merge mean | 1,198.558 us | 1,157.115 us | 1,179.065 us |

SGLang has three main calls and four merge calls clipped by the measured window
boundary, rather than the complete 28 calls per layer family per step. Histories
and step ranges differ. The arithmetic sum of means is descriptive, not a
matched-shape speedup estimate. It is about 3.6% higher than vLLM and 1.7% higher
than SGLang, much smaller than the ordinary long-64 completion gap of 17.3% and
16.7% in the [latest serving report](BENCHMARK_AFFINE_POSITION_REPAIR_2026-10-03.md).

Across the whole LunaFlux window, the partitioned partial consumes 1,847.957 ms
and merge 10.358 ms. The merge is only about 0.56% of this attention chain. The
partial's approximately 37% share of our window does **not** mean a 37% excess
over the references. Other pure-C16 kernel activity averages approximately
7.43 ms per LunaFlux step, versus 6.38 ms for vLLM and 6.72 ms for SGLang; those
different-work traces motivate further attribution, not a claim of causality.

## Selected kernel hardware counters

These are single cold-cache NCU captures, not repeated unprofiled timings.
LunaFlux uses its traced vector and synthetic operands; references use live
near-envelope serving operands. Raw duration exports are normalized from
nanoseconds to microseconds.

| Main attention counter | LunaFlux partial | vLLM decode | SGLang decode |
| --- | ---: | ---: | ---: |
| Capture time | 1,264.736 us | 1,199.008 us | 1,192.416 us |
| Grid | 32 × 8 × 8 | 1 × 16 × 8 | 54 × 8 × 1 |
| Threads per block | 64 | 128 | 128 |
| Registers per thread | 80 | 236 | 56 |
| Dynamic shared memory | 33,040 B | 81,920 B | 9,216 B |
| Active warp occupancy | 8.62% | 8.24% | 65.31% |
| Eligible warps per scheduler cycle | 0.12 | 0.03 | 0.08 |
| Issue active versus active peak | 11.25% | 3.45% | 7.52% |
| Executed warp instructions | 61.517 M | 17.789 M | 42.207 M |
| Average warp latency per issued instruction | 9.85 cycles | 28.99 cycles | 105.50 cycles |
| Long scoreboard contribution | 2.86 cycles | 19.88 cycles | 49.82 cycles |
| Short scoreboard contribution | 3.16 cycles | 0.29 cycles | 1.28 cycles |
| Barrier contribution | 0.66 cycles | 3.36 cycles | 54.10 cycles |
| Wait contribution | 1.60 cycles | 2.94 cycles | 0.72 cycles |
| Source correlated excessive shared wavefronts | 0 | 1,856 | 61,440 |

The stall contributions are cycles per issued instruction. For example,
LunaFlux's short-scoreboard share is 3.16 / 9.85, about 32%, and long-scoreboard
share about 29%; these are not percentages of end-to-end time. NCU sampling can
attribute a dependency to an instruction immediately after its producer.

The LunaFlux partial executes 3.46 times vLLM's warp instructions and 1.46 times
SGLang's, but its captured time is only 5.5% and 6.1% higher. All engines have
load waits. High occupancy and low conflict counts alone do not predict speed.
No spill instructions were recorded. Source-local allocation is zero for LunaFlux
and SGLang; the vLLM source-local sector field is unavailable. DRAM byte and
bandwidth counters are unavailable in these GB10 exports, so this run cannot
establish equal DRAM traffic or a bandwidth ceiling. Hardware aggregate shared
conflict metrics remain nonzero even where LunaFlux source excess is zero.

The isolated merge capture takes 12.800 us and 177,408 warp instructions. Its
warm serving mean is 6.360 us. It should not be the first target for closing a
large serving gap.

## Instruction and source explanation

The actual frozen generated source declares one score owner per 32-lane warp.
Each head serially visits keys within its 32-key tile, computes scalar QK dots
and performs five shuffle reduction levels per key. PV accumulation also visits
keys serially. Its selected arithmetic does not contract multiply-add.

The captured partial executes 14.219 M FADD, 9.885 M FMUL, 6.576 M IMAD.U32,
5.585 M SHFL.DOWN and 4.203 M LDS.U16 instructions. vLLM instead executes
4.325 M BF16 tensor MMA instructions; SGLang executes 8.723 M FFMA.FTZ and
uses a different lane decomposition. These are different numerical schedules,
not freely interchangeable bitwise-equivalent implementations.

One correction is essential: vLLM's kernel name contains `splitkv`, but this
captured specialization is **not an eight-way split-K launch**. It has no combine
kernel on the pure C16 steps. The FlashAttention launch mapping makes grid y
the batch dimension and grid z the head dimension when splitting is disabled.
The demangled specialization and trace agree with that interpretation. Do not
infer partitions from the name or grid z alone. The local FlashAttention launch
template used to inspect this mapping is not the pinned container's exact source.

LunaFlux's double-buffered K/V stage uses enough shared memory to limit residency
to two blocks per SM. With only two warps per block, that is about four resident
warps per SM, consistent with measured occupancy. This is a resource constraint,
not proof that increasing occupancy will accelerate the chain.

The hottest LunaFlux sampled PCs include a barrier after asynchronous-copy
completion waits and bounds predicates dependent on page-index global loads.
Inspection of the preceding SASS shows the dependency, rather than evidence
that the barrier opcode itself performs the global load. The current address
program shares an address among vector consumers but retains page lookup and
validity dependencies in successive key-row copy slots.

The pinned SGLang container's `flashinfer/attention/decode.cuh` provides a more
specific comparison. The selected template has two shared stages, vec8,
bdx16, bdy2 and bdz4. It prepares a shared table of future KV offsets, refreshes
that table every bdx iterations, reloads future K after QK and future V after PV,
and reduces history work across z owners. Its 54 segment grid is not the same
as our fixed eight partitions per active row. It still uses block synchronization
and still exhibits substantial barrier and memory waits.

Our current plan prefetches the next K/V tile after QK, softmax and current-V
readiness, before PV. It overlaps the next copy principally with PV, rather than
independently advancing K and V stage lifetimes. This source fact plus the sampled
copy-readiness waits supports an experiment, not a promised speedup.

The owning source files are
[blockwise fold](../kernels/luna_cuda_attention_tile_source/source_blockwise_fold.mbt),
[decode rendering](../kernels/luna_cuda_attention_tile_source/source_blockwise_decode.mbt),
[address ownership](../kernels/luna_cuda_attention_tile_source/source_decode_address.mbt),
and [grouped effects](../kernels/luna_attention_tile_schedule/grouped_decode_effects.mbt).

## Next experiments in the functional compiler

1. Represent page-epoch and future-offset reuse as an immutable address plan;
   lower it to retained offsets so page-index latency is amortized across key
   rows and stages. Compare a same-law A/B before combining it with other changes.
2. Give K and V independent explicit stage lifetimes. Schedule K refill after
   its readers finish and V refill after PV. Derive waits from the effect graph;
   do not remove synchronization required for shared ownership.
3. Search key-parallel and grouped matrix decode alternatives under explicit
   arithmetic contracts. Carry ownership, tile shape, resource estimates and
   partition cost through IR; select using whole partial-plus-merge timing.
4. Attribute output, down, head and sampling on the same actual pure-step vectors.
   A decode-only improvement cannot be assumed to close the remaining serving gap.

No extra IR layer is needed merely to rename these decisions. The relevant
changes belong in pure ownership/address/lifetime planning and device lowering,
with measured selection offline. No profiling, filesystem checks or numerical
qualification enters the production token path.

## Reproduction and retained results

The diagnostic runner is
[profile_partitioned_decode.mbtx](../benchmarks/gpu_pipeline/profile_partitioned_decode.mbtx).
Pure-step attribution and complete raw metric parsing are in
[summarize_partitioned_decode.mbtx](../benchmarks/gpu_pipeline/summarize_partitioned_decode.mbtx)
and [decode_counter_metrics.mbtx](../benchmarks/gpu_pipeline/decode_counter_metrics.mbtx).
All three pass strict native warning-denied checks and their `--self-test` cases.
The reference helper argument must expose the expected launch-filter and cache
seams; an incompatible helper fails instead of silently profiling different work.

The terminal campaign is
`/home/wlc004s/lunaflux-partition-trace-20261003.nUNqLRpo`.
It retains three history-vector replays, both selected LunaFlux entry-point
counters, pinned reference counters and SASS, exact commands, numerical probe
results, memory samples and source snapshots. The identical-module replay is a
repeatability/correctness check, not an optimization A/B. An earlier setup failed
on a CreateNew header-copy collision before GPU execution; that directory is
preserved separately and contributes no measurement.

GPU runs were serialized with a monitored 32 GiB available-memory floor and
8 GiB container memory/swap caps. The user service completed successfully and no
diagnostic compute process remained. No production container was changed.
The archive excludes build caches and model/toolchain copies, has SHA-256
`2f968ec9e0fb1b54fa4daee80bec6905842eddb67c4bad470e115dd7434e0930`,
and is downloaded to
`/private/tmp/lunaflux-partition-trace-20261003.xtwTVE80/decode-handoff.tar.gz`.
