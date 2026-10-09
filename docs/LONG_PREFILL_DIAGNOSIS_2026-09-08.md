# Long-prefill diagnosis: phase, work, and physical schedule

## Scope

Read-only reanalysis of the September 8 physical traces for runtime `8f3708b`.
Qwen3-0.6B BF16, RTX 5060 Ti, input/output/concurrency = 1528/32/8.
No kernel modifications or new performance trials were made for this report.
The comparison is with installed vLLM 0.24.0 and SGLang 0.5.2, not a claim
about every upstream configuration. Neighboring repository code explains
implementation structure; it is not assumed identical to those installed versions.

## What the trace actually says

The latest order-balanced end-to-end result remains 159.825 output tokens/s,
versus vLLM 444.446 and SGLang 419.672 (2.78x and 2.63x throughput gaps).
The representative profiled LunaFlux window is 1617 ms; profiling and ordinary
timing results must not be substituted for one another.

Reconstructing forwards from groups of 28 QKV launches gives:

| Phase | Forwards | Approximate window | Forward interval |
| --- | ---: | ---: | ---: |
| Initial prefill | 8 | 9–790 ms | 97.5 ms |
| Later prefill/mixed | 4 | 790–1332 ms | 131.8–138.3 ms |
| Small-row decode | 31 | 1332–1618 ms | mostly 9.0–9.3 ms |

Thus about 1.32 seconds precedes the pure small-row decode tail. A request can
receive a token at 1061 ms, then at 1198 and 1331 ms, before settling into roughly
9 ms spacing. Those early 137 ms gaps include other requests' prefill work;
they are not measurements of a single pure-decode kernel.

The trace contains 336 large QKV calls (12 x 28 layers), versus 196 large QKV
equivalents in each baseline (7 x 28). This does **not** establish 12/7 times
as much arithmetic: chunk sizes and mixed batches differ. Prefix caching is
explicitly disabled by both baseline launch scripts. Increasing LunaFlux's
1024-token budget is a distinct scheduling experiment, not a demonstrated fix.

## GPU-time accounting

Cumulative kernel duration in the representative long-C8 request windows:

| Family, all phases | LunaFlux ms | vLLM ms | SGLang ms |
| --- | ---: | ---: | ---: |
| QKV projection | 270.98 | 77.98 | 85.16 |
| Output projection | 128.71 | 41.57 | 44.82 |
| MLP | 520.51 | 189.34 | 202.51 |
| Attention, including decode | 535.94 | 188.23 | 175.97 |
| Residual normalization | 19.70 | 8.70 | 8.34 |
| Vocabulary head | 33.91 | 28.18 | 28.92 |
| Sampling | 0.49 | 0.94 | 0.34 |

QKV/output/MLP account for about 611 ms more GPU time than vLLM; attention
accounts for about 348 ms more. These are accounting differences across whole
workload windows, not independently measured causal speedups.

Separating large-row matrix work is more revealing:

- LunaFlux: QKV 240.43 + output 114.86 + gate/up 283.02 + down 183.94
  = **822.26 ms**.
- Baseline large tensor-op GEMMs: **222.94 ms vLLM**, **240.17 ms SGLang**.
- LunaFlux prefill attention alone: **410.45 ms**.
- vLLM causal/prefill-shaped FlashAttention calls: **85.18 ms**.
- SGLang ragged prefill: **49.61 ms**, plus paged-prefill **14.38 ms**, with
  additional merge work. Comparing only 49.61 ms would omit part of its route.

LunaFlux's summed kernel time is about 1564 ms in a 1617 ms request window.
This trace does not support CPU dispatch, HTTP, or startup authentication being
the dominant long-prefill bottleneck. Kernel-duration sums are not a substitute
for an overlap-aware utilization calculation.

## Concrete physical-schedule issues

### 1. Large GEMMs still lack an effective multistage block schedule

`kernels/luna_projection_tile_compiler/compile.mbt` explicitly sets
`reuse_input_tile = false`. Semantic sibling-dot reuse exists, but cooperative
cross-warp materialization is not selected. The selected direct projection
generator issues global WMMA fragment loads and MMA in K=16 increments.
The MLP generator follows the same reduction structure. There is no selected
multistage block transfer pipeline in this path.

The baseline trace instead contains larger CUTLASS tensor-op schedules, including
`64x64_32x6` for vLLM and `256x128_32x3`/`64x256_32x4` for SGLang. These names
describe different physical tiling/staging; they do not alone quantify cache
misses or prove a particular bandwidth bottleneck.

The functional compiler needs a profitable realization of shared operand reuse,
larger block tiles, layout selection, and overlapped transfers—not just another
semantic CSE pass. The existing synchronous cooperative alternative adds two
barriers per K=16 step, so simply enabling it is not an established improvement.

### 2. MLP down projection launches mostly non-computing warps

`source_mlp.mbt` uses a 512-thread block, but `down_groups = min(4, groups)`
and the selected non-cooperative path returns when `warp >= down_groups`.
Only four of sixteen launched warps perform the down-projection matrix work.
The physical trace confirms block512, 104 registers/thread, and 16 KiB shared
memory, with 183.94 ms cumulative time for the large-row down kernel.

This mismatch is real. Its precise residency/stall penalty still requires
hardware counters or a controlled alternative launch; the report does not
claim that reducing block size alone yields a fourfold speedup.

### 3. Prefill attention materializes loop state repeatedly

The remote c312 source confirms Q32/K32, eight warps, matrix QK/PV,
shared-key factoring, and `LF_ASYNC_PREFILL=0`.

- QK matrix work is guarded by `warp < LF_KEY_VALUE_TILE / 16`: only two of
  eight warps perform that stage. The others participate in other stages.
- QK accumulators are written to shared scores; softmax creates shared weights.
- The output accumulator is shared FP32 state. Every KV tile rescales it in
  shared memory, then loads it into WMMA and writes it back after PV.
- For a full Q32/head128 tile, the accumulator is 16 KiB. Rescale read/write
  plus PV accumulator load/store imply approximately **64 KiB of logical shared
  traffic per KV tile**, before scores/probabilities and operand traffic. This
  is source-derived traffic, **not measured DRAM bytes**.
- The selected synchronous loop has seven CTA barriers per KV tile. A query
  tile reaching position 1527 visits 48 K32 tiles: up to 336 such barriers.
- Causal upper bounds already stop traversal at the maximum query position;
  this is not blindly processing the entire future context.

The missing compiler transformation is register-resident attention fold state
with compatible QK/softmax/PV lane layouts and synchronization scopes. Merely
reusing an SSA value does not retain it in registers across these stages.

### 4. Current-token K/V need not always take the paged-cache read route

LunaFlux's selected prefill loads both new and previous K/V through page-table
addressing. SGLang's trace shows a ragged-prefill route as well as paged work.
Its `flashinfer_backend.py` uses direct K/V for no-prefix prefill; when needed,
it computes ragged new-token and paged-prefix attention states and merges them.

A generic compiler can represent these as two memory views of the same logical
attention operands and compose stable softmax states. This does not require
Qwen-specific semantics. The isolated contribution of page addressing has not
been measured here, so it must not be assigned the entire attention gap.

### 5. Chunking amplifies latency, but cannot explain away slow matrix work

Twelve large forwards interrupt early decode tokens and repeat per-forward
launch/setup work. Seven larger baseline matrix forwards amortize that work
differently. A fair diagnosis must measure both the deployed configuration and
an equal-budget comparison. Counting launches is insufficient to establish
padding, redundant FLOPs, or exactly recoverable wall time.

## What not to conclude

- Larger tiles, more split-K partitions, or async copy are not automatically
  faster. The existing Qwen attention transfer experiment found c316 slower
  than c312 on measured paired shapes; see `QWEN_ATTENTION_TRANSFER_2026-09-08.md`.
- Sampling and vocabulary-head improvements will not close this long-prefill
  gap. Their remaining absolute costs are small compared with matrix/attention
  work.
- There is no evidence here that functional semantics themselves are costly.
  The gaps are in physical scheduling, storage/lifetime realization, and batch
  planning. Mutation-free IR can express all required transformations.
- No fresh Nsight Compute stall/DRAM/Tensor-Core counters were captured in this
  investigation. Code-derived diagnoses are distinguished from measured times.
- Cross-framework generated sequences are not uniformly identical. Comparisons
  retain the prior workload/correctness caveat; they are not exact numerical
  equivalence claims.

## Next discriminating measurements, before choosing a fix

1. Record actual M/N/K, valid rows, and issued tiles per forward; replay identical
   GEMM shapes against the selected baseline libraries. Measure Tensor Core
   utilization, memory traffic, eligible warps, register spills, and barrier stalls.
2. Isolate MLP-down launch geometry from changes to arithmetic and storage.
3. Replay identical attention descriptors for paged versus contiguous K/V,
   shared versus register fold state, and matched Q/K tiles.
4. Compare 1024/2048/4096 token budgets with the same kernels, reporting TTFT,
   mixed-phase token gaps, pure-decode intervals, and throughput separately.

## Trace locations

- Latest LunaFlux database:
  `/private/tmp/lunaflux-five-ops-profile-results-20260908-r1/trace.sqlite`.
- Its long-C8 client window is epoch ns
  `1788862209527147609..1788862211145749905`; session epoch is
  `1788862107178022270`.
- Latest compact kernel/request tables are in that directory's `compact/`.
- Baseline complete kernel/family histograms:
  `/private/tmp/lunaflux-operation-results-20260908.q8vVnF/{vllm-r2,sglang-r2}/i1528-o32-c8-t1-v2.*`.
- Inspected physical attention source:
  `/dev/shm/lunaflux-five-ops-runtime-20260908-r2/candidate/reusable-qwen-prefill-attention-c312/kernel.cu`.
- End-to-end results: `OPERATION_OPTIMIZATION_2026-09-08.md`.
