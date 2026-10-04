# How the selected reference decode kernels overlap work

Fresh GB10 counters do **not** support the premise that vLLM/SGLang have
eliminated waiting, or that LunaFlux's remaining serving gap is simply worse
warp-level latency hiding. They identify different computation and ownership
strategies, not an additive explanation of the entire serving gap.

No production kernel, numerical law, selector or deployment was changed.

## Capture scope

NVIDIA GB10, sm121, 48 SMs, CUDA 13.0.88, Nsight Compute 2025.3.1;
UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`. Qwen3-0.6B, head dimension
128, 8 KV heads, 16 query heads. Selected decode kernels only, not prefill
or a new end-to-end throughput campaign.

LunaFlux replays the unchanged qualified owned8 c468 module twice with
16 active rows, a 32-row launch bucket, history 4096 and synthetic fragmented
pages. References execute live C16 4096/64 requests; one launch is captured
per engine using the preceding near-envelope selection recipe. vLLM uses
global matching-launch skip 924; SGLang uses skip zero. Their exact history
vectors were **not freshly instrumented**. Retained `work-vectors.json` belongs
to the earlier trace, not these new launches. Kernel grids establish selected
geometry, not equality of operands, page locality or exact history lengths.

All captures flush caches (`--cache-control all`). Nsight replay durations
are diagnostic, not ordinary latency. They must not be substituted into the
previous matched serving comparison. GPU work was serialized. Luna counter
containers were capped at 8 GiB; reference containers at 64 GiB, without swap.
Minimum sampled host MemAvailable was 82,370,528 KiB for vLLM and 79,837,684 KiB
for SGLang, above the 32 GiB reserve. Both reference containers report
`OOMKilled=false`; exit 137 follows the recorded bounded diagnostic shutdown.
The final GPU compute-process query is empty.

## Fresh hardware counters

LunaFlux's repeat gives 1246.976 µs versus 1242.784 µs initially, with identical
41,646,848 executed warp instructions. The table uses the first capture.
Shared memory below is the kernel's dynamic allocation, excluding driver space.

| Metric | LunaFlux owned8 c468 | vLLM FlashAttention | SGLang FlashInfer |
| --- | ---: | ---: | ---: |
| Cold replay µs | 1242.784 | 1203.264 | 1203.840 |
| Grid | 32 × 8 × 1 | 1 × 16 × 8 | 54 × 8 × 1 |
| Threads/block | 64 | 128 | 128 |
| Registers/thread | 148 | 236 | 56 |
| Dynamic shared bytes/block | 33,040 | 81,920 | 9,216 |
| Achieved occupancy | 8.04% | 8.24% | 66.03% |
| Eligible warps/scheduler/active cycle | 0.15 | 0.03 | 0.08 |
| Active-cycle issue activity | 12.64% | 3.40% | 7.42% |
| Executed warp instructions | 41,646,848 | 17,788,544 | 42,206,504 |
| Average warp cycles/issued instruction | 11.36 | 29.38 | 106.97 |
| Long-scoreboard cycles/issued instruction | 2.78 | 19.67 | 48.19 |
| Short-scoreboard cycles/issued instruction | 3.44 | 0.29 | 1.30 |
| Barrier cycles/issued instruction | 0.42 | 3.72 | 55.74 |
| MIO-throttle cycles/issued instruction | 2.58 | 0.11 | 0.65 |
| Source-correlated excessive shared wavefronts | 0 | 1,856 | 61,440 |

Waiting values are normalized warp-state counters, **not percentages of
kernel wall time**. Concurrent warps, instruction mix and differing launch
work prevent using them as additive milliseconds. NVIDIA documents these
semantics in the [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
Fewer instructions can increase cycles/issued-instruction without increasing
absolute waiting time. High occupancy does not imply many eligible warps.
DRAM byte/throughput metrics are unavailable in these captures, not zero;
this round does not prove bandwidth saturation.

## vLLM: amortize work and overlap two dependency chains

The executed kernel specializes `Flash_fwd_kernel_traits<128,64,128,4,...>`:
head dimension 128, query tile 64, KV tile 128, four warps. The captured grid
has 128 CTAs. The `splitkv` symbol is not proof of eight history partitions:
the captured specialization has `Split=false`, and grid Z represents KV heads.

Exact device source was recovered read-only from the pinned vLLM image, not
inferred from a different checkout. Its unmasked mainloop in
`flash_fwd_kernel.h:957–1007` does the following:

1. Wait for current K; issue current V's async copy, then compute QK MMA.
2. Wait for V and retire K readers; issue next K's async copy.
3. Compute softmax and PV while next K is in flight.

`utils.h:139–184` additionally loads the next shared-memory fragment before
issuing the current fragment's MMA. `kernel_traits.h:74–144` defines swizzled
producer/consumer layouts and 128-bit copy ownership. Its paged copy assigns
contiguous row slices to each thread to amortize page lookup. These are
executable dataflow decisions, not merely stage-count metadata.

Fresh SASS confirms 544,768 `LDGSTS...128` warp instructions, 4,325,376 BF16
HMMA instructions and `LDSM` fragment loads. It does **not** establish TMA or
warp-specialized Hopper/Blackwell execution for this selected path.
Total instructions are 2.34× fewer than LunaFlux, but these are distinct
SIMT/tensor schedules and numerical laws, not a same-law compiler speedup.
Probabilities are converted to BF16 for PV; softmax uses log2 scaling/exp2.

The hottest sampled reference PCs are barriers immediately after
`DEPBAR.LE 0`, with long-load wait samples. Another hotspot is a page-index
consumer following `LDG`. Its loads are overlapped but **not fully hidden**.

## SGLang: compact vector state and more independent segments

Fresh selected template:
`BatchDecodeWithPagedKVCacheKernel<0,2,1,8,16,2,4,...>`.
Two stages, eight components/thread, 16-lane component groups, two query-head
groups and four Z groups produce a 128-thread block. The grid has 54 planned
segments × eight KV heads. This is not evidence that every request has the
same partition count.

The retained pinned `flashinfer-decode.cuh` explains the distinct ownership:

- Q and eight output components remain in thread-local vector state.
- QK uses vector BF16 loads and 16-lane reductions; scores remain in local
  `s[]` state and feed PV without a separate shared score publication.
- Two-stage K/V copies commit independently. Next K overlaps PV; next V
  overlaps the following QK. Future page offsets are prepared in epochs.
- More history segments provide independent CTAs; small shared/register
  footprints allow more resident work.

This is still SIMT, not a selected tensor-core decode kernel. Fresh SASS shows
8,722,944 `FFMA.FTZ`, 2,101,248 `SHFL.BFLY`, 1,062,912 `LDS.128` and 531,456
128-bit async-copy instructions. It also executes 1,055,232 workgroup-barrier
instructions and has large load/barrier waits. The hottest contexts contain
`DEPBAR.LE 3 → BAR.SYNC → LDS.128`. Compact state and segmentation are useful
mechanisms, but this capture does not show globally fewer instructions or
less waiting than LunaFlux.

## What differs in LunaFlux, and what is already implemented

LunaFlux already has independent K/V retirement, two-stage async copies,
retained page origins and epoch lookup reuse in
[`source_decode_independent_pipeline.mbt`](../kernels/luna_cuda_attention_tile_source/source_decode_independent_pipeline.mbt).
Those mechanisms must not be proposed again as missing features.

The ordinary c468 fold differs materially:
[`source_blockwise_fold.mbt`](../kernels/luna_cuda_attention_tile_source/source_blockwise_fold.mbt)
uses distributed QK score ownership, shared score/probability exchange,
strict F32 probabilities and ordered value accumulation. Its SASS executes
4,202,496 `LDS.U16`, 5,914,112 `IMAD.U32`, 9,221,376 `FADD` and 8,958,208
`FMUL`. More fine-grained shared consumption and address/arithmetic work
are concrete differences; MIO and short-scoreboard counters warrant attention.
Zero source-excess wavefronts does not remove those costs.

However, the preceding [five ablations](BENCHMARK_DECODE_DEPENDENCY_EXPERIMENTS_2026-10-04.md)
already tested register scores, explicit FMA, offset reuse, QK interleave and
earlier metadata loads. None established a C16 gain. FMA removed 20.56% of
instructions with neutral replay time. Thus these differences identify
experiments, not proven remedies or the full 10% serving-gap attribution.

The strongest next comparison is a common standalone workload/oracle for
all selected entries: identical query/history vectors, page mapping, precision
contract and output semantics; alternating unprofiled repeats; counters for
the same invocation. Vary vector ownership, tile extent and segmentation
jointly, rather than adding stages in isolation. Include combine cost and
the actual selected serving route. Any altered association/FMA/BF16 law must
be explicit; it cannot be presented as a bitwise-preserving transform.

## Serving context: previous, not remeasured here

The last matched three-round [whole-serving campaign](BENCHMARK_SPARK_CLOSURE_2026-10-04.md)
remains the throughput comparison. Output tok/s at C16:

| Input/output tokens | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| 4096/64 | 220.93 | 243.75 | 242.71 |
| 4096/256 | 319.53 | 346.09 | 340.79 |

These rates were not inferred from this round's cold replay durations.

## Reproduction and preserved capture

Offline runner:
[`profile_reference_latency_hiding.mbtx`](../benchmarks/gpu_pipeline/profile_reference_latency_hiding.mbtx).
It passes formatting, native warning-denied script check and its seam regression
test (1/1). No production package changed, so no unrelated physical/full-suite
campaign was imposed.

Successful root: `/home/wlc004s/lunaflux-reference-hiding-v2-20261004.Rrij83wt`.
The first root, `lunaflux-reference-hiding-20261004.NVi5qUlu`, contains a
Docker diagnostic-name collision before reference GPU work. Unique names
fixed the harness; failed logs and previous containers were not removed.
The seal includes both roots, 435 readable files and 632 build exclusions.

Pinned images: vLLM
`73e1b0f3230a9377a3109d20be7f8607963a41a715eb3000bc647f5f73c198ff`;
SGLang `3a9d399434923689565e1a72f5c3fc409323ce039b65bdbe5c2d6ed14abad3a9`.
The vLLM headers were copied from retained container
`lunaflux-remeasure-vllm-53deab59`, whose image matches the profiled image.
SHA-256 of recovered `flash_fwd_kernel.h`:
`242cbc331bd09d6ccfd023aae2d31461484b2a4801dcf66733b4151f9018622a`.
The seal also includes `kernel_traits.h`, `utils.h`, `softmax.h`, retained
FlashInfer source, raw reports, SASS, metric JSON, command recipes and memory logs.

Downloaded archive/inventory:
[`benchmarks/results/reference-hiding-20261004.M6B5y69E`](../benchmarks/results/reference-hiding-20261004.M6B5y69E/).
Remote seal:
`/home/wlc004s/lunaflux-reference-hiding-archive-20261004.jnzUiObu/sealed`.
Downloaded archive hash matches:
`894fd1657a5b84b44b8a665be771acfb4f5226b292aacae48be03125bdc4b16d`.
Nothing was extracted into the MoonBit module or overwritten.
