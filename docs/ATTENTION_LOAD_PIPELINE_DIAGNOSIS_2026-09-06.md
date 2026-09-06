# Attention load-pipeline diagnosis — 2026-09-06

This continues the [fine-grained diagnosis](FINE_GRAINED_KERNEL_DIAGNOSIS_2026-09-06.md)
with NEW fixed-context measurements and instruction-level GPU counters.
No production kernel, routing policy, or deployment changed.

## Conclusion

The current decode kernel has a concrete **synchronous K/V staging bottleneck**.
Its split companion partitions the same expensive inner loop; it does not
replace that loop with a better load/compute pipeline. Splitting helps B1
substantially but barely helps B8 at long context. Merge is not the main cost.

For B8/L1528, around 90% of the **long-scoreboard stall samples**, not total
runtime, land on four shared-store instructions waiting for preceding scalar
global K/V loads. DRAM throughput is only ~27% of peak, while ~73–75% of
scheduler cycles have no eligible warp. This supports inadequate latency
hiding/dependency scheduling, not exhausted peak DRAM bandwidth.

The next compiler experiment should change vectorized staging and software
pipelining, with resource-aware tile selection. Merely enabling more split
partitions, more graphs, or more compiler passes is not supported as a fix.

## 1. Controlled experiment

- RTX 5060 Ti only; runtime UUID
  `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, PCI `00000000:17:00.0`.
- Unchanged current attention CUBIN from
  `/dev/shm/lunaflux-decode-qwen-55c0650-r1/attention.cubin`, SHA-256
  `258cbe310c5e6a7a8702c0fd858e68363efec7d8891a1fb93b4185682f885f04`.
  Matches module 8's digest in that runtime bundle; the module contains both
  direct decode and split8 partial/merge entry points.
- BF16, 16 query heads, 8 KV heads, head dimension 128, page size 8.
  Fixed context vector `[59,128,256,512,1528,4096]`, batch vector `[1,8]`.
  Each row has one decode query; no prefill/mixed-row work is measured here.
- Deterministic synthetic Q/K/V and noncontiguous physical pages. Allocations,
  input uploads, double-precision CPU oracle and output checks are outside timing.
- 200 identical launches captured in a CUDA Graph, three warm replays, one
  event-timed replay per measurement. Split total captures partial then merge.
  Three independent process runs per shape/grid; context/grid order reversed
  in the middle trial. Direct precedes split within each process: this is not
  a fully randomized trial. Clocks were not locked.
- Two grids: compact `grid.x=batch` and capacity `grid.x=32`. All other launch
  dimensions, data and machine code remain unchanged. 72 matrix runs plus a
  smoke run passed, with empty timing stderr. All tested outputs are finite,
  max absolute error <= 0.000244, and persistent K/V buffers are unchanged.
- Isolated partial/merge timings are supporting measurements. Cache and launch
  interactions mean their sum need not exactly equal measured pipeline total.
- Warm reuse of one layer is not full-model serving. Do not substitute these
  microseconds into a tokens/second claim or compare them directly to a
  baseline's varying-context serving average.

## 2. Split crossover depends strongly on batch

Means of three runs, microseconds for one layer, compact grids:

| Context | B1 direct | B1 split total | B8 direct | B8 split total |
| ---: | ---: | ---: | ---: | ---: |
| 59 | 12.112 | 13.524 | 13.145 | 20.299 |
| 128 | 24.178 | 15.078 | 26.201 | 32.125 |
| 256 | 47.123 | 15.375 | 50.983 | 56.093 |
| 512 | 92.855 | 15.645 | 100.577 | 105.185 |
| 1528 | 274.246 | 39.936 | 428.455 | 422.579 |
| 4096 | 731.718 | 100.985 | 1144.556 | 1101.665 |

B1 gains 6.87× at L1528 and 7.25× at L4096. B8 gains only 1.014× and
1.039× respectively, and loses at shorter contexts. The B8/L1528 ranges are
427.663–429.829 µs direct and 421.601–423.752 µs split: the small difference
is consistent across these three trials, but not an end-to-end improvement.

The current compiler-readonly selection already restricts split to batch <=4
and uses context bucket threshold 256; see
`engine/device_step/paged_decode_split_prepare.mbt:102` and `:123`.
Thus the big B1 direct→split number is **not a newly available serving gain**.
These measurements do not justify simply removing the B8 restriction.

### Merge and inactive grid capacity

| B/context | Compact partial | Compact merge | Compact total | Capacity-grid total |
| --- | ---: | ---: | ---: | ---: |
| 1/512 | 14.218 | 1.471 | 15.645 | 28.631 |
| 1/1528 | 38.473 | 1.471 | 39.936 | 51.389 |
| 8/512 | 102.926 | 2.345 | 105.185 | 111.655 |
| 8/1528 | 421.579 | 2.345 | 422.579 | 429.000 |

Compacting split grids removes measurable overhead, especially for B1, but
does not fix B8's inner-loop cost. At B8/L1528, removing all isolated merge
time would save only about 0.6% of split total. The direct B8 path already
uses a compact grid in the serving trace; do not credit that saving twice.

## 3. Counters distinguish insufficient parallelism from latency hiding

Separate Nsight Compute collection: B8, compact grid, L128 then L1528;
direct/partial/merge once per context. Kernel replay, 40 passes per kernel,
`--cache-control all --clock-control none`. These are cold-cache replay
counters, **not** warm Graph timings or full-serving counters. Six kernels
completed with the numerical oracle intact. Privileged profiling did not
change persistent driver permissions or clocks.

| Counter | L128 direct | L128 partial | L1528 direct | L1528 partial |
| --- | ---: | ---: | ---: | ---: |
| Blocks | 64 | 512 | 64 | 512 |
| Waves per SM | 0.89 | 7.11 | 0.89 | 7.11 |
| Achieved occupancy | 28.58% | 29.52% | 29.54% | 32.37% |
| Theoretical occupancy | 33.33% | 33.33% | 33.33% | 33.33% |
| No eligible warp cycles | 76.63% | 74.18% | 75.32% | 73.22% |
| Eligible warps/scheduler | 0.32 | 0.35 | 0.32 | 0.36 |
| DRAM throughput / peak | 24.30% | 22.57% | 26.88% | 27.65% |
| SM throughput / peak | 24.93% | 27.37% | 26.41% | 27.46% |
| Long-scoreboard share of average issue interval | 57.54% | 54.66% | 56.57% | 57.01% |

Both direct and partial reserve 34076 dynamic shared bytes per block.
Nsight reports a two-block/SM shared-memory limit, versus four blocks from
registers. More blocks increase waves, not resident resources beyond this
limit. At B1, direct has only eight blocks for 36 SMs, so split can fill idle
SMs; B8 already has 64 direct blocks. This explains why additional partition
parallelism has much less headroom there.

L1528 direct reads 50.126 MB from DRAM and partial 50.130 MB in this collection.
They do not eliminate K/V traffic by partitioning. No local spilling requests
are reported. Occupancy is a constraint, not a performance objective by itself:
other implementations can be faster at lower occupancy through better pipelines.
The low L2 hit rates here follow cache-flushed replay and must not be presented
as warm-serving L2 hit rates.

## 4. Instruction-level attribution: the K/V staging dependency

The SASS report attributes long-scoreboard samples to these waiting instructions
in direct L1528 (addresses from this specific process):

| Waiting instruction | Samples | Share of long-scoreboard samples |
| --- | ---: | ---: |
| `STS.U16 [R39-0x4000], R32` | 5544 | 22.70% |
| `STS.U16 [R39-0x3a00], R31` | 5524 | 22.62% |
| `STS.U16 [R39-0x3e00], R33` | 5509 | 22.56% |
| `STS.U16 [R39-0x3c00], R32` | 5482 | 22.45% |
| Total of these four sites | 22059 / 24419 | 90.34% |

The corresponding split-partial sites account for 20367 / 22597 = 90.13%.
Their producer chain contains `LDG.E.U16` K/V loads followed by shared stores.
The shared-store PC is where the warp waits for global-load results; it is
**not evidence that shared-memory bank conflicts caused that long-scoreboard
stall**. Nor is 90% a predicted removable fraction of wall time.

The generated implementation stages K and V synchronously into one 64-key
shared tile, waits, computes, waits, then stages the next tile. It does not
overlap the next K/V tile with current computation. The same structure exists
in `kernels/luna_cuda_attention_tile_source/source_grouped_split.mbt:204`.
This connects measured stalls to a specific compiler-emitted memory schedule,
rather than a generic assertion that attention is slow.

## 5. What is concretely different in SGLang / vLLM?

The previous same-workload serving traces still provide the three-engine
comparison. For short-input C8, average attention time per 28-layer forward
over the first and last 32 decode forwards was:

| Framework | First window, ms | Last window, ms | Growth, ms |
| --- | ---: | ---: | ---: |
| LunaFlux | 0.695 | 2.397 | 1.702 |
| vLLM | 0.367 | 0.857 | 0.490 |
| SGLang, decode + merge | 0.279 | 0.806 | 0.527 |

These are approximately aligned context windows, not identical per-token
alignment: vLLM/LunaFlux have 255 decode forwards, SGLang 256. They also grow;
their incremental cost is smaller. No new baseline throughput run was needed
or claimed for the isolated LunaFlux experiment.

For SGLang, we additionally inspected the **installed FlashInfer header from
the benchmark environment**, not an unrelated latest implementation. The
actual trace symbol instantiates `BatchDecodeWithPagedKVCacheKernel` with:
two shared stages, tile_size_per_bdx=1, vector size=8, bdx=16, bdy=2, bdz=4.
That is 128 threads and **128-bit BF16 copy vectors**. The installed
`attention/decode.cuh`:

- Loads Q into a thread-local vector before the traversal (line 469).
- Prepares reusable paged offsets (lines 477–487).
- Preloads K/V with predicated `cp_async` vectors (lines 490–513).
- Interleaves QK/state computation with future K/V loads and rotates the
  two shared stages (lines 535–582).

This directly contrasts with the synchronous scalar staging at LunaFlux's
measured stall sites. It is a strong optimization hypothesis, **not an isolated
proof that this single difference explains the complete framework ratio**.
We have not collected identical-input instruction counters for the baseline
kernels. The vLLM trace selects `flash_fwd_splitkv_kernel`; no claim about its
exact instruction-level stall distribution is made here.

## 6. Compiler work implied by the diagnosis

Keep attention as a pure mathematical fold/merge with explicit memory effects;
improve its physical schedule rather than adding model-specific branches.

1. **Vectorized tile transfer lowering:** prove alignment/contiguity and form
   wider loads; handle partial tiles with predicates. First compare this alone
   against the measured scalar staging, without changing softmax arithmetic.
2. **Software pipeline scheduling:** represent load readiness and buffer
   lifetimes explicitly; overlap next-tile transfer with the current fold.
   CUDA asynchronous-copy details remain in device lowering.
3. **Resource-aware storage selection:** jointly choose tile size, stages,
   register/shared residency and work distribution. Blindly doubling this
   34 KB buffer may reduce occupancy further; larger tiles are not always better.
4. **Shape-conditioned direct/partition choice:** benchmark batch, live context,
   head geometry and device. B1's win must not become a global split8 rule.
5. Afterwards measure Q residency, address reuse and alternative reduction
   schedules. Do not silently reassociate floating-point folds or turn on
   relaxed math as a supposed semantics-preserving optimization.

The immediate next A/B is vectorized staging, then pipelining, then their
combination with explicit resource measurements. This turn diagnoses the
existing compiler kernels; it does not implement or claim those improvements.

## Reproduction artifacts

- Remote/local raw results:
  `/dev/shm/lunaflux-attention-crossover-20260906-r1` on the NVIDIA host;
  `/private/tmp/lunaflux-attention-crossover-20260906-r1` locally.
- `summary.csv`: 3-repeat means/min/max; `counter-summary-v3.csv`: units retained;
  `counters-sass.csv` and `sass-top-stalls.txt`: exact instruction attribution.
- `harness-timing.cu` and `harness-counter.cu` preserve the two diagnostic
  harness versions. `flashinfer-decode.cuh` is the inspected installed header.
  Script orchestration/analysis is `.mbtx`; the installed async package lacks
  the skill's newer shell API, so scripts use its shell-free process API.
- Local archive: `/private/tmp/lunaflux-attention-deep-diagnosis-20260906-r1.tar.gz`.
  SHA-256: `faac84228d882ea567a78a68ea86151411c23b10863616796ddf9d18b855596a`.
- Nsight report SHA-256:
  `f8bdb84e095e3a5d3268fe54d74ffe5a068479e1a7a1603e8bcd968b3330fb59`;
  downloaded local hash matches. Final GPU process query is empty.
- No production files were rebuilt/replaced, no persistent profiling permission
  changed, and no production rollout performed. Numerical checks are specific
  to this fixture, not complete model-output equivalence or a new sanitizer run.
