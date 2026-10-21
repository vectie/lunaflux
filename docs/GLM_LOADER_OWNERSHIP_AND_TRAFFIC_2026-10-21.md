# Loader ownership and projection traffic: bounded follow-up

## Result

**The loading failure and fresh paired vLLM trace are not fixed. The two
projection candidates reduce traffic, but regress latency and are withdrawn.**
The accepted GLM service is restored, not replaced by either candidate.
No overall benchmark, payload hashing, reserve relaxation, changed reference,
numerical tolerance waiver or unproven production selection was performed.

This round's declared budget was two loading trials and two projection
candidates. All four are preserved. The negative results narrow the diagnosis;
they are not relabelled as a performance improvement.

## Installed source, not a presumed implementation

The generic loader in `../vlm/vllm/model_executor/model_loader/weight_utils.py`
and the installed source both yield a clone from the local Runai iterator.
The streamer's `memory_limit=0` allocates a buffer for the largest tensor across
the requested files; it does not bound clones, driver memory or other processes.

The deployed image also contains `vllm/models/glm5next/nvidia/model.py`, which
is absent from the local checkout. Its FP8 indexer/projection helpers can retain
weight/scale pairs until both arrive. Blindly deleting the clone would therefore
be unsafe for tensors backed by a reusable reader buffer. These helpers' presence
does **not** establish that they caused this NVFP4 run's memory failure.
The language-only vision wrapper uses the generic meta-device/no-init context;
its vision-attention initialization log is not proof of resident vision weights.

Installed model snapshot:
`/tmp/lunaflux-glm-installed-trace-sources-20261017-v2/models-glm5next-nvidia-model.py`.
Fresh installed loader and instrumented copy:
`/tmp/lunaflux-loader-observer-20261021-v1/{original.py,weight_utils.py}`.

## Loading ownership observations

`scripts/prepare-glm-loader-observer.mbtx` prepares an external-reference-only
observer. It preserves the cloned payload and consumer, records process
`smaps_rollup`, RSS categories and PyTorch allocated/reserved device bytes,
and retains only weak references to cloned tensor objects of at least 1 MiB.
It samples every 128 yields and on tensors of at least 64 MiB. A read-only mount
in the benchmark launcher selects it explicitly. No compiler/runtime import,
GPU synchronization or tensor-content scan is added.

The two model configurations remain identical: cached installed vLLM image,
NVFP4, TP2, context8192, eight sequences, batch64, KV256MiB, graph maximum8,
local Runai concurrency1/largest-tensor buffer. Container caps remain96GiB
memory and96GiB memory+swap. The device ceiling is100GiB/Spark, host floor16GiB,
with the existing18GiB early-retirement guard unchanged.

| Trial | Host minimum R0/R1, GiB | GPU maximum R0/R1, GiB | Outcome |
| --- | ---: | ---: | --- |
| Loader ownership observer | 17.983 /18.979 | 89.628 /89.628 | Early guard; pair retired |
| Same + defer CUPTI initialization | 17.980 /19.310 | 90.005 /90.005 | Early guard; pair retired |

These are sampled envelopes, not continuous physical peaks. No observed sample
crosses16GiB. Both configurations disallow additional swap. Exit137 follows
our requested Docker retirement and is not an OOM diagnosis.

Roots:

- `/tmp/lunaflux-reference-bench-glm-loader-ownership-20261021-v1`
- `/tmp/lunaflux-reference-bench-glm-loader-deferred-20261021-v1`

At rank0's identical observed tensor frontier, index4352,
`model.language_model.layers.39.mlp.experts.217.down_proj.weight_scale`:

| Worker observation, KiB unless bytes stated | Ordinary initialization | Deferred initialization |
| --- | ---: | ---: |
| PSS anonymous | 2,923,888 | 2,853,612 |
| PSS shared-memory mappings | 2,528,976 | 2,528,976 |
| PSS total | 5,777,637 | 5,704,009 |
| PyTorch GPU allocated, bytes | 94,996,518,912 | 94,996,518,912 |
| PyTorch GPU reserved, bytes | 95,017,762,816 | 95,017,762,816 |
| Observed large clone objects surviving more than two yields | 0 | 0 |

In the first trial, anonymous PSS rises from1,612,684KiB at index0 to
2,923,888KiB at index4352. This is not explained by an observed growing set of
those large cloned tensor **objects**. The weak-reference observation does not
cover smaller tensors, unobserved intervals, storage aliases, reader buffers or
allocator-retained pages. It is not a complete allocation attribution.
Shared-memory PSS is not assumed to be POSIX `/dev/shm` or added to GPU usage;
mapping-level ownership remains to be measured. Do not sum overlapping RSS/PSS
and device accounting into a purported physical total.

The deferred control performs no CUPTI activity initialization before `start`.
Its bounded fixture first verified two windows, all16 kernels/eight pre-existing
graph nodes, and zero dropped/invalid/error/refused records despite `_exit`.
Root: `/tmp/lunaflux-activity-deferred-fixture-20261021-v1`.
Both full-model trials fail **before readiness**, so this control does not
provide a full-model execution trace or prove zero observer overhead. It does
show that deferring initialization is insufficient to clear the loading failure.
The unsuccessful tracer change is archived in
`benchmarks/experiments/glm_deferred_activity_20261021.patch`; the accepted
collector remains unchanged.

## Projection experiment: reduce reads without a larger shared allocation

Target: actual sparse-output geometry M64/N4096/K8192, BF16 operands/publication,
unchanged ascending K16 F32 matrix fold, runtime grid1024/block128. The baseline
is the accepted M32/N64/K64 two-slot swizzled fragment pipeline. Candidate1 uses
M32/N128/K32; candidate2 uses M64/N128/K32. There is no split-K or partial-output
reduction. The pure `MatrixFragmentPipeline` owns geometry/resource requirements;
private lowering consumes them. Explicit offline export exercises the plans;
neither is selected by the serving compiler.

For these full rows, logical operand traffic is
`2*K*(M*ceil(N/tile_N) + N*ceil(M/tile_M))` bytes:
192MiB baseline,160MiB candidate1,96MiB candidate2. Shared allocations are
24/20/24KiB. This avoids the previous K128 experiment's doubled shared footprint,
but raises register demand, reduces useful output CTAs128→64→32, and doubles
operand publication epochs128→256. Those are real competing costs, not free reuse.

Export root:
`/tmp/lunaflux-glm-projection-staging-column-reuse-export-20261021-v1`.
Run/counter root:
`/tmp/lunaflux-glm-projection-staging-column-reuse-20261021-v1`.

Seed713, five alternating pairs,20 graph replays per pair. Both candidates match
all active/inactive output bytes on live rows0/1/3/22/36/64/65 and preserve the
existing sampled F64 oracle gate. This is bounded fixture correctness, not an
all-input or full-model proof. No sanitizer/serving qualification is claimed for
these rejected candidates.

| Rows | Baseline→candidate1 median, us | Baseline→candidate2 median, us |
| ---: | ---: | ---: |
| 1 | 294.440→292.906; inconclusive | 300.360→304.562; regression |
| 3 | 306.613→368.643 | 305.166→361.354 |
| 22 | 309.894→363.949 | 310.139→367.691 |
| 36 | 310.486→368.546 | 314.030→386.488 |
| 64 | 323.224→381.856 | 314.294→389.150 |

At64 rows, ratios of medians regress18.14% and23.82%. Median paired regressions
are20.50% and18.32%, respectively; these are different statistics. Row1 uses the
unchanged decode branch, so its differences do not establish a decode mechanism.

### Same-campaign counters

| Metric | Baseline | Candidate1 | Candidate2 |
| --- | ---: | ---: | ---: |
| Profile duration, us | 366.880 | 453.632 | 459.360 |
| Warp instructions | 17,119,232 | 14,803,712 | 10,237,440 |
| L2 read sectors | 6,294,228 | 5,246,489 | 3,149,817 |
| Registers/thread | 46 | 69 | 119 |
| Shared allocation, KiB | 24 | 20 | 24 |
| Active warps, % peak | 22.19 | 11.63 | 9.00 |
| Tensor activity, % peak | 10.23 | 8.05 | 11.71 |
| Long-scoreboard share, % | 62.43 | 56.26 | 47.12 |
| Barrier share, % | 22.54 | 30.48 | 24.49 |
| Short-scoreboard share, % | 1.17 | 1.41 | 3.81 |

Profile durations are not unprofiled graph medians. Stall shares are active-cycle
ratios, not additive wall-time allocations. The measured traffic reductions agree
with the ownership calculation, but neither buys lower completion time.
Candidate2 halves L2 reads and removes40.2% of instructions, while the active-warp
measure falls and dependency/barrier costs remain substantial. This experiment
changes ownership and epoch size together; it does not assign an exclusive
causal percentage to occupancy, registers, barriers or underfilled SMs.

The retained installed vLLM entry
`nvjet_sm121_tst_mma_128x64x64_4_32x64x64_tmaAB_bz_TNNN` on identical fixture
operands has3,149,645 L2 read sectors and2,293,848 warp instructions. Candidate2
therefore reaches essentially the same traffic but still executes4.46× the
instructions. Its47.12% load-dependency/24.49% barrier shares are not comparable
to a wall-time budget; the reference's retained shares are1.316%/.061%.
This is a historical matched **leaf** comparison, not a fresh full-model trace.

Installed dispatch routes unquantized linear through PyTorch/cuBLAS. The actual
nvjet implementation is not readable source in `../vlm`; retained SASS shows
uniform TMA coordinate handling, swizzled matrix loads, reused fragments,
load/MMA interleaving and phase waits. Our candidate still uses per-thread
`cp.async` address/copy work and workgroup retirement. Matching traffic does not
match that execution schedule. The library's256-CTA launch alone still does not
prove its useful tile count or a split-K factor.

Both unsuccessful compiler/lowering changes have been removed from executable
packages, with their patch preserved in
`benchmarks/experiments/glm_column_reuse_20261021.patch` against`aab5a489`.
No new artifact or slower path was deployed. The AKO budget ends this physical
campaign rather than allowing an unbounded sequence of unqualified variants.

## Validation, restoration and remaining blocker

Retained changes are external loader observability only. Launcher tests10/10,
observer test1/1 and reserve-parser test1/1 pass. After withdrawing the projection
candidates, grouped-matrix tests21/21 and tiled-source tests16/16 pass with the
existing migration-warning exclusions. Native check also passes with those same
exclusions; this is not a new warning-free full-release-suite claim. Both archived patches pass a read-only
`git apply --check`. No unrelated dirty migration files were staged or changed.

Restoration evidence:
`/tmp/lunaflux-glm-loader-ownership-restoration-20261021-v1`.
The accepted `ed6be466` units and `b1788d1d` projection artifact are restored.
Private health is healthy and the complete arithmetic response is`323`, with
`finish_reason=stop`. This is restoration verification, not a public-key test or
broader quality/throughput claim. Sampled GPU maxima are94.239/94.268GiB and host
minima17.031/18.981GiB, ranks0/1 respectively. Owned swap remains zero. The
restoration observer has retired after saving terminal units, health and answer.

Still required: identify actual allocation/mapping ownership behind the loader's
resident reader/host footprint, establish safe vLLM readiness, and capture both
reference ranks. For projection, a successful transfer/issue schedule must reduce
per-thread transport overhead without losing useful parallelism; reduced traffic
alone has now been directly falsified as a sufficient fix. There is **no new
full-model time/overlap/TP-wait graph** to substitute for the retained reports.
