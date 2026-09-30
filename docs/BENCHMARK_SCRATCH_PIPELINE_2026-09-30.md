# Scratch lifetime reuse and decode readiness pipelines

This follow-up implements the source-level diagnosis rather than adding another
IR layer or forcing a new kernel family to win. End-to-end and fresh pinned
reference measurements are recorded below when their terminal runs complete.

## Changes

1. **Projection geometry:** an immutable `SequentialScratch` plan distinguishes
   overlapping from disjoint operand-staging and epilogue lifetimes. Disjoint
   storage uses `max(producer, consumer)`, not their sum. CUDA lowering retires
   all operand readers before reusing the arena. Multiple column windows retain
   separate storage because the lifetimes genuinely overlap. This admits larger
   row tiles without weakening the shared-memory ceiling or changing arithmetic.
2. **Decode readiness:** two-stage blockwise candidates express K readiness,
   probability production, V readiness and reader retirement as effects. The
   lowering consumes that plan instead of synchronously staging the entire tile.
3. **History-axis parallelism:** blockwise decode now has an explicit partial
   map and F32 `(max, denominator, numerator)` merge ABI, including empty
   partitions and invalid metadata. It is independently exported and probed.
   This new numerical family is **not yet a production runtime split route**.
4. **Selection:** bounded offline measurements compare all legal finalists,
   including the existing schedules. Typed canonical exports are used for the
   runtime entry points; diagnostic symbols cannot be substituted into a serving
   bundle. Candidates absent under the resource cap are skipped, not fabricated.

The semantic and physical plans are immutable MoonBit values. GPU writes,
barriers and asynchronous readiness remain explicit lowering effects. No model
builder or scheduler imports CUDA-specific scheduling decisions. These changes
are general compiler mechanisms exercised with a Qwen3-0.6B workload; they do
not establish equal benefit across all models and devices.

Kernel source snapshot: `466ede5c`, archive SHA-256
`1882919d5c6757c39c929dde146df3fbd2fcc786be84be022bdb36759558089c`.
Calibration availability fix: `9d4ea9a3`. Canonical export and isolated campaign
packaging fixes: `8cfb7fc1`, `0fa441b0`, `0d5eee11`. Spark-only target substitutions
are sm121/CUDA 13.0.88; unrelated working-tree changes are excluded.

## Calibration

Spark GB10, 48 SMs, UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.
Five paired unprofiled samples per candidate/cell; runtime-bucket geometry.
Ten cells cover short/long prefill, 1/8/16 decode rows and 127/4095 history.
Resources and candidate source identities accompany the workload-scoped records.

| Operation / cell | Candidate | Median GPU time |
| --- | --- | ---: |
| Full ingress, query2048/rows8/history2048 | row32, K32, two stages | 508.277 µs |
| Same | row64, K32, two stages | 500.274 µs |
| Same | row16, head2, K16 | 1075.380 µs |
| Decode, query16/rows16/history4095 | existing c441 | 1262.948 µs |
| Same | blockwise sync c450 | 1783.399 µs |
| Same | blockwise async c452 | 1639.102 µs |

In the paired C1/history4095 probe, c452 improves blockwise c450 from
1005.899 to 724.478 µs (28.0%). It does not beat every existing decode schedule.
Its reported local allocation is 192 bytes/thread; register and local-memory
cost must not be hidden behind the asynchronous-copy label. Candidate c453 is
absent from the 48 KiB serving frontier, not a failed numerical experiment.

The selected serving package therefore uses row64/K32 ingress, async c322
prefill and c441 decode. It is not a forced-new-family experiment. Automatic
per-cell runtime dispatch is still absent: measurements across ten cells do not
imply ten live routes.

## Validation and remaining scope

The affected package tests pass 194/194. The exact packaged ingress, prefill
and decode cubins pass memcheck (including leaks), racecheck and synccheck.
These numerical probes are not a broad model-quality evaluation.

Failed isolated setup attempts are preserved separately: diagnostic decode
symbol substitution, missing probe source, then missing geometry header. None
reached model serving, and none is relabeled as a performance result.

Primary run directory:
`/home/wlc004s/lunaflux-scratch-pipeline-v2-20260930.3g24CVOM`.
Selected package: `selection-v2`; serving: `serving-v4`; completed calibration:
`calibration-final`. GPU workloads are serialized. LunaFlux runs have a 64 GiB
unit ceiling, no swap growth and a 32 GiB available-memory reserve. Pinned
reference containers have an 80 GiB ceiling and no swap growth, plus the same
reserve monitor; their limits are not claimed to be identical to LunaFlux's.

## End-to-end LunaFlux result

`serving-v4` completed all nine cells, acknowledged drain, exited the worker
with code zero and closed it. Runtime stderr is empty. Same Qwen3-0.6B BF16
model and token-ID workload as the previous selected run: one warm-up, three
measured trials, medians; greedy, ignore EOS, prefix reuse disabled.

| Input/output | C | Previous fastest wall ms | Current wall ms | Current output tok/s |
| --- | ---: | ---: | ---: | ---: |
| 128/32 | 1 | 246 | 246 | 130.08 |
| 128/32 | 8 | 322 | 321 | 797.51 |
| 128/32 | 16 | 388 | 383 | 1336.81 |
| 4096/64 | 1 | 747 | 738 | 86.72 |
| 4096/64 | 8 | 2845 | 2752 | 186.05 |
| 4096/64 | 16 | 5376 | 5177 | 197.80 |
| 4096/256 | 1 | 2530 | 2529 | 101.23 |
| 4096/256 | 8 | 7707 | 7625 | 268.59 |
| 4096/256 | 16 | 14261 | 14083 | 290.85 |

The largest throughput gain against the previous fastest package is 3.8% at
4096/64 C16; 4096/256 C16 gains 1.3%. The comparator is historical, not a
contemporaneous old/new serving A/B. This is a modest improvement, not parity
or an explanation that all remaining time is projection overhead. In particular,
the row64 versus current row32 calibration differs by only 1.6%.

TTFT p50/p95 at 4096/64 C16 is 1469/2601 ms; ITL is 45/162 ms. At
4096/256 C16 these are 1482/2632 ms and 46/48 ms. The selected package restores
the prior split-decode ABI rather than treating a faster unsplit blockwise
probe as proof of an equally fast serving graph.

## Fresh pinned reference comparison

vLLM and SGLang completed the identical nine-cell token workload sequentially,
after LunaFlux drained. The existing stopped benchmark containers were started,
inspected and stopped again; no image or framework configuration was changed.
These are pinned NVIDIA 26.01 reference configurations from the preceding
campaign (vLLM v0.13.0 build / FlashAttention; SGLang / FlashInfer), not a claim
to compare every current framework release or optimal configuration.

Completion medians, milliseconds, lower is better:

| Input/output | C | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: | ---: |
| 128/32 | 1 | 246 | 298 | 288 |
| 128/32 | 8 | 321 | 284 | 280 |
| 128/32 | 16 | 383 | 322 | 329 |
| 4096/64 | 1 | 738 | 772 | 785 |
| 4096/64 | 8 | 2752 | 2271 | 2334 |
| 4096/64 | 16 | 5177 | 4210 | 4251 |
| 4096/256 | 1 | 2529 | 2712 | 2771 |
| 4096/256 | 8 | 7625 | 6645 | 6855 |
| 4096/256 | 16 | 14083 | 11840 | 11980 |

LunaFlux is faster at C1 in these cells but takes 23.0%/21.8% more time at
4096/64 C16 than vLLM/SGLang. At 4096/256 C16 the remaining time gap is
18.9%/17.6%. This improvement does **not** close the concurrency gap.

All 225 measured sequences match the previous fastest LunaFlux package exactly.
Against each fresh framework reference, 216/225 sequences match exactly and
225/225 first tokens match; all 150 long-input sequences match. Short-input
greedy divergences remain and are not silently described as numerical identity.

## Partitioned blockwise experiment

Twelve sync/async, KV32/KV64, 2/4/8-partition variants passed five paired trials
in each of six cells. Timings include partial and merge launches. The baseline
here is **unsplit c441**, not the existing production split-decode chain.

| Rows | History | Fastest variant | c441 unsplit µs | Partial+merge µs |
| ---: | ---: | --- | ---: | ---: |
| 1 | 127 | c450-p4 | 14.358 | 14.443 |
| 8 | 127 | c450-p4 | 16.419 | 24.554 |
| 16 | 127 | c450-p4 | 28.438 | 36.700 |
| 1 | 4095 | c452-p8 | 320.124 | 106.662 |
| 8 | 4095 | c452-p4 | 594.932 | 690.954 |
| 16 | 4095 | c452-p2 | 1281.002 | 1291.858 |

History-axis parallelism gives a real 3.0× C1 improvement against this unsplit
probe, but loses at C8/C16. It must not become an unconditional runtime path.
Production already has a split-decode path at small batch: the 3.0× result is
**not an incremental serving speedup**. Binding this F32 blockwise ABI into the
runtime and comparing it with that existing full chain remain separate work.
All six per-cell finalists also pass memcheck with leak checking, racecheck
and synccheck (18 checks); no memory, race or synchronization errors are reported.

## Paired selected-kernel counters

`counters-v3` compares the prior fastest packaged ingress/prefill modules with
the current package on query2048/rows8/history2048. Exactly two matching kernel
launches are captured per operation. Profiler timings are not substituted into
the serving table. The prior run did not retain a standalone paired decode
module; no prior decode artifact was invented.

| Ingress metric | Prior fastest | Current |
| --- | ---: | ---: |
| CTAs | 2048 | 1024 |
| Profiled GPU time | 893.824 µs | 682.496 µs |
| Warp instructions | 82,608,128 | 67,588,096 |
| Registers/thread | 86 | 117 |
| Tensor activity | 18.33% | 24.09% |
| Long-scoreboard metric | 30.54% | 25.95% |
| Barrier metric | 4.36% | 5.49% |
| Local load/store sectors | 0/0 | 0/0 |

These stall metrics are per-warp-active percentages, **not** the previous
source-correlated average warp-latency contributions. They are not interchangeable.
Requested DRAM byte counters are absent from this GB10 raw capture and are not
reported as zero or inferred from another column.

Current prefill remains approximately the same: 94,723,968 → 95,081,344 warp
instructions, 809.824 → 825.152 µs profiled time, 43.75% → 42.98% tensor
activity, zero local load/store sectors. A single profiled replay cannot establish
a 1.9% regression; it does show no large reduction in supporting instructions.

Thus ingress improved, but unchanged attention/MLP/head execution and the
concurrent serving graph still dominate the remaining gap. More IR layers alone
will not remove it. The schedule space still needs geometry independent of
headwise epilogue ownership, a lower-overhead attention realization, and
workload-specific full-chain selection. No parity or production promotion is
claimed.

## Saved results

Minimum sampled available memory: LunaFlux 104,594,888 KiB (99.75 GiB),
vLLM 56,276,836 KiB (53.67 GiB), SGLang 57,110,668 KiB (54.46 GiB).
All exceed the 32 GiB reserve. No OOM occurred; no benchmark GPU process
remained after completion. Failed profiler setup attempts (expired sudo cache
and missing standalone prior decode artifact) remain separately preserved.

Completed calibration, selections, packaged kernels, raw serving trials,
container inspections, partition probes/sanitizers and paired Nsight reports
are archived without model weights or build caches. Archive SHA-256:
`834c105f6fd7b63d16b42d5c8d01ce4ce8593558de204b868ccd67df15daf84a`.
