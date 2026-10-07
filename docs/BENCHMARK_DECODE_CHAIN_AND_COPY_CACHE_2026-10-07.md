# Complete decode-chain attribution and copy-cache isolation

**The main C16 serving gap is not fixed.** Capturing the complete split chain
rules out its merge as the main target. Bypassing L1 for asynchronous operand
copies improves one long, low-batch ordinary-decode cell, but does not produce
a repeatable C16 win. No serving bundle, compiler default or production route
was changed.

This finite experiment follows the
[independent-dot experiment](BENCHMARK_DECODE_INTERLEAVED_QK_2026-10-07.md).
Its scope is one complete-chain attribution and one cache-policy ablation,
with matched counters, strict numerical checks and boundary/sanitizer tests.

## Fix the diagnostic before changing the kernel

The reusable collector now distinguishes ordinary decode from split decode.
Split profiling includes **partial and merge**, for both frozen control and
candidate. It checks the ordered kernel inventory in Nsight's raw launch CSV,
not repeated entries in the source/SASS export. A missing or reordered merge
fails the collector instead of yielding a misleading complete-chain result.
The collector also explicitly requests L2 read/write sectors.

The earlier chain capture ran the initial complete-chain collector; the final
evidence transaction independently verified its four raw launches. Both cache
captures ran the strengthened launch-verifying collector. Source snapshots are
retained to distinguish these versions.

The selected ordinary control is candidate 468,
`lunaflux_attention_decode_tile_compiler_v1_owned8_blockwise_f32_v4`. It uses
KV32, D128, GQA2, block 64, two independently refilled K/V slots and the explicit
`owned8-blockwise-f32-probability-v4` numerical law. Split partial 3903 has eight
partitions; merge 3904 completes the chain. The C16 launch geometry is 16 x 8
for ordinary decode, 16 x 8 x 8 for partial and 16 x 16 for merge.

## Split merge is not the main cost

Matched C16, history 4095, split-chain replay compares the frozen control with
the previously tested two-independent-dot candidate. These are profiler replay
times, not alternating unprofiled acceptance samples.

| Entry | Control, microseconds | Two-dot candidate, microseconds | Control warp instructions | Candidate warp instructions |
| --- | ---: | ---: | ---: | ---: |
| Partial | 1209.344 | 1207.328 | 43,503,616 | 43,454,464 |
| Merge | 12.928 | 12.224 | 166,144 | 166,144 |

Merge accounts for approximately **1.06%** of this control chain. Partial has
0.08 eligible warps per scheduler cycle in both captures. Its register count
rises from 146 to 153 without a material replay improvement. This confirms the
previous unprofiled non-win; it does not qualify the candidate.

The serving trace discussed in the
[route report](BENCHMARK_DECODE_ROUTE_AND_FAIRNESS_FIX_2026-10-06.md) is dominated
by **ordinary decode**, not this split chain. Optimizing a roughly 13-microsecond
merge cannot fix the main ordinary C16 cost. Chain attribution and actual
serving launch frequency are separate measurements and must both guide priority.

## One cache-policy change, no arithmetic change

The offline diagnostic changes only the 16-byte operand-copy policy:
`cp.async.ca` becomes `cp.async.cg`. Both ordinary and split-partial entries are
transformed; merge is unchanged. All 32 copy sites are checked, 16 per entry.
The key/component order, shuffle tree, softmax, PV fold, page addressing,
layout, stage lifetimes, synchronization and launch reservation remain fixed.
No FMA, fast-math or numerical-tolerance change is allowed.

An initial preparation assertion incorrectly expected four copy sites. It
failed before compilation or GPU execution and is preserved separately. A new,
non-overwriting root contains the corrected 32-site transform. Failed evidence
is not relabeled as a completed GPU run.

Five alternating pairs per cell produce these results. Gains are **median
paired completion-time gains**; a positive value means faster. Acceptance
requires every pair to gain at least 1%. Independent medians are not used to
override a paired non-win.

| Useful rows / envelope / history | Chain | Control median, microseconds | Bypass median, microseconds | Median paired gain | Minimum paired gain | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 16 / 16 / 4095 | Ordinary | 1160.768 | 1153.039 | +1.346% | -1.528% | Inconclusive |
| 2 / 8 / 32767 | Ordinary | 1430.237 | 1384.142 | **+4.638%** | **+3.202%** | Improved |
| 1 / 1 / 127 | Ordinary | 8.228 | 8.227 | +0.013% | -0.078% | Inconclusive |
| 16 / 16 / 4095 | Partial + merge | 1180.000 | 1169.292 | -0.171% | -1.348% | Regression |
| 2 / 8 / 32767 | Partial + merge | 1195.429 | 1161.463 | +3.647% | +0.590% | Inconclusive |
| 1 / 1 / 127 | Partial + merge | 9.981 | 9.987 | +0.074% | -1.418% | Inconclusive |

All **30** samples passed bitwise equality, the independent scalar oracle and
KV integrity checks. The accepted ordinary long-C2 cell does not justify a
global default: the existing split control is already faster in that workload,
and the split bypass candidate fails the robust acceptance rule.

## C16 waits move rather than disappear

Matched ordinary C16 replay proves that bypass instructions actually execute.
The control executes 524,288 `LDGSTS.E.128` operations plus 8,192 zero-fill
operations. The candidate executes the corresponding
`LDGSTS.E.BYPASS.128` operations, with the same counts.

| Counter | Control `.ca` | Bypass `.cg` |
| --- | ---: | ---: |
| Replay duration, microseconds | 1208.320 | 1183.744 |
| Warp instructions | 41,500,160 | 41,500,160 |
| Registers per thread | 148 | 148 |
| L2 read sectors | 8,406,216 | 8,397,226 |
| L2 write sectors | 12,492 | 10,590 |
| L2 throughput, percent | 19.36 | 19.78 |
| Eligible warps per scheduler cycle | 0.09 | 0.09 |
| Issue active, percent | 8.79 | 8.93 |
| Short-scoreboard cycles per active issue | 3.85 | 0.57 |
| Long-scoreboard cycles per active issue | 3.02 | 7.31 |
| MIO-throttle cycles per active issue | 2.21 | 0.01 |
| Barrier cycles per active issue | 0.38 | 1.90 |
| Fixed-wait cycles per active issue | 0.64 | 0.64 |

Instruction count and L2 read demand barely change. Short shared-load waits
and MIO throttling fall, but global-copy completion and synchronization waits
increase. Eligible issue remains unchanged. The bypass capture's hottest
sampled instruction is `BAR.SYNC.DEFER_BLOCKING`, with 51,057 samples, of which
51,017 are classified as load dependency. This is evidence of a changed wait
location, not evidence that removing the barrier is safe or beneficial.

Stall ratios are cycles per active issue, **not additive wall-time shares**.
L2 sectors describe 32-byte requests, not bytes necessarily read from DRAM.
The GB10b profiler does not expose the requested DRAM-byte counter; its absence
must not be reported as zero traffic or used to claim DRAM saturation.

## Long, low-batch behavior differs

The long-C2 ordinary replay is 1593.824 → 1502.592 microseconds. Its unprofiled
paired gain is 4.638%, not the larger replay ratio. Instructions remain
41,295,520 and registers remain 148. L2 read sectors are 8,398,303 → 8,397,315.
Eligible warps are 0.36 → 0.35 and issue active is 36.20% → 35.11%.

Short/MIO waits fall slightly, long/barrier waits increase slightly, and the
unprofiled cell still improves. This demonstrates workload dependence. It does
not establish a single stall metric as the causal explanation for every shape,
nor establish a serving speedup on a route that was not tested end to end.

## Validation and decision

All **22 boundary cases** passed for ordinary and split chains, with ragged
useful rows inside envelope eight and histories 0, 6, 7, 8, 30, 31, 32, 63, 64,
66 and 4096. All **18 sanitizer cases** passed: memcheck with leak checking,
racecheck and synccheck on both chains at histories 0, 32 and 4096. Compilation
reports no spills or stack frames. The four owned automation files passed
warning-denied native tests, **5/5** in total.

GPU work was serialized on Spark 179, UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, with pinned CUDA 13.0.88. User jobs
and counter containers were bounded to 8 GiB, with no swap and a 900-second
runtime ceiling. The minimum sampled `MemAvailable` across the three counter
captures was 120,625,128 KiB, above the 32 GiB reserve. Final GPU compute-process
inventory was empty. Serving processes and unrelated work were untouched.

Keep bypass as a narrow qualified diagnostic, not a serving route or compiler
default. The main next hypothesis is a **producer/consumer readiness schedule**
that overlaps operand-copy completion with useful work while preserving slot
retirement, shared-reader publication and the ordered numerical fold. This
requires an ownership/effect plan and executable lowering, not removal of
synchronization by inspection. This experiment does not yet prove that such a
schedule wins.

Last verified serving remains **225.600 output tokens/s**, 4096-input/64-output
C16. Historical matched vLLM/SGLang controls are 243.03/241.85; they were not
rerun here. The end-to-end parity gate remains open. The broader **“fix all”**
request is incomplete; no new whole-engine improvement is claimed.

## Evidence and reproduction

Owned automation:

- `benchmarks/gpu_pipeline/decode_consumer_window_20261007.mbtx`: complete split-chain capture, launch verification and explicit L2 metrics.
- `benchmarks/gpu_pipeline/decode_register_exchange_gates_20261007.mbtx`: validated one- or two-variant boundary/sanitizer campaigns.
- `benchmarks/gpu_pipeline/decode_copy_cache_20261007.mbtx`: finite cache-policy experiment.
- `benchmarks/gpu_pipeline/finish_decode_copy_cache_20261007.mbtx`: terminal verification and non-overwriting evidence seal.

Frozen control source SHA-256:
`dc9f5a1a30e0d2f969f7a9e82cdcc7b8b9a8c80f2058c698b19ea9416e9d43b6`.
Frozen CUBIN SHA-256:
`318d74ff5b8ebfb5e0a97ea6daea16501a98a5ef1b67116350cf8b737266c185`.
The two-dot chain candidate remains the previously sealed artifact referenced
by the preceding experiment, not a newly rebuilt control.

| Evidence | Immutable remote archive | SHA-256 | Verified manifest entries |
| --- | --- | --- | ---: |
| Copy policy | `/home/wlc004s/lunaflux-decode-copy-cache-v2-20261007.jpdNuZXR.verified.tar.gz` | `9aaf2c532a9de96fd85c9a8cd71f0520ead457ae72b748f8e57992fc01802858` | 322 |
| Split chain | `/home/wlc004s/lunaflux-decode-chain-counters-20261007.5lBX7ksU.verified.tar.gz` | `c02b3c93062503085b37053cfb3ee8412fa2cf8bb5a6b22a8cddb92a8b5f9c29` | 33 |

Both archives were downloaded without overwrite to
`/tmp/lunaflux-decode-cache-evidence-20261007.gHA1EFj5`, their archive hashes
matched locally, and all **355 manifest entries** verified after extraction.
Raw paired outputs, counter reports, SASS, source/CUBINs, sanitizer outputs,
commands and failed preparation are retained. No prior evidence was deleted.
