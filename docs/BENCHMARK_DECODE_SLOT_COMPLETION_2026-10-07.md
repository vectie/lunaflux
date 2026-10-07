# Decode slot completion and polling overhead

The serving performance gap remains open. Separating slot readiness from
reader release removes most CTA barrier waits in this diagnostic, but does
not produce an accepted timing improvement. Repeated phase checks increase
control and memory-pipeline pressure. Racecheck also reports a copy-to-shared
hazard, so this candidate is blocked from production even though the numerical
checks pass.

## Frozen workload and numerical contract

This follows the [dedicated-copy experiment](BENCHMARK_DECODE_COPY_ROLES_AND_PROBE_REPAIR_2026-10-07.md).
The finite budget was one structural candidate, six alternating paired cells,
22 boundary cases, an 18-case sanitizer gate and one C16 counter pair. The
sanitizer gate stopped on its second case; it did not complete all 18 cases.

The ordinary control is
`lunaflux_attention_decode_tile_compiler_v1_owned8_blockwise_f32_v4`, with
GQA2, KV32, D128, block 64, two independent K/V slots and 33,040 shared bytes.
Partial 3903 has eight partitions; merge 3904 completes its chain. The
`owned8-blockwise-f32-probability-v4` law and strict arithmetic remain fixed.
The repaired probe launches partial and merge with independent geometry.

The candidate retains two consumer and two producer warps, block 128 and
merge block 64. Eight 64-bit barriers occupy an additional 64 shared bytes:
K ready, V ready, K released and V released for each slot. Producers wait for
the preceding reader release before slot reuse. Consumers release K after QK
and V after PV. Initialization and final join retain whole-CTA barriers.

The private immutable `SlotEpoch` plan describes slot, generation and preceding
release; its tests cover 1–16 slots through 1,024 epochs. PTX is restricted to
the offline CUDA diagnostic renderer. This is not a production compiler pass
or generic generated-kernel coverage: rendering is pinned to one exact module.
No numerical reassociation, FMA substitution, tolerance change, KV layout
change, cache-policy change or serving dispatch change is included.

## Paired timing results

Five alternating pairs per cell require bitwise equality, independent scalar
oracle checks and KV integrity. Acceptance requires every pair to improve by
at least 1%. Positive paired gain means faster completion. Independent medians
cannot override paired decisions.

| Useful rows / envelope / history | Chain | Control median µs | Slot median µs | Median paired gain | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| 16 / 16 / 4095 | Ordinary | 1166.096 | 1165.396 | -0.238% | Regression |
| 2 / 8 / 32767 | Ordinary | 1448.317 | 1636.005 | -12.629% | Regression |
| 1 / 1 / 127 | Ordinary | 8.225 | 8.217 | +0.091% | Inconclusive |
| 16 / 16 / 4095 | Partial and merge | 1167.037 | 1184.788 | -1.521% | Regression |
| 2 / 8 / 32767 | Partial and merge | 1193.431 | 1176.145 | +1.791% | Inconclusive |
| 1 / 1 / 127 | Partial and merge | 10.124 | 10.262 | -1.391% | Regression |

All 30 timing pairs and all 22 numerical boundary cases pass with bitwise
equality and maximum differential zero. The long split cell has a positive
median but includes a negative pair; it is not an accepted winner.

## Barriers decrease but phase checks add work

The matched C16 capture contains exactly two ordinary launches, grid 16 × 8,
control block 64 and candidate block 128. Replay durations are diagnostic;
unprofiled paired samples above determine acceptance.

| Metric | Control | Slot completion |
| --- | ---: | ---: |
| Replay duration µs | 1199.360 | 1190.336 |
| Registers per thread | 148 | 110 |
| Active warps percent of peak | 7.99 | 16.52 |
| Eligible warps per scheduler cycle | 0.09 | 0.14 |
| Issue active percent of active cycles | 8.86 | 11.72 |
| Hardware warp instructions `smsp__inst_executed.sum` | 41,500,160 | 54,065,752 |
| Source execution count `inst_executed` | 41,500,160 | 85,194,967 |
| L2 read sectors | 8,398,446 | 8,398,187 |
| Long scoreboard per active issue | 3.07 | 3.36 |
| Short scoreboard per active issue | 3.80 | 4.53 |
| Barrier per active issue | 0.39 | 0.03 |
| MIO throttle per active issue | 2.18 | 3.39 |
| Branch resolving per active issue | 0.02 | 2.38 |
| Source excessive shared wavefronts | 0 | 0 |

Hardware instructions increase about 30.3%. The source execution count grows
more and differs from the hardware count: phase polling varies across profiler
replays and collection methods. These values must not be presented as the
same-pass instruction total or added together.

Within the source-counter capture, `SYNCS.PHASECHK.TRANS64.TRYWAIT` executes
13,377,399 times and `BRA` 13,910,761 times. The frozen control executes
299,776 `BRA` instructions. CTA barrier instructions fall from 131,584 to
1,024. Mathematical work remains unchanged: `FADD` 9,207,296, `FMUL` 8,945,408;
128-bit copies remain 524,288. The candidate omits speculative tail copies,
but this does not overcome the waiting overhead.

The hottest sampled sites include phase checks and their retry branches.
One retry branch has 6,213 load-dependency samples out of 6,608 samples.
That identifies a waiting site, not an independently proven producer or a
wall-time percentage. Stall ratios are not additive elapsed-time shares.
L2 sectors are requests, not necessarily DRAM bytes. Missing DRAM-byte data
is unavailable, not zero traffic. Neither compilation nor replay reports spills.

**The supported conclusion is narrower than “barriers were the entire gap.”**
Explicit readiness improves one counter but replaces inexpensive coordinated
waiting with substantial phase/control work in this lowering. Lower registers,
higher occupancy and fewer barriers are insufficient to establish a speedup.

## Sanitizer blocks qualification

History-zero ordinary memcheck, including leak checking, passes. The following
racecheck case reports an asynchronous shared write at kernel offset `0x1fc0`
against reads beginning at `0x3260`. Disassembly identifies the write as
`LDGSTS.E.128` and the reads as `LDS.U16`. The gate exits nonzero and preserves
the report; later sanitizer cases were not run.

The host uses NVCC 13.0.88 and Compute Sanitizer 2025.3.1. NVIDIA documents
later fixes for `cp.async.mbarrier.arrive` tracking and says full accuracy of
the distinct async-arrival metadata requires a CUDA 13.1 or newer compiler.
That makes tool tracking a plausible contributor, not proof of a false positive.
The hazard remains unresolved and blocked; neither async checking nor the
error exit code was disabled. [NVIDIA sanitizer release notes](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html)

The PTX contract distinguishes async-copy completion from ordinary shared-store
publication. The next experiment must preserve both, plus reader retirement,
while reducing repeated waiting work. A version-pinned minimal protocol test
should separate a real race from tracking limitations before any promotion.
The wait instruction's suspension behavior is implementation-dependent; its
name alone does not establish inexpensive waiting. [NVIDIA PTX synchronization](https://docs.nvidia.com/cuda/parallel-thread-execution/)

## Measurement repair and preservation

The timing wrapper and child both attempted to create `paired.command.json`.
The child failed before measurements. Wrapper receipts now use `launch-` names,
with a regression test keeping parent and child ownership separate. The failed
logs remain intact. A fresh bounded unit resumed the existing contract directly
without overwriting either log or rebuilding the candidate.

Build, boundary, resumed timing, counter and sealing units completed. The initial
timing and sanitizer units retain their diagnosed failures. Qualification is
explicitly blocked; passing numerical tests do not override the hazard. GPU
work was serialized under 8 GiB limits and no swap, with a 32 GiB available-memory
reserve. Final MemAvailable was 122,501,088 KiB and no GPU process remained.

No production artifact, route, compiler default or numerical law changed. This
round does not establish a new token/s result or a fresh vLLM/SGLang comparison.
The last verified serving value remains 225.600 output token/s for 4096/64 C16;
the full-model parity gate and serving performance gap remain open.

## Reproduction identities

Remote evidence:
`/home/wlc004s/lunaflux-decode-slot-completion-20261007.7PRkRYlT`.

Downloaded evidence:
`/tmp/lunaflux-decode-slot-sealed-20261007.4Up9g4/evidence`.
All 260 manifest members verified locally. Archive SHA-256:
`d93ab1784a0624b57a35f4ee60dd500916b595be29bea39f340e571a58c39319`.

| Identity | SHA-256 |
| --- | --- |
| Candidate CUDA source | `25297b800ad6dbe6727f173eafb145e4d1ca9ae115509ef130709cd805b06e05` |
| Candidate cubin | `5b795dc42e8378ae53949ae29b163363443a0fac63001b91d7c163dd582750ec` |
| Candidate recipe | `ce498316635783f91a9f1ae4ab03b9efcab48c5311f3ab57f838e6b7193eaa31` |
| Repaired probe | `49560803fad1705204bea72b1c71546cd69fcac1d931f92323ad6de74be382cd` |
| Frozen control source | `dc9f5a1a30e0d2f969f7a9e82cdcc7b8b9a8c80f2058c698b19ea9416e9d43b6` |
| Frozen control cubin | `318d74ff5b8ebfb5e0a97ea6daea16501a98a5ef1b67116350cf8b737266c185` |

GPU UUID is `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, Spark GB10 sm121.
The sealed source snapshots own the exact executed experiment. Local formatting
and the corrected wrapper snapshot are separately retained; neither changes
the executed cubin or retrospectively repairs a failed gate.
