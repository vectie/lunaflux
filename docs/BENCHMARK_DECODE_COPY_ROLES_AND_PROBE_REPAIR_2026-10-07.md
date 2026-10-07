# Decode copy ownership and independent merge launch repair

**The main C16 serving gap remains open.** Dedicated operand-copy warps passed
correctness and sanitizer checks, but no timing cell passed the paired
acceptance rule. The experiment is diagnostic only; no serving artifact,
compiler default, numerical law or production route changed.

Two genuine diagnostic bugs were fixed: split merge inherited the partial
entry's block size, and profiler container names collided across experiment
roots. These fixes improve measurement validity, not serving throughput.

## Hypothesis and explicit physical ownership

This follows the [copy-cache experiment](BENCHMARK_DECODE_CHAIN_AND_COPY_CACHE_2026-10-07.md).
The finite budget was one structural candidate, six paired timing cells and
one matched C16 hardware-counter pair. The hypothesis was that moving K/V
staging away from arithmetic warps would hide copy latency without changing
the arithmetic fold.

The frozen selected ordinary entry is
`lunaflux_attention_decode_tile_compiler_v1_owned8_blockwise_f32_v4`:
KV32, D128, GQA2, two independent K/V slots, block 64 and shared reservation
33,040 bytes. Partial entry 3903 uses eight partitions; merge 3904 completes
the split chain. The numerical law remains
`owned8-blockwise-f32-probability-v4`.

An offline pure `CopyRoles` mapping assigns the original 64 copy owners to
threads 64–127. Threads 0–63 retain QK, softmax, PV and output ownership.
Both compute entries become block 128; merge stays block 64. Producers alone
commit and wait for asynchronous copies. All warps still join the original
CTA publication barriers, including uniform invalid-page exits and the final
drain. Four validity integers occupy the final 16 bytes of the existing shared
reservation. No arithmetic reassociation, FMA substitution, tolerance change,
layout change or `.ca` cache-policy change is allowed.

This is an exact-source diagnostic transform, not a generic production pass.
The role mapping is tested across subgroup counts 1–16, but generated-code
coverage is restricted to the pinned GQA2/KV32/D128 module. It must not be
advertised as general serving coverage or merged as a compiler default.

## Repair the split-chain probe first

The initial ordinary boundary cases all passed. The split history-zero case
failed because the probe launched merge with the partial block size, 128.
The unchanged merge entry requires 64 and rejects the launch. This was a
launch-contract failure, not evidence that the copy-role arithmetic was wrong.

The probe now reads independent `partition_merge_block`/`merge_block` geometry,
launches merge with that block, and prints both compute and merge blocks.
Legacy recipes without an independent declaration retain their common-block
behavior. Pure C++ regressions cover independent 128/64 and 64/128 geometry and
invalid bounds. The repaired candidate recipe declares merge 64. The original
and repaired candidate CUDA source and cubin are byte-identical.

Both frozen control and candidate were rerun with the same repaired probe.
This is not an asymmetric comparison between two probe versions.

The profiler now scopes a container name by both experiment parent and output
leaf. Failure cleanup uses only the container ID created by this invocation;
a name-allocation failure does not authorize stopping an existing container.
An explicit probe-hash override still requires a complete SHA-256 pin.

## Alternating unprofiled results

Each cell contains five alternating pairs, with strict bitwise equality,
independent scalar-oracle checks and KV integrity. Positive paired gain means
faster completion. Acceptance requires every pair to gain at least 1%.
Independent medians must not override the paired decision.

| Useful rows / envelope / history | Chain | Control median, µs | Copy-role median, µs | Median paired gain | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| 16 / 16 / 4095 | Ordinary | 1159.982 | 1152.715 | -0.192% | Regression |
| 2 / 8 / 32767 | Ordinary | 1436.524 | 1560.079 | -8.881% | Regression |
| 1 / 1 / 127 | Ordinary | 8.227 | 8.218 | +0.117% | Inconclusive |
| 16 / 16 / 4095 | Partial + merge | 1171.184 | 1186.567 | -1.264% | Regression |
| 2 / 8 / 32767 | Partial + merge | 1207.443 | 1204.914 | +0.417% | Inconclusive |
| 1 / 1 / 127 | Partial + merge | 10.159 | 10.262 | -1.008% | Regression |

All 30 samples were bitwise equal, with maximum absolute differential zero.
The long-C2 cell uses useful rows two inside an eight-row physical envelope,
verified in the probe's geometry output. Timing includes both launches of
every split chain.

## C16 counters: waits move into cross-role publication

Matched ordinary C16/history-4095 capture records exactly two launches:
control block 64 and candidate block 128, both grid 16 × 8. Replay times below
are diagnostic, not the unprofiled acceptance samples above.

| Metric | Control | Copy roles |
| --- | ---: | ---: |
| Replay duration, µs | 1226.176 | 1210.944 |
| Warp instructions | 41,500,160 | 45,701,376 |
| Registers/thread | 148 | 154 |
| Active warps, percent of peak | 8.08 | 15.84 |
| Eligible warps per scheduler cycle | 0.09 | 0.11 |
| Issue-active percent, active cycles | 8.82 | 9.41 |
| L2 read sectors | 8,398,566 | 8,399,192 |
| Long-scoreboard cycles per active issue | 3.04 | 1.33 |
| Short-scoreboard cycles per active issue | 3.78 | 6.05 |
| MIO-throttle cycles per active issue | 2.16 | 1.85 |
| Barrier cycles per active issue | 0.39 | 8.96 |
| Source-correlated excessive shared wavefronts | 0 | 0 |

The mathematical work did not decrease: `FADD` remains 9,207,296 and `FMUL`
8,945,408. Both captures execute 524,288 `LDGSTS.E.128` copies plus 8,192
zero-fill copies. Total instructions grow **10.12%**. Executed
`BAR.SYNC.DEFER_BLOCKING` instructions double, 131,584 → 263,168, because more
warps participate in the unchanged CTA barriers. `WARPSYNC.ALL` grows from
zero to 328,192 and `SHFL.IDX` from 639,232 to 1,163,776. NVCC's resulting
control/reconvergence and shared-validity handling costs are not free.

The candidate's hottest sampled PCs have barrier waits: 16,515 of 16,563
samples at a branch, 16,223 of 16,277 at a uniform-register move, and 14,148
of 14,311 at the shared validity load. Those are waiting locations, not proof
that each instruction itself caused the wait. Removing them without preserving
producer/consumer ownership and slot lifetime would be unsafe.

**Conclusion:** merely assigning dedicated producer warps while retaining
every whole-CTA publication point does not produce an efficient pipeline.
Long-load wait falls, but synchronization and supporting instructions rise.
Higher occupancy is not a speedup. The next hypothesis must address the
producer-to-consumer completion dependency and slot-release lifetime together,
with explicit effect/ownership plans and matched numerical checks—not add more
warps, delete barriers blindly or repeat a cache-policy ablation.

Stall ratios are not additive wall-time percentages. L2 sectors are 32-byte
requests, not necessarily DRAM bytes. Missing DRAM-byte counters on this host
are unavailable data, not zero traffic. Compilation and replay report no
spills; local spilling requests are zero.

## Validation, preserved failures and scope

- Repaired boundary checks: **22/22**, ordinary and split, histories
  0, 6, 7, 8, 30, 31, 32, 63, 64, 66 and 4096.
- Sanitizer checks: **18/18**, memcheck with leak checking, racecheck and
  synccheck, both chains at histories 0, 32 and 4096.
- Owned MoonBit automation tests: **7/7**, warning-denied native; C++ geometry
  regression compiled with `-Wall -Wextra -Werror` and passed.
- Initial missing-spec setup, incorrect merge launch, container-name collision
  and collector syntax failure remain preserved. The setup, container and
  syntax failures occurred before GPU execution. None is relabeled as a pass.
- User-systemd and profiler-container limits: 8 GiB, no additional swap,
  900 seconds; preserve 32 GiB `MemAvailable`. Counter minimum observed:
  **120,687,288 KiB** (about 115 GiB). GPU idle at the final seal.

No end-to-end serving or reference-framework campaign ran in this iteration.
The last verified serving value remains **225.600 output tok/s** for
4096-input/64-output, C16. Historical vLLM/SGLang comparisons are not refreshed
by these kernel measurements. Full-model parity and the main serving gap are
still open; this report does not claim that all performance problems are fixed.

## Reproduction and sealed evidence

Host: Spark `.179`, GB10/sm121, GPU
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`. NVCC 13.0.88 SHA-256:
`fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.

| Artifact | SHA-256 |
| --- | --- |
| Frozen control CUDA source | `dc9f5a1a30e0d2f969f7a9e82cdcc7b8b9a8c80f2058c698b19ea9416e9d43b6` |
| Frozen control cubin | `318d74ff5b8ebfb5e0a97ea6daea16501a98a5ef1b67116350cf8b737266c185` |
| Copy-role CUDA source | `19b8b3e0f1da69903e4afa871696df10ef7c42edd5928436ab11f9d752b8fbec` |
| Copy-role cubin | `3a424690e8c48be14a8084511f6b5c13c52c7db84770d976a208a383c110f08e` |
| Repaired recipe | `3fef262e2a07b645d62ce2f94653b98145b9c8c66d1150ef7df1780d7d14c7ab` |
| Repaired probe | `49560803fad1705204bea72b1c71546cd69fcac1d931f92323ad6de74be382cd` |

Remote immutable evidence roots:

- `/home/wlc004s/lunaflux-decode-copy-roles-20261007.RVuolRZ4`
- `/home/wlc004s/lunaflux-decode-role-probe-repair-20261007.M7iXJcOQ`

Local non-overwriting download:
`/tmp/lunaflux-decode-copy-roles-sealed-20261007.PCLMVh`.
Both archive hashes and all **427** manifest members were verified locally:

- Initial archive: `eb148783573bdf36471834f4e5e2c32236fdf88358d78c1d46a9ed78bb923040`, 97 members.
- Repaired archive: `4f2ba237e2c827f1dda64c375698cc181945c921c7006728d5451e4f758b6f49`, 330 members.

The repaired archive contains `followup/paired/summary.json`, all timing and
boundary outputs, sanitizer logs, actual launch geometries, frozen control,
probe source/binary and `role-counter-v2-c16/counter.ncu-rep`, raw CSV and SASS.
Build caches/dependency trees are excluded without deleting originals.
