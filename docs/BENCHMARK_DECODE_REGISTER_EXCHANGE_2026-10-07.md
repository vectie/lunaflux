# Decode register exchange: one narrow win, no C16 serving promotion

Keeping the complete score/probability row in subgroup registers improves the
ordinary two-row, 32768-key decode probe by **3.24% median paired gain**. It does
not reliably improve C16, and it regresses the complete split-plus-merge chain.
The variant remains diagnostic-only. No production route or runtime default
changed, and this is not an end-to-end throughput improvement.

## Hypothesis and implementation

The previous [lookahead experiment](BENCHMARK_DECODE_LOAD_LOOKAHEAD_2026-10-07.md)
left eligible work unchanged. This experiment isolates the cost of exchanging
F32 scores and probabilities through shared memory rather than changing operand
staging, arithmetic or launch geometry.

Two finite-budget alternatives were tested against the same frozen selected
module: ordinary candidate 468 and split partial/merge 3903/3904. Both use KV32,
owned8 blockwise F32 arithmetic, D128, GQA2 and two independent operand stages.
The split control has eight partitions with partition grain 32; timings include
both partial and merge. Neither variant changes the merge.

- `probability`: retain each lane's F32 probability and broadcast it to the PV
  consumer. Scores still cross shared memory.
- `row`: also exchange the original dot-owner's F32 score into its logical-key
  lane, then retain both score and probability in registers.

For logical key `k`, the score source lane is `(k % 8) * 4`, in score group
`(k / 8) * 8`. All active tails 1 through 32 preserve this ownership. Dot-product
association, max/denominator reduction ownership, exponential arithmetic and PV
accumulation order remain unchanged. Complete and ragged loops are transformed
in both ordinary and partial entries. Existing CTA barriers and the conservative
33040-byte dynamic shared reservation remain unchanged to isolate transport
from occupancy and synchronization changes.

The transformations are offline diagnostic automation, not production
source-string compiler passes. They do not relabel the numerical contract or
enable reassociation, fast math or FMA. Production integration, if justified by
a future whole-chain result, belongs in a pure ownership/transport plan consumed
by CUDA lowering—not in the model builder, scheduler or token-step path.

## Matched timings

Each cell has five alternating control/candidate pairs, unchanged independent
scalar-oracle checks and mandatory bitwise control equality. All **60** pairs
passed with `bitwise=true` and `maxabs=0`. The acceptance rule requires at least
1% gain in every pair; a favorable median alone is insufficient.

Positive values mean shorter completion time. Envelope is the spec's maximum-row
domain, not the number of useful rows. Values are median paired gains, not ratios
of independently selected medians.

| Useful rows / envelope / history | Ordinary probability | Ordinary full row | Split + merge probability | Split + merge full row |
| --- | ---: | ---: | ---: | ---: |
| 16 / 16 / 4095 | -0.08%, regression | +0.11%, inconclusive | -1.29%, regression | -0.87%, regression |
| 2 / 8 / 32767 | -1.45%, regression | **+3.24%, improved** | +1.38%, inconclusive | -1.31%, regression |
| 1 / 1 / 127 | +0.08%, inconclusive | +0.73%, inconclusive | -1.50%, regression | +0.78%, inconclusive |

The ordinary long full-row cell's minimum paired gain is +2.40%. Its independent
median times are 1429.905 and 1368.429 microseconds; their ratio must not replace
the paired 3.24% decision. The split long control is already faster than this
ordinary winner in these samples, so the ordinary improvement is not grounds
to switch the serving chain.

## Why the change helps long C2 but not C16

Paired Nsight captures profile the actual ordinary control and full-row CUBINs,
with block 64 and identical operands, reservation and grid per workload. C16 uses
grid 16 x 8; the two-row long-history probe retains envelope eight and grid 8 x 8.
Profile replay is a diagnostic measurement, not the primary paired timing gate.

| Counter | C16 control | C16 full row | Long C2 control | Long C2 full row |
| --- | ---: | ---: | ---: | ---: |
| Replay duration, microseconds | 1189.536 | 1190.560 | 1608.224 | 1548.832 |
| Warp instructions | 41,500,160 | 42,481,920 | 41,295,520 | 42,278,400 |
| Registers per thread | 148 | 143 | 148 | 143 |
| Shared-memory resident-block limit | 2 | 2 | 2 | 2 |
| Active occupancy | 8.15% | 8.01% | 5.69% | 5.40% |
| Eligible warps per scheduler cycle | 0.09 | 0.09 | 0.36 | 0.41 |
| Issue active | 9.03% | 9.06% | 36.19% | 40.64% |
| Short-scoreboard cycles per active issue | 3.76 | 4.44 | 0.68 | 0.51 |
| Long-scoreboard cycles per active issue | 2.97 | 2.42 | 0.19 | 0.24 |
| MIO-throttle cycles per active issue | 2.19 | 2.07 | 0.07 | 0.15 |
| Fixed-wait cycles per active issue | 0.64 | 0.42 | 0.65 | 0.42 |
| Barrier cycles per active issue | 0.38 | 0.41 | 0.05 | 0.07 |

Stall ratios are cycles per active issue, not percentages of wall time. They
cannot be added to attribute a completion-time delta.

The executed instruction exchange is concrete:

| SASS opcode | C16 control → full row | Long C2 control → full row |
| --- | ---: | ---: |
| `LDS.128` | 262,144 → 0 | 262,144 → 0 |
| `LDS` | 98,304 → 32,768 | 98,304 → 32,768 |
| `STS` | 163,840 → 0 | 163,840 → 0 |
| `SHFL.IDX` | 639,232 → 1,819,136 | 624,672 → 1,804,352 |
| `BRA` | 299,776 → 332,544 | 299,104 → 331,872 |

The control packs probability reads into 128-bit shared loads. Replacing each
logical probability broadcast with a scalar shuffle removes shared exchange but
introduces many more indexed shuffles. Both workloads retain 4,194,304
`LDS.U16` operand instructions and 1,048,576 `LDS.64` instructions; FADD and FMUL
counts are unchanged within each pair. Compilation reports no register spills.

For C16, total instructions rise 2.37%, short dependencies rise about 18%, and
eligible work does not increase. Replay is effectively unchanged. In long C2,
instructions also rise about 2.38%, but eligible work rises and short dependencies
fall; replay is 3.69% shorter, consistent in direction with the unprofiled win.
The differing outcome demonstrates why a transport choice needs workload-aware
selection. It does not establish that fewer registers, fewer shared accesses or
more shuffles are universally better.

The remaining C16 target is operand-consumer scheduling and reuse that improves
eligible work without replacing packed loads with a longer dependent scalar
sequence. Removing score exchange alone does not remove its dominant operand
loads. Whole-chain timings, not a partial-kernel win, must decide integration.

## Correctness, resources and scope

All **44** boundary checks passed bitwise with the independent oracle and KV
integrity checks. They cover both variants, ordinary and split/merge, rows two
within envelope eight, histories 0, 6, 7, 8, 30, 31, 32, 63, 64, 66 and 4096.
All **36** sanitizer runs passed: memcheck with leak checking, racecheck and
synccheck, both variants and chains, histories 0, 32 and 4096. Exit failures are
not accepted even if a log contains a zero-error substring.

The experiment automation passed warning-denied check and **3/3** ownership/
transformation tests; the boundary/sanitizer runner and finisher passed
warning-denied checks. This is not a new full-repository native test result.
The source-storage legality repair remains in commit `ca0d3def`; this campaign
does not change production CUDA lowering or defaults.

GPU work was serialized. User-systemd jobs had 8 GiB memory limits, zero process
swap and 900-second limits; counter containers also had 8 GiB limits. Monitoring
retained a 32 GiB host-memory reserve. No memory-limit failure occurred.

The last verified 4096-input/64-output C16 serving rate remains **225.600 output
tok/s**. Historical matched vLLM/SGLang rates remain 243.03/241.85 tok/s
(approximately 7.7%/7.2% completion-time overhead). None of those serving runs
was repeated here. Synthetic kernel equality does not close the previously open
end-to-end token-parity gate or establish universal performance parity.

## Reproduction and retained evidence

Host: Spark .179, GB10 sm121, 48 SMs, UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`. nvcc 13.0.88 SHA-256:
`fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.

| Artifact | SHA-256 |
| --- | --- |
| Frozen source | `dc9f5a1a30e0d2f969f7a9e82cdcc7b8b9a8c80f2058c698b19ea9416e9d43b6` |
| Frozen CUBIN | `318d74ff5b8ebfb5e0a97ea6daea16501a98a5ef1b67116350cf8b737266c185` |
| Unchanged check probe | `821fa177be7e30219dbaf92a58c7337af45529075dca84c9fe162017c7ee1b66` |
| Probability source | `37615292b3bb99cba157e1ffba399ee38dc164027d5250368287a2a2772fc6bf` |
| Probability CUBIN | `2d9949a1720b3e27650ae99d0a85d8e89a442345c7164031ba4e4bd3872b2aba` |
| Full-row source | `28e8688885e81f7f5c1c8f68c085f40bbb61c545ee8952efdd8c7b236d4d1c54` |
| Full-row CUBIN | `74558bf77153f0e5e095629ad486b3e7b7e9496aaac4ad8d5a15a0fa3917ecd3` |

Automation:

- `benchmarks/gpu_pipeline/decode_register_exchange_20261007.mbtx`:
  `FROZEN_BASE PROBE_ROOT NEW_ROOT build|run MOON`.
- `benchmarks/gpu_pipeline/decode_register_exchange_gates_20261007.mbtx`:
  `FROZEN_BASE PROBE_ROOT TRIAL_ROOT boundary|sanitizer`.
- `benchmarks/gpu_pipeline/finish_decode_register_exchange_20261007.mbtx`:
  `TRIAL_ROOT C16_COUNTER_ROOT LONG_COUNTER_ROOT FROZEN_BASE PROBE_ROOT MOON SEALER`.

Three failed preparation attempts are retained: an incorrect expected source
barrier spelling, an unused score pointer rejected by nvcc, and a macro spelling
mismatch. These were automation/build failures before GPU execution, not accepted
candidate results. Actual frozen-source validation and both final CUDA builds
passed. Initial sealing also referenced an absent helper; the missing helper was
uploaded and sealing completed without rerunning measurements or overwriting
artifacts. The committed finisher checks sealer availability before snapshots.

Remote archive:
`/home/wlc004s/lunaflux-decode-register-exchange-v4-20261007.CeQgdl09.verified.tar.gz`.
Archive SHA-256:
`738772f76fdaead8b037fe8d3f061055b4809fd6ce09f3eb1c101eebd51a46f0`.
Local copy: `/tmp/lunaflux-register-exchange-evidence-20261007.kcCpgSDM/`.
Archive hash and all **468** manifest entries verified after extraction into a
fresh directory. Recipes, sources, binaries, paired results, boundary/sanitizer
logs, both raw Nsight reports and unit statuses/journals are retained; original
artifacts were not overwritten.
