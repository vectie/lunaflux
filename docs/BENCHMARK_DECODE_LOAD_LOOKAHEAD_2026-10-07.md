# Decode live storage repair and load-lookahead measurements

The blockwise source validator now derives its minimum shared extent from a
pure physical-IR storage plan instead of unconditionally adding sixteen bytes.
Two arithmetic-preserving load-lookahead experiments did **not** produce a
reliable speedup. Neither is selected for production. This repairs a compiler
legality bug; it does not close the remaining serving performance gap.

## Compiler repair

`compiler/attention_physical_ir/BlockwiseTileStorage` describes staged BF16 K/V
operands, F32 score exchange and optional shared validity cells. Region sizes
and offsets are pure computations; invalid geometry and signed launch-byte
overflow are rejected before multiplication. The CUDA renderer supplies the
validity ownership already proven by its physical effect plan.

For D128, KV32, two stages and two query consumers, register-retained validity
requires 33024 bytes. Two shared validity cells require 33032 bytes. The previous
unconditional minimum was 33040 bytes. Conservative launch reservations remain
legal and the selected launch still reserves 33040 bytes: minimum live extent
is not the same as an optimal residency policy. The previous extent experiment
showed that shrinking the reservation can regress C16 despite permitting more
resident blocks. No NVIDIA allocation granularity enters the portable plan.

This change preserves generated CUDA instructions, numerical law, selection
and request-path behavior. It fixes false rejection of a sufficient reservation,
not an out-of-bounds write in the existing conservative artifact. Shared cells
remain required whenever the register-retention proof is absent.

Also fixed in the affected validation path: a production package import used
only by white-box tests, and a redundant enum qualification rejected by warning
73. These are build hygiene fixes, not throughput optimizations.

## Experiment and matched controls

The frozen selected decode module contains ordinary candidate 468 and split
partial/merge entries 3903/3904. The ordinary schedule is owned8 blockwise F32,
KV32, two independent operand stages, BF16 storage and F32 probabilities. The
split chain uses eight partitions and includes its merge; a partial-only win
cannot qualify the serving chain.

The two diagnostic variants preload two or four independent score products,
then accumulate them in their original logical order. Both complete and ragged
loops in both ordinary and split entries are transformed. No reassociation,
FMA substitution, numerical-law relabeling or production source-string pass is
introduced. Strict nvcc options retain the control's arithmetic behavior.

Each cell has five alternating pairs, an unchanged independent scalar oracle,
and mandatory bitwise control equality. All 60 captured pairs have
`bitwise=true` and `maxabs=0`. The decision requires at least 1% gain in **every**
pair. A positive median alone does not establish a reliable gain.

Positive values below mean shorter completion time; values are median paired
gains, not ratios of independent medians. Envelope is the spec's maximum-row
domain, distinct from useful rows.

| Useful rows / envelope / history | Ordinary lookahead 2 | Ordinary lookahead 4 | Split + merge lookahead 2 | Split + merge lookahead 4 |
| --- | ---: | ---: | ---: | ---: |
| 16 / 16 / 4095 | -0.02%, regression | -1.50%, regression | -0.07%, regression | -0.05%, regression |
| 2 / 8 / 32767 | -0.38%, regression | +0.42%, inconclusive | +0.40%, inconclusive | -0.07%, regression |
| 1 / 1 / 127 | 0.00%, inconclusive | +0.03%, inconclusive | -1.33%, regression | +0.84%, inconclusive |

Neither candidate clears the gate in any cell. Some independent medians look
more favorable because they discard pairing; they must not replace the paired
decision. This is selected-kernel testing, not a new end-to-end benchmark.

## Why lookahead did not help

Matched Nsight replay compares the actual ordinary C16 kernel, grid 16 x 8,
block 64, history 4095, identical 33040-byte reservation and operands.

| Counter | Control | Lookahead 2 |
| --- | ---: | ---: |
| Warp instructions | 41,500,160 | 41,570,560 |
| Registers per thread | 148 | 142 |
| Shared-memory resident-block limit | 2 | 2 |
| Active occupancy | 7.93% | 7.83% |
| Eligible warps per scheduler cycle | 0.09 | 0.09 |
| Issue active | 8.80% | 8.98% |
| Long-scoreboard cycles per active issue | 3.00 | 2.81 |
| Short-scoreboard cycles per active issue | 3.79 | 4.08 |
| MIO-throttle cycles per active issue | 2.17 | 1.99 |
| Barrier cycles per active issue | 0.42 | 0.43 |
| Profile replay duration, microseconds | 1200.960 | 1205.888 |

Instruction count rises 0.17%. Short dependency waits rise 7.65%, eligible work
does not increase, and replay duration rises 0.41%. Stall ratios are cycles per
active issue, **not percentages of wall time**; their changes are not additive
attributions of completion time. Both captures report zero source-correlated
excessive shared wavefronts. Compilation reports zero register spills.

SASS counts retain 4,194,304 `LDS.U16` instructions, 8,945,408 `FMUL` and
9,207,296 `FADD` instructions. The source preload spelling changes register
allocation but does not create a materially better executed shared-load
schedule. This falsifies this local lookahead implementation as the remedy;
it does not prove that all latency-hiding designs are ineffective.

Combined with the previous packet and reservation ablations, the remaining
target is consumer/data-reuse scheduling that increases useful eligible work
without increasing shared-pipeline pressure. More occupancy, fewer instructions
or fewer conflicts individually are insufficient. A future alternative must
still qualify ordinary, mixed-envelope and complete split/merge workloads.

## Validation and serving status

Affected-package native check and tests passed: **133/133**. The checks use
`--deny-warn --warn-list '+73-20-25-79'`: existing toolchain-migration warning
categories 20, 25 and 79 are suppressed explicitly. This is not a claim that
the full repository passes the unsuppressed warning-denied boundary. The new
storage tests cover exact extents, retained/shared validity, region boundaries,
conservative reservations, unsupported geometry and overflow. `moon info`
regenerated the owned package interface.

The load-lookahead MoonBit automation passed warning-denied check and its
transformation test (1/1); the evidence finisher passed warning-denied check.
All GPU diagnostic units finished successfully. No production kernel or
runtime default was changed by these experiments.

The last verified 4096-input/64-output C16 serving rate remains **225.600 output
tok/s**, not a new result from this campaign. Historical matched-workload
vLLM/SGLang rates remain 243.03/241.85 tok/s, corresponding to approximately
7.7%/7.2% completion-time overhead. Neither reference nor serving was rerun
here, and the prior end-to-end token-parity gate remains open.

## Reproduction and evidence

Host: Spark .179, GB10 sm121, 48 SMs, UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`. Tool: nvcc 13.0.88, SHA-256
`fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.

Frozen source SHA-256:
`dc9f5a1a30e0d2f969f7a9e82cdcc7b8b9a8c80f2058c698b19ea9416e9d43b6`.
Frozen CUBIN SHA-256:
`318d74ff5b8ebfb5e0a97ea6daea16501a98a5ef1b67116350cf8b737266c185`.
Unchanged check-probe SHA-256:
`821fa177be7e30219dbaf92a58c7337af45529075dca84c9fe162017c7ee1b66`.

Automation is `benchmarks/gpu_pipeline/decode_load_lookahead_20261007.mbtx`
(`FROZEN_BASE PROBE_ROOT NEW_ROOT build|run MOON`) and
`finish_decode_lookahead_20261007.mbtx`
(`TRIAL_ROOT COUNTER_ROOT FROZEN_BASE PROBE_ROOT MOON SEALER`). The archive
also contains the paired probe, recipes, generated sources, CUBINs, unchanged
controls, raw trial output, Nsight report/source counters and unit journals.

Remote archive:
`/home/wlc004s/lunaflux-decode-load-lookahead-20261007.5ggIhQYP.verified.tar.gz`.
SHA-256: `8f959d5f354c0658bcbcf366707b1f3b63650be515d66a50180f278fe7a9bee7`.
Local copy: `/tmp/lunaflux-lookahead-evidence-20261007.wadgG9Cb/`;
archive hash and all 136 manifest entries verified after extraction into a
fresh directory. Original artifacts were not overwritten.

Workloads were serialized, with user-systemd 8 GiB memory, zero process swap
and 900-second limits. Counter containers used an 8 GiB limit. Host monitoring
reserved 32 GiB; no memory-limit failure occurred. No inference deployment,
runtime JIT or concurrent GPU benchmark was introduced.
