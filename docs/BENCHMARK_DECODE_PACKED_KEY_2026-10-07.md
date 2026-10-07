# Decode packed key loads add work without a reliable gain

Neither wider QK shared-load variant improves the complete decode chain reliably.
The experiment closes the load-width hypothesis, not the remaining serving gap.
Both variants remain diagnostic-only; production selection and the runtime are
unchanged.

## Exact change and numerical law

The frozen module contains ordinary candidate 468 and split partial/merge
3903/3904: owned8 blockwise F32 arithmetic, head dimension 128, KV32, GQA2, two
independent operand stages, block size 64 and 33040 bytes of dynamic shared
storage. Split measurements include both partial and merge, with eight
partitions and partition grain 32.

The QK loop assigns four lanes adjacent BF16 components. Two alternatives replace
the 16-bit key read with an aligned packed read and bitwise extraction:

- `word32` reads two adjacent components in one 32-bit word.
- `word64` reads four adjacent components in one 64-bit word.

Each lane still consumes its original single component. The transform preserves
the component ownership, per-dot addition order, shuffle reduction, softmax, PV
fold, barriers, storage reservation and merge. Complete and ragged loops are
changed in both ordinary and partial entries. No fast math, FMA or reassociation
is enabled. Alignment follows from the existing aligned stage allocation and
128-component row stride. The implementation exhaustively checks BF16 payload
extraction and the owner/component mapping.

This is an offline load-width ablation, not a production source-string compiler
pass. General integration would belong in a pure ownership/transport plan and
terminal CUDA lowering, not in the model, scheduler or request path.

## Matched workload vector

Each cell uses five alternating control/candidate pairs and the unchanged
independent scalar oracle and KV integrity checks. All **60** pairs passed
bitwise control equality with `maxabs=0`. Acceptance requires at least 1% gain
in every pair. Positive values below mean shorter completion time; they are
median paired gains, not ratios of independently selected median durations.

| Useful rows / envelope / history | Ordinary word32 | Ordinary word64 | Split and merge word32 | Split and merge word64 |
| --- | ---: | ---: | ---: | ---: |
| 16 / 16 / 4095 | -1.50%, regression | -0.19%, regression | +0.70%, inconclusive | +0.01%, inconclusive |
| 2 / 8 / 32767 | -1.03%, regression | -3.39%, regression | -0.04%, regression | -0.26%, regression |
| 1 / 1 / 127 | +0.14%, inconclusive | +0.08%, inconclusive | -0.66%, regression | +0.02%, inconclusive |

The ordinary C16 independent median durations are 1147.518 → 1166.335
microseconds for word32 and 1149.083 → 1161.318 for word64. The slightly positive
split C16 medians include negative paired samples; neither meets acceptance.
The ordinary long word64 cell's worst paired regression is 5.23%.

## Actual emitted loads and added instructions

Two paired Nsight captures use the actual ordinary control and candidate CUBINs,
identical inputs and grid 16 x 8. Replay durations are diagnostic, not substitutes
for the unprofiled paired gate. The control is separately captured in each pair.

| Counter | Control for word32 | Word32 | Control for word64 | Word64 |
| --- | ---: | ---: | ---: | ---: |
| Replay duration, microseconds | 1202.688 | 1200.000 | 1188.320 | 1200.704 |
| Warp instructions | 41,500,160 | 46,490,112 | 41,500,160 | 49,947,136 |
| Registers per thread | 148 | 210 | 148 | 150 |
| Shared-memory resident-block limit | 2 | 2 | 2 | 2 |
| Eligible warps per scheduler cycle | 0.09 | 0.10 | 0.09 | 0.11 |
| Issue active | 8.83% | 10.17% | 9.00% | 10.57% |
| Short-scoreboard cycles per active issue | 3.89 | 3.61 | 3.77 | 3.57 |
| Long-scoreboard cycles per active issue | 3.04 | 2.59 | 2.91 | 2.20 |
| MIO-throttle cycles per active issue | 2.20 | 1.65 | 2.17 | 1.44 |
| Source-correlated excessive shared wavefronts | 0 | 0 | 0 | 0 |

Stall ratios are cycles per active issue, not percentages of wall time. A lower
ratio can accompany more issued instructions; it cannot establish that absolute
waiting time fell or attribute a completion-time delta.

The control executes 4,194,304 `LDS.U16` QK instructions. Word32 replaces these
with 4,194,304 scalar 32-bit `LDS` instructions; word64 replaces them with
4,194,304 `LDS.64` instructions. The original 1,048,576 `LDS.64` PV reads remain,
giving word64 5,242,880 executed 64-bit shared loads in total. **The QK load
instruction count does not decrease.** Each lane reads more bits but still owns
only one component of each packed word.

`LOP3.LUT` execution rises from 2,153,216 to 6,365,184 in word32 and 6,352,896
in word64. FADD and FMUL counts remain 9,207,296 and 8,945,408. Wider transport
therefore retains the mathematical work and adds component extraction and
selection work: total instructions rise **12.02%** and **20.35%**. Word32 also
raises register usage substantially. Both builds report zero stack frames and
zero register spill loads/stores. Neither changes the limiting shared reservation
or introduces source-correlated excessive shared wavefronts.

The narrower dependency metrics do not offset this added work. The result rejects
“widen the load and then unpack independently in every lane” as the C16 fix. It
does not reject packed loads when one consumer genuinely uses all packed elements.
The existing PV loop already does that; the QK ownership in this experiment does
not.

## Correctness and resource limits

All **44** boundary checks passed, covering both variants and chains, two useful
rows within envelope eight, and histories 0, 6, 7, 8, 30, 31, 32, 63, 64, 66 and
4096. All **36** sanitizer invocations passed: memcheck with leak checking,
racecheck and synccheck for both variants and chains at histories 0, 32 and 4096.
Each also checks output equality, oracle error and KV integrity.

The packed transform passed warning-denied check and **2/2** tests. The reusable
gate runner now accepts two validated diagnostic variant names while preserving
its prior invocation; its warning-denied check and **1/1** safety regression
passed. The finisher passed warning-denied check. These are focused automation
checks, not a new full-repository native suite result.

GPU work was serialized on Spark 179, UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, sm121, with pinned CUDA 13.0.88.
User units cap memory at 8 GiB, disable swap and cap runtime at 900 seconds.
Counter containers use an 8 GiB memory and combined memory/swap limit. The
32 GiB host reserve remained intact; paired-trial available memory stayed above
119 million KiB. No serving process or unrelated workload was terminated.

One failed preparation is preserved: the first uploaded MoonBit diagnostic had
invalid unary bitwise syntax and deprecated integer conversion. It failed before
GPU admission. The corrected script uses bit masks `-2` and `-4` in its MoonBit
ownership tests; CUDA expressions retain valid `~1` and `~3` syntax. Its successful
build, trials and gates run in a new directory.

## Decision and next experiment

Do not promote either variant. The preceding
[register-exchange experiment](BENCHMARK_DECODE_REGISTER_EXCHANGE_2026-10-07.md)
also left C16 unimproved. Width and exchange changes cannot be represented as
completed performance fixes merely because they preserve outputs.

The next hypothesis should target **independent QK work scheduling**, rather than
reading unused adjacent payloads: interleave two or four independent key dots
while preserving each dot's ordered component fold and original owner. This can
expose independent loads and accumulators without changing the numerical law;
whether the compiler emits a better schedule must be measured, not assumed.
Use a fresh finite budget and retain complete split/merge timing and ragged-tail
checks. Any production integration still requires actual serving dispatch and
end-to-end verification.

The last verified serving result remains **225.600 output tokens/s** for
4096-input/64-output C16. Historical matched controls were 243.03 for vLLM and
241.85 for SGLang, approximately 7.7% and 7.2% longer completion time for
LunaFlux. This kernel campaign is not a new serving or baseline retest. The
end-to-end parity gate remains open; synthetic kernel bitwise equality does not
close model-quality verification.

## Reproduction and preserved evidence

Owned automation:

- `benchmarks/gpu_pipeline/decode_packed_key_load_20261007.mbtx`
- `benchmarks/gpu_pipeline/decode_register_exchange_gates_20261007.mbtx`
- `benchmarks/gpu_pipeline/finish_decode_packed_key_20261007.mbtx`

Remote terminal experiment:
`/home/wlc004s/lunaflux-decode-packed-key-v2-20261007.K8QgNcUm`.
Its word32 and word64 paired counter roots are
`/home/wlc004s/lunaflux-packed32-counter-20261007.mN9xCdPj` and
`/home/wlc004s/lunaflux-packed64-counter-20261007.JaT2gwEx`.

Archive SHA-256:
`87825ccfa6d1a9c28e3e2008f3bcbfe75577f5dbd40545c3b20e54e1157d541d`.
Downloaded without overwrite to
`/tmp/lunaflux-packed-key-evidence-20261007.1wR4WU6j`;
the local archive hash matches and all **426** manifest entries verify.
The archive retains variants, frozen control, unchanged probe, raw trials,
sanitizers, counter reports/SASS, preparation failure and terminal unit records.
No original evidence was removed.
