# Exact numeric lowering after address sharing

Follow-up to [address sharing](ATTENTION_INSTRUCTION_SHARING_2026-09-14.md).
This change removes avoidable control flow and scalar BF16 packing in the
query-owned attention source generator. It applies to synchronous and async
schedules, not a Qwen-specific graph pattern. It is a CUDA backend realization
of pure if-conversion and pair conversion, not a new universal LunaTile IR pass.

## Semantics

`isinf(score) ? 0 : expf(score-next)` becomes the same `expf` followed by a
bitwise selection. CUDA floating-point evaluation here has no observable trap
or errno effect. Consequently evaluating the unused exponential is safe even
for `inf-inf`. The mask selects **positive zero**, including when the unused
result is NaN. Multiplication by zero would not be an equivalent transformation.
No approximate exponential, FTZ flag, or reassociation was enabled.

Two independent FP32→BF16 round-to-nearest-even conversions followed by packing
become `cvt.rn.bf16x2.f32`, preserving low/high ordering. NVIDIA syntax remains
inside the CUDA source package. The standalone device regression in
`tests/attention_tile_cuda_source_probe/numeric_probe.cu` checks 524,288 pairs
and 3,145,728 exponential selections, covering every upper-16-bit float pattern
with eight low-bit patterns, rounding ties/neighbours, signed zero, subnormals,
infinities and NaNs. Both mismatch counters were zero.

## Async c322 results

RTX 5060 Ti, sm120, CUDA 13.1, `--fmad=false`; same metadata ABI, grid and input
fixture as the preceding report. Each entry is the median of 30 observations
(six processes × five alternating measurements). All 480 comparisons were
bitwise equal with maxabs=0, and KV-unchanged checks passed.

| Total query tokens | Rows | Base history | Original sync c318 µs | New async c322 µs |
|---:|---:|---:|---:|---:|
| 63 | 1 | 0 | 19.41 | 16.78 |
| 64 | 1 | 0 | 18.94 | 16.61 |
| 65 | 1 | 0 | 22.63 | 20.36 |
| 512 | 8 | 0 | 38.00 | 33.05 |
| 520 | 8 | 0 | 55.48 | 47.11 |
| 1528 | 1 | 0 | 459.57 | 331.11 |
| 1528 | 8 | 0 | 112.25 | 91.03 |
| 2048 | 1 | 0 | 763.16 | 544.08 |
| 2048 | 8 | 0 | 163.43 | 129.70 |
| 2048 | 16 | 0 | 118.99 | 97.78 |
| 2048 | 1 | 2048 | 2179.34 | 1466.49 |
| 2048 | 8 | 2048 | 1646.09 | 1034.75 |
| 1528 | 8 | 4096 | 2584.40 | 1457.31 |
| 2048 | 1 | 4096 | 3626.73 | 2390.68 |
| 2048 | 8 | 4096 | 3148.18 | 1950.27 |
| 2048 | 16 | 2048 | 1615.98 | 1003.50 |

The 1528/8/4096 point improved from the previous shared-address c322's
1790.15 µs to 1457.31 µs: **18.6% less time**. Relative to the original async
2366.24 µs it is 38.4% less time, or 1.62× as fast. The original synchronous
baseline is rerun contemporaneously; prior c322 and framework measurements
are reused, not represented as freshly rerun endpoints.

| 1528/8/4096 hardware counter | Address sharing only | This change |
|---|---:|---:|
| Warp instructions | 166,814,816 | 147,429,824 |
| MOV | 26,182,592 | 16,659,776 |
| BRA (excluding BRA.U) | 8,277,376 | 5,002,624 |
| BSSY | 5,369,664 | 2,094,912 |
| Scalar F2F.BF16.F32 | 3,274,752 | 0 |
| PRMT | 1,686,528 | 49,152 |
| Tensor activity % | 58.87 | 74.40 |
| Registers/thread | 239 | 233 |
| Local load/store sectors | 0 / 0 | 0 / 0 |

Instructions fall another 11.6%; relative to the original async kernel,
222,423,936 → 147,429,824 is a cumulative 33.7% reduction. MMA work is unchanged.
This does **not** eliminate the instruction-count gap: the prior vLLM/SGLang
paged counts were 50,401,280 / 64,378,912, so ratios remain 2.93× / 2.29×.
Their corresponding timing was 1349.84 / 1271.65 µs; this kernel is now
1.08× / 1.15× that time. This is standalone attention, **not full-serving speed**;
the API/layout/cache-control caveats in the three-way report still apply.

Remaining work is to attribute the residual move/control/address sequences
and precise numeric arithmetic before changing schedules again. Replacing
`expf` with approximate math is not part of this correction. A further isolated
`fmaxf`→PTX max experiment did not reduce static counts of MOV, FMNMX, FSETP,
FSEL, BRA or BSSY; it was not integrated or claimed as a speedup.

## Synchronous c318 cross-check

The same numeric lowering is active in c318. Its full 16-case × 30-observation
matrix also passed bitwise equality and KV-unchanged checks. Representative
medians below compare original c318 and newly compiled c318 in alternating
runs; they must not be substituted with the faster async measurements above.

| Total queries / rows / history | Original sync µs | New sync µs | Reduction |
|---|---:|---:|---:|
| 1528 / 1 / 0 | 460.37 | 374.78 | 18.6% |
| 2048 / 1 / 0 | 763.04 | 612.27 | 19.8% |
| 2048 / 8 / 2048 | 1637.93 | 1415.43 | 13.6% |
| 1528 / 8 / 4096 | 2581.79 | 2209.97 | 14.4% |

Together these are 960 bitwise-equal timing comparisons across two schedules.
Changing the running kernel selection to async still requires an actual
selection/export update and end-to-end remeasurement; this report does not
claim that a running service already uses the fastest measured artifact.
Sync timing artifacts: `/tmp/lunaflux-instruction-compiler318-timing-20260914-r1`.

At 1528/8/4096, corrected sync executes 146,025,280 instructions versus
corrected async's 147,429,824 (only 0.96% more). Yet async takes 1457.31 versus
2209.97 µs: **34.1% less time**. Tensor activity is 74.40% versus 47.95%.
Both have zero local-memory sectors. Thus the corrected async path now has a
substantial measured benefit despite nearly equal instruction counts; counting
instructions alone cannot explain the remaining schedule difference.

## Validation and reproduction

- Native full suite: **3739/3739**, warning-denied check passed.
- Source tests: 32/32; AOT compiler tests: 5/5; moon info completed.
- Async memcheck, racecheck and synccheck passed on 1528/1/0,
  2048/8/2048 and 1528/8/4096.
- No runtime selection override or production deployment was performed.

Generate with `moon run tests/attention_tile_cuda_source_probe --target native
-- 322 instruction-map` and compile using `-O3 -arch=sm_120 --fmad=false
-lineinfo -Xptxas=-v -cubin`. For the numeric regression, compile the generated
source via nvcc's `-include` option together with `numeric_probe.cu` as an
executable using the same target and math flags.

Source SHA-256: `4f516a80c825fc05444df6e59f594cc1fe2377c84c4accd499ba0048f46d56ed`.
Cubin SHA-256: `378041b96edea0e273ba20345e0d33cb1ec38c38cec66ebe3ea51c6839731cac`.
Remote artifacts: `/tmp/lunaflux-instruction-compiler2-{timing,counters,sanitizer}-20260914-r1`.
