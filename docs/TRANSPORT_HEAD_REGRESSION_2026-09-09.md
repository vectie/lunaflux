# Shared projection transport: selected-row regression

The row-preserving XOR experiment must not be promoted wholesale. The shared
generator also covers selected-row vocabulary projection, which has different
reuse and resource requirements from dense QKV/output. There is no model-name
branch in this experiment.

## Exact-source comparison

The baseline was exported from detached commit `ca5796e`, not from an older
unpadded experiment. Both sources were compiled with CUDA 13.1, sm120, O3,
`--fmad=false`, and `--maxrregcount=128`. The test uses GPU UUID
`GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`.

Three paired trials, twenty CUDA-graph launches per timing, median microseconds:

| Input tokens | Selected rows | Baseline | XOR with retained padding |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 737.62 | 737.14 |
| 2 | 2 | 1230.87 | 1229.44 |
| 7 | 7 | 1236.36 | 1234.78 |
| 8 | 8 | 1237.47 | 1235.56 |
| 17 | 17 | 1362.95 | 1350.96 |
| 32 | 32 | 1401.60 | 1438.12 |
| 257 | 32 | 1417.46 | 1477.14 |
| 1024 | 32 | 1408.54 | 1479.97 |

Every trial passed bitwise comparison, sampled independent arithmetic checks,
and untouched-tail checks. Memcheck completed with zero errors. Racecheck was
canceled after the regression was established; it is **not** a pass. Subsequent
sanitizer stages have not been claimed.

The approximately 2.6–5.1% regression at 32 selected rows prevents production
adoption of this shared-generator version. The preceding QKV/output isolated
improvements do not override this result. No end-to-end speedup is claimed.

## Next controlled experiment

The new permutation does not need the old row padding. A prepared experiment
reduces static operand storage from 27,648 to 18,432 bytes, changing both stage
and input/weight partition offsets consistently. The numerical fold and output
layout remain unchanged. The corrected r2 experiment passed the same eight
cases and three trials, but still regressed: at 1024 tokens its median was
1464.67 us against paired baseline 1410.71 us. Padding removal alone does not
explain or fix the regression. The r1 scratch transformation changed only the
first producer offset occurrence, failed bitwise comparison at two tokens,
and was rejected; r2 changes both producer and consumer offsets.

The integrated generator therefore retains the original WMMA implementation
for `QueryRowEnds` demand. This is a demand-based lowering decision, not a
model-name special case. Its exported head source differs from the pinned
baseline only by one blank line; QKV is byte-identical to the previously
measured swizzle. The projection package now passes 34 native tests, including
producer-vector, fragment-read, and result-write/read bank-map checks.

Remote result directory:
`/run/user/1000/lunaflux-transport-head-20260909-r1`.
Runner: `/private/tmp/lunaflux-transport-head.mbtx`.
Compact results: `/run/user/1000/lunaflux-transport-head-compact-20260909-r2`.

## Down transport experiment

The down-only experiment changes the rejected compact operand map to
`row * 32 + (column ^ (((row / 2) & 3) * 8))`, keeping the gate/up entry
unchanged. It compares against the accepted kernel, not the rejected r5.
Three paired trials of thirty launches passed bitwise checks at
`1,7,17,63,64,65,255,256,257,504,1024` tokens.

| Tokens | Baseline median us | New median us |
| ---: | ---: | ---: |
| 256 | 49.75 | 45.16 |
| 257 | 64.04 | 55.37 |
| 504 | 114.20 | 106.80 |
| 1024 | 186.47 | 180.30 |

Registers increase from 116 to 128; static shared storage stays 24,576 bytes.
Counter and sanitizer validation remain outstanding; down is not integrated.
Remote artifacts: `/run/user/1000/lunaflux-down-transport-20260909-r1`.
No all-kernel or all-shape conflict-zero claim is made.
