# Shared projection transport: selected-row regression

Update: the original regression below is resolved by jointly changing the
bounded selected-row transfer width to 64, using compact row-XOR operand
storage, and loading explicit fragments with `ldmatrix`. The original K32
experiments remain useful negative controls; their regressions are not hidden.

## Joint selected-row solution

Three paired trials (20 graph launches each), same baseline and GPU as below:

| Tokens | Selected rows | Baseline median us | Joint K64 median us |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 737.13 | 736.66 |
| 2 | 2 | 1231.00 | 733.83 |
| 7 | 7 | 1236.19 | 739.50 |
| 8 | 8 | 1237.38 | 740.50 |
| 17 | 17 | 1363.50 | 815.24 |
| 32 | 32 | 1397.42 | 830.14 |
| 257 | 32 | 1422.66 | 829.72 |
| 1024 | 32 | 1405.49 | 827.56 |

All cases pass bitwise, sampled independent arithmetic and untouched-tail
checks. This is approximately 41% less head time at 32 selected rows, not an
end-to-end serving speedup. A single 1024-token launch passes memcheck,
racecheck, initcheck and synccheck. Its source profile has 4,102,272 executed
shared instructions, zero copy excess and zero other shared excess.

The K32 `ldmatrix` control still regressed (about 1528 us at 1024 tokens).
Matrix-load substitution alone was insufficient: transfer geometry and layout
must change together. The pure compiler schedule uses four ordered microtiles
per transfer for bounded selected-row products with up to eight output tiles;
wider output products retain two, and the backend retains their existing
consumer. There is no model-name or vocabulary-size special case.

The integrated generator export differs from the measured K64 experiment only
in whitespace around two fragment-load declarations. Its own remote rebuild
was blocked by automatic upload review, including a retry citing the user's
existing source-upload authorization. Do not claim that rebuild or a new
end-to-end benchmark completed. Production deployment is unchanged.

Physical results and report:
`/run/user/1000/lunaflux-head-k64-20260909-r1`.
K32 control: `/run/user/1000/lunaflux-head-ldmatrix-20260909-r1`.

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

At that checkpoint the integrated generator retained the original WMMA implementation
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
