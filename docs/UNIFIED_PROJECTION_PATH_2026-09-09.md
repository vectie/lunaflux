# Unified multirow projection path

The eligible dense QKV/output and sibling gate/up schedules now start at two
rows, not 256. Down uses K64 compact row-XOR operands, explicit matrix loads
and MMA, a permuted result map, and ceil-divided masked row tiles. Single-row
register/shuffle execution remains distinct. Hardware instructions remain in
CUDA lowering; schedule thresholds and transfer geometry remain in the pure
compiler plan. This is not a claim that unsupported shapes or all head
variants have acquired a new schedule: the existing matrix eligibility and
selected-row wide-product constraints remain.

## Fresh integrated measurement

GPU `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, CUDA 13.1, sm120,
O3, fmad=false, register cap 128. Baseline is the immediately preceding
current-source export, not a historical pre-optimization binary. Three paired
trials, 100 captured launches per timing; median microseconds, old → new.

| Tokens | QKV | Output | Gate/up | Down |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12.94 → 12.75 | 5.16 → 5.12 | 13.72 → 9.38 | 5.76 → 7.47 |
| 7 | 16.74 → 15.30 | 24.57 → 17.28 | 36.48 → 26.38 | 20.14 → 29.61 |
| 17 | 27.23 → 17.21 | 24.64 → 18.44 | 36.85 → 28.53 | 20.44 → 29.61 |
| 63 | 48.29 → 29.07 | 24.87 → 18.86 | 73.40 → 40.95 | 38.11 → 29.91 |
| 64 | 48.42 → 29.08 | 24.92 → 18.90 | 73.40 → 41.92 | 38.15 → 29.92 |
| 65 | 60.11 → 38.71 | 48.27 → 23.17 | 73.28 → 64.73 | 58.07 → 29.97 |
| 255 | 177.43 → 102.58 | 95.22 → 50.85 | 219.01 → 133.09 | 151.62 → 46.83 |
| 256 | 103.86 → 104.20 | 51.36 → 51.37 | 147.25 → 133.49 | 48.92 → 46.98 |
| 257 | 109.74 → 110.08 | 51.66 → 51.65 | 159.14 → 157.84 | 62.96 → 69.60 |
| 504 | 197.26 → 197.90 | 100.66 → 100.68 | 268.20 → 254.77 | 113.26 → 89.49 |
| 1024 | 385.45 → 386.86 | 187.12 → 187.16 | 535.00 → 500.22 | 188.21 → 174.50 |

All 44 family/length cases pass full-buffer bitwise comparison. All four
families pass memcheck, racecheck, initcheck and synccheck at 65 tokens (16
sanitizer runs). The affected native packages pass 69 tests, warning-denied
check, formatting and interface generation. The single-row
arithmetic route is unchanged, but whole-function resource allocation changes:
gate/up registers fall 92→63; down static shared rises 24576→32768 bytes and
registers rise 116→128. Those costs remain visible, including down regressions
at 1, 7, 17 and 257 tokens, as explicitly accepted for the new-path switch.
These are isolated operations, not end-to-end serving or fresh competitor runs.

Artifacts: `/run/user/1000/lunaflux-unified-path-20260909-r1`.
Baseline: `/run/user/1000/lunaflux-current-audit-20260909-r1`.
Head was recompiled but not retimed in this campaign; its eligible path is
unchanged. Previous experimental conflict measurements must not be relabeled
as fresh integrated all-kernel conflict coverage.

## Removal of emitted fallback code

For schedules with a new pipeline, the exporter now omits the superseded
multirow WMMA code instead of merely placing it after the new path's return.
Default QKV/output/head and paired MLP export tests explicitly reject any
`wmma::` call. Down no longer emits a redundant full-tile branch around its
masked pipeline. Single-row strided pairwise arithmetic remains unchanged.
Schedules without a new pipeline still have separate implementations; this
change is not repository-wide elimination of every older implementation.

Fresh paired graph measurements against the immediately preceding unified
version, three trials and 100 launches, median us (before → after removal):

| Tokens | QKV | Output | Gate/up | Down |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 15.35 → 15.37 | 17.31 → 17.30 | 26.41 → 26.48 | 29.61 → 29.60 |
| 255 | 103.67 → 103.79 | 51.37 → 51.37 | 133.28 → 133.25 | 47.33 → 47.29 |
| 1024 | 386.78 → 387.13 | 187.01 → 187.01 | 500.01 → 500.19 | 174.53 → 174.64 |

All 44 family/length comparisons pass bitwise checks; 69 affected native
tests and 16 sanitizer runs at 65 tokens pass. Down registers drop 128→113,
but timings are essentially unchanged.
No fresh all-kernel conflict-zero or end-to-end speedup claim is made.
Artifacts: `/run/user/1000/lunaflux-no-fallback-20260909-r1`.
