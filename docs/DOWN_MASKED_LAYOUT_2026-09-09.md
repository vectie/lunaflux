# Down projection: masked transport coverage

The zero-conflict row64/K64 experiment still dispatched short inputs and
partial row tiles to the old WMMA fallback. Profiling only 1024 tokens missed
that path. The following physical experiment uses the same compact row-XOR
producer and explicit matrix-load consumer for every multirow input, with
ceil-divided row tiles, zero-filled invalid input rows and masked output rows.
Single-token execution retains the register/shuffle path and executes no
shared-memory instructions. This is an experimental CUDA export; it has not
yet been integrated into the general compiler or deployed.

## Source-correlated conflicts

CUDA 13.1, sm120, GPU `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`.
Each measurement profiles one launch. Counts below are source-correlated
excessive shared wavefronts, not aggregate hardware arbitration replays.
Copy excess is zero in both versions at all ten lengths.

| Tokens | Previous experiment: other excess | Masked: other excess |
| ---: | ---: | ---: |
| 7 | 512 | 0 |
| 17 | 1024 | 0 |
| 63 | 2048 | 0 |
| 64 | 2048 | 0 |
| 65 | 2560 | 0 |
| 255 | 8192 | 0 |
| 256 | 0 | 0 |
| 257 | 512 | 0 |
| 504 | 2048 | 0 |
| 1024 | 0 | 0 |

The single-token profile has no measured shared instructions and is recorded
as shared-memory-free, not passed through the positive-coverage summarizer.
These results do not establish every possible model dimension or GPU target.

## Paired performance and correctness

Three trials, 100 CUDA-graph launches per timing, median microseconds. The
timing baseline is the accepted original down kernel, not the preceding
row64/K64 experiment used in the conflict table.

| Tokens | Original | Masked |
| ---: | ---: | ---: |
| 1 | 5.73 | 7.43 |
| 7 | 20.03 | 29.47 |
| 17 | 20.44 | 29.61 |
| 63 | 38.04 | 29.85 |
| 64 | 38.06 | 29.85 |
| 65 | 57.96 | 29.89 |
| 255 | 151.63 | 46.84 |
| 256 | 48.70 | 46.83 |
| 257 | 62.90 | 69.51 |
| 504 | 112.69 | 89.09 |
| 1024 | 187.16 | 174.39 |

All eleven lengths pass full output-buffer bitwise comparison, including the
unchanged tail. The masked 65-token launch passes memcheck, racecheck,
initcheck and synccheck. This is differential correctness, not a new
independent arithmetic oracle. Single-token code is unchanged, but increased
static shared allocation affects its launch resources. Padding short batches
to 64 rows also adds real work. The user explicitly accepts a slowdown to
eliminate conflicts; these costs must nevertheless remain visible.

## Remaining integration

Move the masked row-coverage policy into the immutable compiler schedule and
the operand/result maps into CUDA lowering, with exhaustive map and boundary
tests. Remove the conflicting fallback from that schedule rather than leaving
it reachable on ragged inputs. Re-export and repeat validation on the actual
integrated compiler output. No production completion is claimed here.

Remote artifacts:

- `/run/user/1000/lunaflux-down-shape-counters-20260909-r1`: single-row profile;
- `/run/user/1000/lunaflux-down-shape-counters-20260909-r2`: old dispatch;
- `/run/user/1000/lunaflux-down-shape-counters-20260909-r3`: masked dispatch;
- `/run/user/1000/lunaflux-down-masked-20260909-r1`: source, binary, paired timing;
- `/run/user/1000/lunaflux-down-masked-sanitize-20260909-r1`: four sanitizers.
