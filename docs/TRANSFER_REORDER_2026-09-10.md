# Gate/up transfer rescheduling experiment

## Outcome

Real-operand rescheduling reduced the T=1024 gate/up kernel from about 559 us
to 490 us (approximately 12.3% less time), while retaining zero hardware
shared load/store conflicts in both profiled launches. This is a kernel-only
experiment, not an end-to-end improvement or an all-shape zero-conflict claim.
Production lowering remains unchanged in this experiment.

| Schedule | Normal T1024 time (us) | Hardware load/store, two samples |
| --- | ---: | --- |
| Full-register phased control | 559.17–559.22 | 0 / 0 |
| Compute shared addresses just before publishing | 548.99–549.61 | 0 / 0 |
| One register batch at a time | 630.37–632.69 | 0 / 0 |
| Late addresses, reuse outer publish-completion barrier | 489.52–491.18 | 0 / 0 |

All variants pass the existing paired 11-length bitwise comparison and
T1024 racecheck/synccheck. Source-attributed excessive shared wavefronts are
also zero. Only T1024 has fresh hardware-counter coverage for these reordered
variants; other lengths require counters before replacing the prior
ten-length-tested production lowering.

## Transformation

The successful variant retains the barrier between global reads and shared
publishing. It removes only the stage lambda's final CTA barrier. The existing
initial barrier still precedes the first consumption; the existing loop-end
barrier completes publication before the next iteration consumes that stage.
Next-stage publishing and current-stage consumption address different slots.
The original reduction order, operands, output epilogue and launch geometry
are unchanged. Delaying address construction also shortens address-register
lifetime across the read/publish barrier.

The single-batch variant doubles the number of phase boundaries and is slower.
These results support synchronization placement as a useful lever, but do not
quantify independent register/occupancy costs. The approximately 390 us older
nonzero-conflict path has not been recovered; 490 us is still roughly 26%
above that historical number. The probe's `old_us` near 499 us denotes its
older r5 reference binary, not the 390 us path; do not mix those baselines.

## Reproduction

Experiments are preserved on the target GPU host in:

- `/run/user/1000/lunaflux-transfer-reorder-20260910-r1`
- `/run/user/1000/lunaflux-transfer-reorder-20260910-r2`

Versioned `.mbtx` drivers with matching names are alongside those directories.
They use the MoonBit agent workflow and passed local warning-denied build-only
validation. Each directory contains generated CUDA, CUBINs, numerical/timing
logs, sanitizer logs, NCU reports and source/detail summaries.

Device: RTX 5060 Ti, sm120, UUID
`GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`. Compilation uses CUDA 13.1,
`-O3 --fmad=false --maxrregcount=128`. NCU uses SourceCounters and SpeedOfLight,
explicit shared load/store hardware totals, cache-control all and
clock-control none. Normal timings above are not NCU durations; three trials
are descriptive measurements, not a statistical confidence interval.
