# Gate/up transfer rescheduling experiment

Follow-up: [residency isolation and the 356-us schedule](SIBLING_RESIDENCY_2026-09-10.md)
separates Source address-layout excess from hardware arbitration totals. The
numbers and hardware-zero constraint below describe the earlier r1–r3 work,
not the latest optimization objective or validation status.

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

## Attempt to recover 390 us

The follow-up r3 campaign tried moving next-stage transfer after current-stage
MMA and lowering the compiler register budget. No production code was changed.

| Variant | T1024 time (us) | Hardware load, two samples | Hardware store, two samples |
| --- | ---: | --- | --- |
| Deferred transfer, default register budget | 501.37–504.96 | 0, 0 | 0, 0 |
| Prior 490-us schedule, register budget 64 | 394.33–394.44 | 79,045; 80,132 | 585,952; 593,029 |
| Deferred transfer, register budget 64 | 371.32–371.50 | 74,035; 75,836 | 696,343; 698,862 |

The low-budget prior schedule actually uses 60 registers rather than the
default schedule's 80, with zero stack or spill bytes in ptxas output. It is
numerically correct but fails the user's hardware-zero constraint. The faster
371-us combination fails that constraint too. All four r3 configurations
(including the repeated default control) pass paired numerical comparison,
racecheck and synccheck; source-attributed excessive wavefronts remain zero.

This sharp tradeoff strengthens the need to isolate occupancy/concurrent CTA
effects from source bank-address collisions. Register allocation changes
instruction scheduling as well as residency eligibility, so it does not yet
prove an exact hardware arbitration mechanism. In particular, earlier zero
totals must not be described as a layout-only fix.

The best measured zero-total schedule remains approximately 490 us. We have
not achieved 390 us with zero totals. The faster nonzero variants are retained
only as experiments in `/run/user/1000/lunaflux-transfer-reorder-20260910-r3`,
with its adjacent versioned `.mbtx` driver and ptxas resource logs.
