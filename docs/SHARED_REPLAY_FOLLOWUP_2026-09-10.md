# Aggregate shared replay follow-up

## Result

The aggregate-counter objective is **not fixed**. No production kernel was
changed. Two new measurements of the existing final gate/up CUBIN confirm
nonzero hardware totals while source-attributed shared excess remains zero.
A global-load cache-policy experiment did not improve the totals and is not
adopted.

Device: RTX 5060 Ti, UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`,
PCI `00000000:17:00.0`; idle before measurement. Existing gate/up replay:
T=1024, grid=768, block=512, 61 registers/thread, 24,576 bytes static shared.
The driver prints both old and new kernel attributes; the selected new kernel
is the 61-register variant. No live serving workload was profiled.

| Measurement | Hardware load total | Hardware store total | Source excess |
| --- | ---: | ---: | ---: |
| Previous final report | 57,286 | 484,410 | 0 |
| Unchanged repeat 1 | 56,102 | 478,418 | 0 |
| Unchanged repeat 2, additional sections | 60,073 | 478,726 | 0 |
| Global-load cache policy `cg` | 55,667 | 548,685 | 0 |

Reports use cache-control all, clock-control none, one selected launch.
The first repeat and cache experiment used five passes; the second repeat
used nineteen with memory, scheduler, warp-state and throughput sections.
These are not controlled-clock statistical confidence intervals.

## What the additional counters establish

Unchanged repeat 2 duration was 406.21 microseconds; tensor active 63.80%,
DRAM throughput 8.22%, L1/TEX throughput 48.13%. No local/shared spilling
requests were reported. Shared actual and ideal wavefronts both equal
12,189,696. Global theoretical excess is 589,824 sectors out of 10,235,904.

No eligible warp: 59.72% of scheduler cycles. Eligible warps/scheduler: 0.88.
Average warp cycles per issued instruction: 19.41, including long scoreboard
4.89, barrier 4.01, math-pipe throttle 3.03, wait 2.55, short scoreboard 2.12,
and MIO throttle 0.24. These are overlapping warp-latency components, not
additive fractions of application wall time or proof of replay causality.

NVIDIA explains that aggregate bank-conflict counters also include other
arbitration replays; source excessive wavefronts distinguish address-bank
serialization. See [NVIDIA's explanation](https://forums.developer.nvidia.com/t/shared-memory-bank-conflicts-and-nsight-metric/115731/15/).
That explanation is not proof of the exact cause of every residual here.

The cache experiment compiled the existing source with the same options plus
`-Xptxas=-dlcm=cg`, into a separate CUBIN. Shared geometry is unchanged, but
branch instruction count changed from 6,672,384 to 7,274,496: the generated
instruction schedule is not otherwise identical. Thus this experiment rejects
that particular compiler-option change, not every possible cache/arbiter
hypothesis. No independent correctness or timing promotion was performed for
this rejected experiment.

## Remaining work

Instruction-isolated producer/consumer and synchronization ablations are
needed before changing the generic lowering. Preserve zero source excess,
compare hardware totals and latency, and inspect emitted SASS. Do not use
artificial serialization merely to hide replay counts. No all-kernel claim
follows from this gate/up-only follow-up. The separate norm/sampling/decode
driver previously blocked by approval was not uploaded or run.

Reports under remote `/run/user/1000/`:

- `lunaflux-replay-20260910-gate-admin1.ncu-rep`
- `lunaflux-replay-20260910-gate-admin2.ncu-rep`
- `lunaflux-replay-20260910-gate-cg1.ncu-rep`

The unprivileged first attempt failed with `ERR_NVGPUCTRPERM`; subsequent
captures used sudo, as previous captures did. Driver permission policy,
production services and original artifacts were not changed.
