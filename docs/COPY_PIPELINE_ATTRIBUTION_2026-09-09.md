# Attribution of the rejected shared-layout regression

This analysis reads existing full Nsight Compute reports for the accepted
projection implementation and rejected r5 experiment. Production kernels are
unchanged. It localizes the regression to asynchronous operand transport, not
merely the number of arithmetic/address instructions.

## Matched selected 1024-token kernels

| Metric | QKV accepted → r5 | Output accepted → r5 | Down accepted → r5 |
| --- | ---: | ---: | ---: |
| Paired event median, µs | 415.98 → 634.61 | 202.88 → 329.74 | 186.38 → 367.61 |
| Executed instructions | 77,791,232 → 80,691,200 | 34,516,992 → 35,274,752 | 15,178,752 → 14,974,976 |
| Async-copy global read sectors | 12,582,912 → 25,165,824 | 6,291,456 → 12,582,912 | 6,291,456 → 12,582,912 |
| L2 TEX read requests | 5,505,238 → 25,166,023 | 2,752,573 → 12,582,967 | 3,932,204 → 12,582,961 |
| L2 throughput, % of peak sustained elapsed | 37.94 → 98.48 | 38.73 → 91.59 | 51.66 → 85.86 |
| Long-scoreboard cycles per issued instruction | 0.318 → 4.798 | 0.092 → 5.543 | 1.949 → 15.915 |
| Barrier cycles per issued instruction | 0.764 → 1.365 | 0.923 → 2.052 | 0.649 → 1.419 |
| MIO-throttle cycles per issued instruction | 0.379 → 1.146 | 0.456 → 1.894 | 0.576 → 1.945 |

Event timings and profiler counters are separate measurements. GPU clocks were
not locked; the exact Nsight durations differ from event medians. Stall ratios
are not percentages of wall time and must not be summed into a latency budget.

The instruction-count changes are approximately +3.7%, +2.2%, and -1.3%.
In particular, down becomes much slower despite executing fewer instructions.
Instruction count alone cannot explain these regressions.

## Exact QKV instruction attribution

`LDGSTS.E.BYPASS.128.ZFILL` executes 786,432 times in both reports. Its attributed
global sectors double from 12,582,912 to 25,165,824. Nsight's source-correlated
excessive-global-sector count changes from zero to 12,582,912. This is the copy
operation itself, not additional tensor work or a local-memory spill.

The same copy instructions have source-correlated **excessive shared wavefronts**
of 2,359,296 in the accepted version versus 44,040,192 in r5. This is a distinct
metric from `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`, whose much lower
r5 value previously obscured the producer-side transport cost. These metrics
must not be added together or treated as interchangeable bank-conflict counts.

The recurring r5 copy instruction is at report PC `0x756907771ab0`; the following
`WARPSYNC.ALL` at `0x756907771f30` carries 21,004 long-scoreboard samples. The
accepted report's corresponding recurring wait has 666 samples. Raw sample
counts are corroborating locations, not normalized causal time measurements.

QKV's DRAM throughput is only 3.76% in r5 while L2 reaches 98.48%. Its L2 read
miss sectors remain 327,681 in both reports. This points to inefficient cached
transport/request amplification, not a larger off-chip data footprint.

## Why the layout is implicated

The rejected source retains logical producer assignment `element = vector * 8`
but stores into compact 8×8 microtiles. For one logical row, vectors at columns
0, 8, 16, ... now target shared byte offsets 0, 128, 256, ... instead of adjacent
16-byte destinations. This changes the joint global-to-shared copy mapping even
though each lane still copies 16 bytes and the global logical addresses are
unchanged. The consumer-friendly mapping was not assessed for producer-side
transaction efficiency.

Together, the copy-specific sector amplification, shared wavefront increase,
L2 request pressure and waiting-instruction attribution strongly identify the
operand-copy path as the principal regression mechanism. They do not establish
an exact percentage of latency attributable to each cause. A controlled
single-variable ablation remains needed before selecting a replacement layout.

## Compiler implication

Layout selection must evaluate the composition of **producer mapping, physical
layout and consumer mapping**, not only consumer bank separation. A pure layout
plan can retain logical semantics while estimating/checking:

- global sector utilization and shared destination grouping for copies;
- consumer matrix-load wavefronts;
- address and register-rearrangement instruction cost;
- resource occupancy and synchronization dependencies.

CUDA-specific transaction constraints belong in backend lowering. Functional
semantics and the ordered numerical reduction do not need to change.

## Scope and reproducibility

Accepted reports:
`/run/user/1000/lunaflux-counter-fixes-20260909-r1/ncu-{0,1,3}-2.ncu-rep`.
Rejected reports:
`/run/user/1000/lunaflux-all-bank-20260909-r5/ncu-{qkv,output,mlp}.ncu-rep`.
Read-only analysis scripts:
`/private/tmp/lunaflux-memory-attribution.mbtx` and
`/private/tmp/lunaflux-copy-attribution.mbtx`.

The proposed scratch `.cg` → `.ca` cache-policy ablation was **not uploaded or
run**: upload was rejected by the permission reviewer. No ablation result is
claimed. Attention and other kernel families are not covered by this matched
projection attribution and must not inherit its numerical conclusions.
