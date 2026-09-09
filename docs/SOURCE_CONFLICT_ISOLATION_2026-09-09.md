# Instruction-level conflict isolation

This is a diagnostic boundary, not an all-kernel completion or deployment.
The selected 1024-token launch is profiled separately from event timing on
RTX 5060 Ti UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6` with CUDA 13.1,
sm120, full Nsight Compute counters and unlocked clocks.

## Correct the measurement target

`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` is not a pure count of
solvable address conflicts. NVIDIA explains that hardware counters include
other arbitration replays and recommends summing source-view **L1 Wavefronts
Shared Excessive** to isolate bank conflicts. Wider-access ideal wavefronts
must not be counted as conflicts.

Source: [NVIDIA explanation](https://forums.developer.nvidia.com/t/shared-memory-bank-conflicts-and-nsight-metric/115731/15/).

Keep both measurements; do not erase aggregate counts or claim every hardware
replay disappeared. Never serialize a whole kernel merely to make that broader
counter zero. Production selection remains subject to numerical correctness
and performance after isolation is complete.

## Measured source coverage

| Selected kernel/version | Executed shared instructions | Copy excess | Other shared excess | Status |
| --- | ---: | ---: | ---: | --- |
| QKV row-preserving XOR | 9,437,184 | 0 | 0 | Previously measured faster; integrated in a3f3179 |
| Output row-preserving XOR | 4,653,056 | 0 | 0 | Previously measured faster; integrated in a3f3179 |
| Down synchronous-copy isolation | 4,390,912 | 0 | 0 | Bitwise checks pass; slower experiment only |
| Attention explicit-fragment r2 | 7,767,040 | 0 | 0 | Prior numerical mismatch remains; experiment only |

These are individual profiled launches, not coverage of every token length,
schedule or kernel. Gate/up has its separately recorded selected-path result.
Head, normalization, sampling, decode and fallback shapes still require the
same source-level audit. Missing reports are not zero.

## Down: copy mechanism is a separate axis

All three new down variants passed bitwise checks for
`1,7,17,63,64,65,255,256,257,504,1024`, with three paired event-timing trials
and thirty launches per timing.

| Transport | Paired baseline median us at 1024 | Experiment median us | Source copy excess | Aggregate hardware conflicts |
| --- | ---: | ---: | ---: | ---: |
| XOR, async .cg | 186.47 | 180.30 | 1,572,864 | 7,211 |
| XOR, async .ca | 186.34 | 188.79 | 1,572,864 | 815,161 |
| XOR, synchronous uint4 + shared vector store | 186.23 | 272.05 | 0 | 91,030 |

Non-copy source excess was zero in each profile. The cache-policy change alone
does not fix the copy-side excess. Synchronous transport removes it but is
about 46% slower than its paired baseline. Keep it as the conflict-free
reference for subsequent producer/layout/consumer optimization, not as a
production replacement. Sanitizer qualification of this isolated variant is
still outstanding. Profiler durations are not substituted for event timings.

Attention r2 previously passed its independent numerical oracle but differed
from the old kernel in four BF16 results at 33 tokens. This profiling run does
not resolve that numerical mismatch or waive the acceptance requirement.

## Reproduction

`scripts/summarize-shared-conflicts.mbtx NCU_EXECUTABLE REPORT.ncu-rep`
reads one source report, resolves counters by column name, separates copy and
other instructions, and rejects missing/truncated coverage. Run its
`--self-test` mode locally. It is diagnostic tooling, not runtime validation.

Remote reports:

- `/run/user/1000/lunaflux-transport-swizzle-20260909-r1/ncu-qkv.ncu-rep`
- `/run/user/1000/lunaflux-transport-swizzle-20260909-r1/ncu-output.ncu-rep`
- `/run/user/1000/lunaflux-down-transport-20260909-r1/ncu-down.ncu-rep`
- `/run/user/1000/lunaflux-down-copy-isolation-20260909-r1/ncu-down.ncu-rep`
- `/run/user/1000/lunaflux-down-sync-isolation-20260909-r1/ncu-down.ncu-rep`
- `/run/user/1000/lunaflux-all-attention-20260909-r2/ncu-conflict-only.ncu-rep`

The synchronous down source, binary, timing logs and report were downloaded to
`/private/tmp/lunaflux-down-sync-isolation-results-20260909-r1`.
