# Dual-Spark long-history exponential comparison — 2026-10-05

## Result and scope

Both Sparks ran the paired sweep concurrently. The already-implemented explicit
approximate exponential candidate reduces selected attention-kernel time by
4.37–5.36% median paired gain across six host/workload cells. Three cells pass
the predeclared criterion of at least 3% gain in every pair; three remain
inconclusive under that conservative criterion. No pair regresses in this
capture. This is neither a newly implemented compiler optimization nor an
end-to-end serving/model-quality claim.

Earlier C16/history4096 measurements were neutral or slower. These results
show why query/history/batch vectors matter; they do not justify a global switch.

## Frozen artifacts and geometry

- `.179`: GB10/sm121, UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.
- `.178`: GB10/sm121, UUID `GPU-9c3d3cf0-439a-5da2-67e9-20414255879f`.
- Baseline c322: `strict-natural-exponential-v1`, symbol
  `lunaflux_attention_prefill_tile_compiler_v1`; cubin SHA-256
  `9aedf7709b833ad4ba05bf017d54199b33908e89d63c7fae51e73f94cf0a655d`.
- Candidate c30322: `approx-base2-f32-v1`, `subnormal=flush-permitted`, symbol
  `lunaflux_attention_prefill_tile_compiler_exp2_v1`; exported with explicit
  `--prefill-approximate-exp2`. Source SHA-256:
  `638ecbaa0d58fddf4f2f8be6b568892c5203ca752226c09c9305e6b66873251a`.
- Candidate cubin SHA-256:
  `66881bc0055ce4c4340e6001e7746d0abd64a8b7f9a02b741e6b87463f0e2922`.
- nvcc13.0.88 SHA-256:
  `fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.

The baseline cubin was reused. Two candidate compilations produced identical
hashes. Both hosts used the same transferred artifacts/probe. Runtime geometry
uses query bucket2048, rows bucket32, grid63×16×1 and block128, not an idealized
small-active-row launch. The numerical law changes; launch geometry does not.

## Finite paired sweep

Five alternating pairs per cell, unchanged correctness/timing boundaries.
Gain is the median of paired gains, not a ratio of separately sorted medians.
Query is current chunk size; history is prior KV positions, not output tokens.

| Host | Query | Active rows | History | Strict median μs | Approximate median μs | Paired gain | Worst pair | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| .179 | 2048 | 1 | 8192 | 2418.532 | 2293.476 | 5.08% | 3.22% | Improved |
| .179 | 2048 | 1 | 28672 | 7738.805 | 7469.986 | 4.37% | 2.39% | Inconclusive |
| .179 | 2048 | 2 | 28672 | 7765.988 | 7425.148 | 4.39% | 1.84% | Inconclusive |
| .178 | 2048 | 2 | 8192 | 2204.256 | 2100.953 | 4.57% | 4.32% | Improved |
| .178 | 1792 | 1 | 30720 | 7014.875 | 6685.708 | 4.76% | 4.28% | Improved |
| .178 | 2048 | 2 | 28672 | 7529.438 | 7156.038 | 5.36% | 2.93% | Inconclusive |

The shared cell moves in the same direction on both hosts. Absolute host times
are not a paired inter-host comparison. Long-cell outputs are non-bitwise:
baseline/candidate maxabs0.000488281; every row passes FP64 oracle ceiling0.003.
This does not establish token parity or model quality. `.178` additionally passes
memcheck/racecheck/synccheck with zero errors/hazards at Q129/rows2/history128;
that scoped check is not long-history sanitizer admission.

## Matched counters: fewer instructions, modest speedup

`.179` captures exactly two launches for Q2048/rows2/history28672: strict then
approximate. Profiled times do not replace unprofiled paired results.

| Metric | Strict | Approximate |
| --- | ---: | ---: |
| Profiled kernel time | 8.472000 ms | 8.103392 ms |
| Executed warp instructions | 1,125,435,264 | 936,583,040 |
| HMMA BF16 instructions | 119,668,736 | 119,668,736 |
| Non-transposed / transposed LDSM | 37,396,480 / 29,917,184 | Same |
| Async LDGSTS instructions | 14,958,592 | 14,958,592 |
| MUFU.EX2 instructions | 31,787,008 | 31,787,008 |
| FFMA / FFMA.SAT / FFMA.RM | 64,229,376 / 31,787,008 / 31,787,008 | 655,360 / 0 / 0 |
| FADD instructions | 99,100,672 | 67,313,664 |
| Registers/thread | 235 | 234 |
| Register-limited resident blocks | 2 | 2 |
| Source-correlated excessive shared wavefronts | 0 | 0 |
| Eligible warps/scheduler/cycle | 0.380812 | 0.326852 |
| Issue-active percentage | 31.317% | 27.496% |
| Mean warp latency/issued instruction | 5.567197 | 6.472763 |
| Fixed dependency wait contribution | 36.51% | 37.90% |
| Long-scoreboard contribution | 20.55% | 14.89% |
| Barrier contribution | 2.18% | 2.37% |

Instructions fall about 16.8%, profiled time about 4.4%. Mathematical MMA/EX2,
fragment loads and async-copy counts remain unchanged. Removing strict-exp
support arithmetic does not remove the attention product/softmax dependency
chain. Issue activity falls despite shorter duration. Per-issued-instruction
latency/stall fractions also have changed denominators: a larger fraction does
not prove more total waiting. Remaining dependency scheduling/ownership merits
investigation; instruction count, occupancy and shared-conflict counts are not
standalone performance objectives.

## Decision and saved results

Keep the candidate as an explicit approximate alternative. Do not replace the
strict law, claim serving propagation, or add this kernel gain to the earlier
C2 decode serving gain. A full selected serving comparison with explicit law,
per-request output/timing vectors and model-quality checks remains necessary.

Each sweep uses user-systemd MemoryMax16G, MemorySwapMax0, RuntimeMaxSec600,
TasksMax64 and a 32GiB MemAvailable reserve before/after every cell. Minimum
recorded available memory exceeds 116GiB. The two hosts overlap, one GPU job per
host. Sanitizer and counters also overlap on different hosts. Both GPUs finish
idle; no production route, old artifact or driver setting changes.

- `.179`: `/home/wlc004s/lunaflux-ako-long-exp-20261005.zYjETdvS/experiment`.
- `.178`: `/home/wlc003s/lunaflux-ako-long-exp-20261005.YDjCs0sP/experiment`.
- Local summaries/counters/report/archives:
  `/tmp/lunaflux-ako-long-exp-20261005.C3xnaR9t`.
- `.178` archive SHA-256:
  `df4b60a5764822b90e583ecb0a8708b64fe11136001b92a610642776e8ee839d`.
- `.179` archive SHA-256:
  `756a191cda8cabba6ae196bdbfe01b9df75b5ece6aff4eb19ad7890ddef94459`.

Both archive hashes match remote values; all 1,092 manifest entries verify
locally. The MoonBit adapter/reporter pass warning-denied native script checks
and focused tests. An initial unsupported StringBuilder.clear call in the
reporter was corrected and tested. No aggregate release-suite pass is claimed.
AKO guided the finite budget, paired measurements and explicit non-bitwise
decision. Production compiler/runtime code is unchanged.
