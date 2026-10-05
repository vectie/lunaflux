# AKO: longer KV histories versus larger MLP chunks

## Result and scope

The previously selected c322 attention route retains its advantage over the
older wide fallback at 8K/16K/32K history. That confirms an existing route win,
not a new improvement over today's c322 serving kernel.

The older r64-c8 MLP ownership alternative becomes consistently faster at an
8192-token chunk: initial whole-chain paired reductions are 6.59% cached and
6.07% with distinct layer-weight allocations. Five additional processes give
6.10–6.24% median paired reductions; all 25 pairs are positive, but one gains
only 2.58%. Therefore the strict all-pairs 3% rule is not completely met.
The newer resource-bounded coowned C16 alternative still has no robust gain.

**Serving selection and its 2048-token prefill chunk remain unchanged.** These
are isolated kernel/MLP-chain measurements, not a newly qualified 8192-token
serving bundle, end-to-end throughput, or fresh vLLM/SGLang comparison.

## The two independent dimensions

Attention depends on query count Q and retained KV history H. MLP depends on
the active query chunk Q, not H. A longer prompt processed in 2048-token chunks
does not automatically execute an 8192-row GEMM.

Attention uses fixed **total Q=2048**, rows 1 or 2, and per-request uniform
H=[8192,16384,32768]. At two rows each request contributes 1024 queries. The
32768-history cells exceed Qwen3-0.6B's 32768 context once queries are added;
they are valid bounded kernel probes, **not extended model-context support**.
The extra tail cell has Q=1792, rows=1, H=30720 (32512 total positions).

MLP uses Q=[2048,4096,8192], runtime row-count metadata 32, hidden width 1024,
intermediate width 3072, BF16, strict ordered folds. All five AOT alternatives
are exported with the same 8192-token capacity and their own launch geometry.
This common larger launch envelope differs from the earlier 2048-capacity
campaign; do not mix absolute timings across campaigns.

Cached weights and 28 distinct gate/up/down layer-weight allocations are
separate workloads. Distinct allocations contain identical deterministic
values; they model address working sets, not real model accuracy or diverse
layer-value distributions. Complete output and intermediate workspace must
match bitwise; a sampled independent scalar oracle must also pass.

## Fixed-query attention measurements

Median unprofiled GPU microseconds converted to milliseconds. Five alternating
paired trials per cell. Comparison: frozen wide c2001 versus frozen c322, with
each route's recipe and actual query-tile launch geometry. Both have the same
semantic work and independent sampled FP64 attention oracle.

| History | Rows | Wide fallback, ms | c322, ms | Median paired reduction |
| --- | ---: | ---: | ---: | ---: |
| 8192 | 1 | 10.443 | 2.369 | 77.20% |
| 8192 | 2 | 9.874 | 2.250 | 77.21% |
| 16384 | 1 | 19.413 | 4.438 | 77.11% |
| 16384 | 2 | 18.911 | 4.408 | 76.61% |
| 32768 | 1 | 37.837 | 8.895 | 76.51% |
| 32768 | 2 | 38.888 | 8.911 | 77.09% |
| 30720, Q=1792 tail | 1 | 30.810 | 7.298 | 76.32% |

Every cell clears the all-pairs 3% criterion. Longer history increases both
routes' cost approximately linearly at fixed Q; the relative route advantage
is stable rather than growing into an additional new optimization.

## Complete MLP-chain paired reductions

Positive means lower time. Values are medians of paired reductions, not ratios
of independently sorted latency medians. Isolated gate/up results do not
replace the whole-chain decision; down source remains identical in every
alternative.

| Alternative | 2048 cached / distinct | 4096 cached / distinct | 8192 cached / distinct |
| --- | ---: | ---: | ---: |
| r64-c8 | +3.87% / +4.53% | +4.35% / +5.28% | +6.59% / +6.07% |
| r128-c16 | −0.39% / −1.77% | +0.86% / +0.85% | +1.42% / +1.19% |
| r128-c8 | −4.07% / −8.20% | +0.63% / +0.39% | −0.21% / +0.51% |
| r64-coowned-c16 | +6.80% / +0.98% | +0.72% / +0.67% | +0.77% / +1.03% |

Only r64-c8's initial 8192 whole-chain cells clear the all-pairs 3% rule.
For example, its distinct-layer chain falls from **2.731 to 2.562 ms**;
gate/up falls from **1.867 to 1.716 ms**. At 4096 the favorable medians still
have weakest pairs below 3%; at 2048 variation remains substantial.

### Five independent r64-c8 / 8192 / distinct-layer processes

Each process contains five alternating whole-chain pairs plus separately
measured gate/up and down. The unchanged down entry point differs by only
noise-scale medians in these confirmations.

| Process | Baseline / alternative median ms | Median paired reduction | Weakest pair | Decision |
| --- | ---: | ---: | ---: | --- |
| 0 | 2.734 / 2.564 | 6.23% | 4.94% | improved |
| 1 | 2.725 / 2.560 | 6.19% | 4.82% | improved |
| 2 | 2.729 / 2.560 | 6.10% | 5.95% | improved |
| 3 | 2.735 / 2.564 | 6.16% | 2.58% | inconclusive |
| 4 | 2.741 / 2.596 | 6.24% | 3.02% | improved |

This is positive evidence for a larger-query domain, not permission to select
it across all token shapes or increase serving chunk size without testing the
other operators, workspace/graph capacities, scheduling and complete serving
latency.

## Compiler boundary and next decision

No production compiler policy or model semantics changed in this experiment.
The existing pure fold/distribution/ownership plans generate all alternatives;
CUDA resource qualifiers remain in CUDA lowering. No runtime JIT, model-name
branch, request-path validation or new hot-path allocation is introduced.

The diagnostic probe now admits at most 8192 query tokens and allocates input,
output and intermediate storage from the checked actual query count. The
exporter scales AOT operand bounds and verifies that both primary and down
carry the same ceiling, avoiding false timings from an early-returning old
2048-token artifact. Default exports remain at 2048.

The finite search budget was four existing alternatives, three query sizes,
two weight modes, then five additional processes for the promising 8192
alternative. No blind schedule sweep or production selector change follows.

## Validation and retained measurement

Projection package tests 107/107 and paired-trial parser tests 3/3 pass.
Affected native checks pass with the preexisting migration warning exclusions
`-79-29-25-20-92-14`. Scoped `moon info --target native` completes with zero
errors and 912 preexisting dependency migration warnings; this is not a
warning-clean whole-tree release claim.

GPU: GB10 sm121, UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.
CUDA 13.0.88 executable SHA-256:
`fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.
GPU work is serialized, with user units configured MemoryMax=8G,
MemorySwapMax=0 and RuntimeMaxSec=1200, plus measured before/after
MemAvailable reserve of at least 32 GiB. Do not interpret the very small
systemd-reported process-memory peak as actual CUDA/unified-memory usage.

Attention remote: `/home/wlc004s/lunaflux-ako-long-history-20261005.XFya41dp`.
Local archive: `benchmarks/qwen3_comparison/results/ako-long-history-20261005.gXo6YCWA/measurement.tar.gz`.
SHA-256: `75f9ed5daf9fc9f2b75e8ae6ced86fdf17bd59ff7a032adc48b7cae6a825b3a6`.
All **77 manifest files** verified locally after a non-overwriting download.

MLP remote: `/home/wlc004s/lunaflux-ako-large-chunks-20261005.u39N3uQA`.
Local archive: `benchmarks/qwen3_comparison/results/ako-large-chunks-20261005.3Lkul0uB/measurement.tar.gz`.
SHA-256: `f9aaeaec0bdf92177ed14686a3259226cc0e58020bec02a651bedc9985625670`.
All **322 manifest files** verified locally after a non-overwriting download.

All 24 initial MLP cells and five independent confirmation processes pass full
bitwise output/workspace comparison and the sampled scalar oracle (reported
error zero). The 8191-token tail passes memcheck, racecheck and synccheck:
zero errors, zero hazards/warnings, zero leaked bytes. No earlier whole-serving
sanitizer startup issue is resolved by this isolated MLP test. The counter CSV
parser regression passes 1/1; remote native driver compilation reports an
existing async-library C warning about an ignored `write` return value.

The new 8192-capacity source hashes are:

- Control: `d580e15fad64623ff473053b756ce28be426c11dc3a7caddeb36bda24f0d2acc`.
- r64-c8: `100b9d46669ce16854939402089acb3c7838b1763e6790746a56db0d2a26243f`.
- Coowned C16: `bb5eb4377b22c8ba680613735f99b4127dddb636fdca97fe1e8991cc92e5c1c1`.

### Fresh 8192-token selected-primary counters

Separate administrator-bounded Nsight replay, not the unprofiled whole-chain
timing above. Each alternative has its own paired control; do not merge the
controls into an imagined simultaneous measurement.

| Metric | Control / r64-c8 | Control / coowned C16 |
| --- | ---: | ---: |
| Warp instructions | 372080640 / 230031360 | 372080640 / 278495232 |
| Registers/thread | 58 / 102 | 58 / 60 |
| Threads/CTA | 512 / 256 | 512 / 512 |
| Waves/SM | 64 / 64 | 64 / 64 |
| Active warps/scheduler active cycle | 7.92 / 3.95 | 7.92 / 7.86 |
| Eligible warps/scheduler active cycle | 1.02 / 0.39 | 1.02 / 0.69 |
| Issue-active | 41.57% / 26.91% | 41.19% / 30.75% |
| Average warp latency/instruction issued | 19.04 / 14.66 cycles | 19.22 / 25.56 cycles |
| Barrier ratio per issue-active | 4.01 / 1.78 | 4.00 / 5.94 |
| Long-scoreboard ratio per issue-active | 1.25 / 3.92 | 3.74 / 4.04 |
| Tensor activity, elapsed-normalized | 43.98% / 45.79% | 44.29% / 44.71% |
| Local spilling requests | 0 / 0 | 0 / 0 |
| Replay primary time | 2.252 / 2.156 ms | 2.233 / 2.216 ms |

r64-c8 removes **38.18%** of warp instructions and lowers normalized barrier
ratio while retaining the same 6144 primary CTAs. Its smaller workgroup has
half the active warps, and fewer eligible warps; instruction savings translate
only partially into elapsed time, but tensor activity improves slightly.
This measured resource trade-off is compatible with the positive larger-chunk
timing result; it does not prove that every smaller chunk improves.

Coowned C16 removes **25.15%** of instructions but issue-active falls about
**25.34%**, eligible warps decline and average warp latency rises. Tensor
activity is almost unchanged, matching its noise-scale chain benefit. These
normalized ratios are not additive milliseconds or source-correlated proof of
a particular load dependency. Baseline long-scoreboard ratios vary across the
two captures; treat the counter explanation as bounded, not exact causality.

The appropriate next serving experiment is a complete, capacity-consistent
larger-query bundle with shape-specific selection and measured total latency.
Do not just switch today's 2048-token route to this alternative and claim the
8192-only gain, or increase chunk capacity in only the MLP artifact.
