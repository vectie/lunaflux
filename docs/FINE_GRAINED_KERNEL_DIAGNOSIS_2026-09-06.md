# Fine-grained kernel diagnosis — 2026-09-06

This supplements the [four-engine comparison](BASELINE_BOTTLENECK_COMPARISON_2026-09-06.md).
It adds per-step attribution and per-launch distributions from the same fresh
traces, plus a NEW three-repeat QKV control experiment. It does not change
production kernels or claim a new end-to-end speedup.

## 1. C8 MLP exposes insufficient block-level parallelism

Actual short-input C8 steady launch geometry, not configured maximum capacity:

| Operation | LunaFlux total blocks × threads/block | vLLM blocks × threads/block |
| --- | ---: | ---: |
| Gate-up | 12 × 512 | 384 × 32 |
| MLP down | 16 × 512 | 64 × 32 |
| Attention output projection | 8 × 256 | 64 × 32 |
| QKV | 32 × 128, auxiliary operations fused | 256 × 32, projection only |
| LM head | 1187 × 256 | 9496 × 32 |

The trace's device table reports 36 SMs, 48 maximum warps/SM, 65536
registers/SM and 102400 shared-memory bytes/SM on the RTX 5060 Ti.
Twelve ordinary CUDA blocks cannot occupy more than twelve SMs simultaneously.
Thus gate-up has an explicit device-wide parallelism ceiling of one third of
the SMs for that launch, even before considering instruction and memory stalls.
Its 88 registers/thread × 512 threads also permits at most one such block per
SM from the register budget alone. This is not a measured occupancy percentage.

The corresponding gate-up median is 86.631 µs vs vLLM's 33.953 µs plus a
small separate activation kernel. Output projection and down have similar
under-distributed grids. These are concrete targets for output-tile/work
partition experiments, not proof that multiplying grid size alone is safe or
will yield a proportional speedup. LM head already has many blocks: its
problem cannot simply be attributed to having fewer blocks than SMs.

## 2. Distributions distinguish persistent cost from occasional outliers

Short-input C8, microseconds per launch. Quantiles are nearest-rank summaries
of the indicated kernel/geometry across one measured trace window, not
confidence intervals across independent runs. Context length changes during
decode, so attention's spread includes changing work, not just timing jitter.

| Kernel | Calls | Median µs | P95 µs | Maximum µs |
| --- | ---: | ---: | ---: | ---: |
| LunaFlux LM head | 258 | 2485.285 | 2487.942 | 2490.342 |
| vLLM steady C8 LM head | 256 | 740.886 | 742.709 | 759.766 |
| LunaFlux gate-up, activation fused | 7112 | 86.631 | 88.903 | 91.335 |
| vLLM gate-up, projection only | 7140 | 33.953 | 35.169 | 37.697 |
| LunaFlux QKV, auxiliaries fused | 7112 | 85.767 | 87.239 | 88.711 |
| vLLM QKV, projection only | 7140 | 22.080 | 22.593 | 23.841 |
| LunaFlux direct decode attention | 7112 | 55.268 | 87.623 | 91.591 |
| vLLM direct decode attention | 7140 | 21.056 | 30.785 | 32.513 |

All displayed LunaFlux launches have a nonzero CUDA graph ID. The vLLM LM-head
row has no graph ID yet is much faster. Therefore “enable CUDA Graph” does not
explain or solve this LM-head gap. Launch shape, library algorithm and memory
schedule remain different variables; graph membership is not a controlled A/B.

## 3. Each prefill chunk, not just the whole request

LunaFlux C1, 1528 input tokens, chunk capacity 256. Each row is one forward
step containing exactly 28 QKV layer launches, one LM head and one greedy
launch. Kernel sums, milliseconds:

| Chunk | Attention | MLP | QKV | LM head | Greedy | All kernels |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.702 | 12.338 | 6.620 | 0.738 | 0.143 | 25.503 |
| 2 | 5.936 | 12.358 | 6.630 | 0.739 | 0.143 | 28.774 |
| 3 | 9.027 | 12.358 | 6.615 | 0.740 | 0.142 | 31.850 |
| 4 | 12.196 | 12.359 | 6.626 | 0.738 | 0.142 | 35.027 |
| 5 | 15.323 | 12.360 | 6.618 | 0.740 | 0.143 | 38.152 |
| 6 | 18.156 | 12.381 | 6.586 | 0.740 | 0.144 | 40.972 |

Attention grows by roughly 3.1 ms per additional chunk while MLP/QKV remain
nearly constant. This separates growing KV traversal from fixed-size linear
work. It does NOT mean earlier query tokens are all recomputed: later queries
legitimately attend to more keys.

The first five chunks cannot emit the final prompt's first generated token,
yet each computes a full LM-head/greedy result. Their summed head + greedy
time is approximately **4.41 ms**. The completion path checks
`produces_token` before selecting/emitting a prefill token, but the measured
device launches have already happened. This is a specific output-demand
elimination candidate. Its measured budget is only ~2.2% of the 200.3 ms
prefill kernel time here; fixing it alone would not close the attention gap.
Skipping these operations needs an executor/graph variant that preserves KV
effects and handles mixed steps where other rows do produce tokens.

### C8 requires separating mixed steps from pure decode

Partitioning the 79 forwards by presence of the prefill-named attention kernel:

| Step class | Steps | Kernel ms | Attention ms | MLP ms | QKV ms | LM-head ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contains prefill, possibly mixed with decode | 48 | 1678.032 | 570.028 | 591.589 | 317.451 | 49.406 |
| Decode-only | 31 | 627.296 | 347.667 | 95.775 | 73.349 | 75.247 |

This shows that 72.8% of kernel time lies in **prefill-containing steps**,
not that 72.8% is pure prefill work. The matrix-tiled prefill emitter traverses
all `row_count` rows rather than only `prefill_count`; some of these steps
include already-decoding requests. The changing head time (about 0.74 to
2.48 ms) is another visible indication that these steps are not identical
single-request chunks. Inferring an exact live-row count from head duration
alone would be invalid.

Consequently, the earlier comparison's 570 ms / 84 ms / 49.5 ms attention
category is a **kernel-route workload comparison**, not isolated equal-work
prefill service time across engines. We need a request-phase/active-row marker
or an isolated prefill-only experiment to remove this scheduling confound.
Splitting mixed-row work between genuinely specialized prefill/decode routes
is a concrete experiment, not an already-proven speedup.

## 4. What grows during decode?

Same short-input C8 trace: average kernel milliseconds per forward over the
first and last 32 decode-only steps. This is not client TPOT.

| Category | First 32 | Last 32 |
| --- | ---: | ---: |
| All kernels | 9.7911 | 11.5096 |
| Attention | 0.6954 | 2.3970 |
| LM head | 2.4856 | 2.4849 |
| MLP | 3.1127 | 3.1228 |
| QKV | 2.3976 | 2.4043 |

About 99% of this window-to-window increase is in attention. The expensive
LM head is a nearly constant per-step tax; attention is the context-growth
tax. For C1 the corresponding attention average grows only 0.6480 → 0.8282
ms, under a different direct/split schedule mix. Neither comparison isolates
the effect of one partition threshold; that requires fixed-context A/B.

## 5. New QKV factorial microbenchmark

Same existing diagnostic QKV CUBINs, deterministic synthetic input, no serving
deployment. Three repeats, 500 launches captured into each warmed CUDA graph
and timed by CUDA events. Fixed-grid A/B order is reversed in repeat 2.
Full output and both KV buffers are compared byte-for-byte; all successful
cases have zero differences and zero non-finite active outputs. This is not
an independent model oracle or a sanitizer test.

The baseline zero-pads a partial input tile inside its output/K loops. The
hoisted variant materializes reusable padded input earlier. This also changes
shared memory 8704 → 40960 bytes and registers/thread 58 → 80, so the variant
does not isolate barriers alone. Both compiled variants report no spills.
Fixed grid X=16; compact grid X=ceil(active tokens/16), with the same CUBIN.
Means of three measurements, µs/launch:

| Tokens | Baseline fixed | Baseline compact | Hoisted fixed | Hoisted compact |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 27.671 | 19.454 | 21.459 | 19.702 |
| 8 | 59.593 | 56.001 | 34.106 | 22.195 |
| 15 | 76.279 | 71.852 | 36.972 | 24.856 |
| 16 | 48.554 | 33.264 | 32.194 | 21.516 |
| 17 | 65.287 | 55.009 | 32.763 | 31.881 |
| 32 | 63.573 | 40.085 | 36.441 | 35.689 |
| 59 | 96.809 | 80.816 | 64.063 | 64.124 |
| 128 | 113.978 | 115.116 | 121.027 | 120.762 |
| 256 | 211.065 | 211.310 | 227.168 | 227.203 |

Tokens 1/8/32 are separate single-token rows; other counts are one prefill
row. The 15→16→17 boundary is therefore the cleaner partial-tile comparison.

- At 8 tokens, the same compact launch improves 56.001 → 22.195 µs (~2.52×)
  with hoisting/resource changes. Compacting baseline alone saves only ~6%.
- At 1 token, hoisting after compacting is slightly worse, not a win.
- At 256 tokens, hoisting is ~7.6% slower; at 128 it is also worse.
- At 16 tokens, the full tile still improves. Therefore “padding removal”
  alone cannot explain the result; input reuse/resource scheduling matters.
- Serving C8 already uses compact QKV grid X=1. Do not credit fixed→compact
  savings to it again. Synthetic microbench latency is not its measured
  85.7 µs serving latency: weights/cache/interleaving and inputs differ.

The actionable compiler feature is a **shape-conditioned choice among legal
storage schedules**, with measured costs; not a global hoisting switch.

## 6. What is still missing?

We now have shape/geometry, per-launch distributions, per-step phase costs,
context growth and one controlled storage/grid experiment. We still do not
have fresh production-shape hardware counters for LM head/MLP/attention,
or an isolated same-runtime mixed-vs-separated attention experiment.
Existing QKV Nsight Compute reports use replay and synthetic fixed grids;
their occupancy/stall numbers must not be relabeled as counters for current
compact C8 serving. Kernel duration alone cannot distinguish DRAM traffic,
L2 reuse, shared-bank conflicts and scoreboard stalls conclusively.

Next discriminating measurements are: gate-up output tiling at fixed math;
LM-head active-row sweep with measured memory traffic; fixed-context attention
direct/split crossover; and mixed-row vs phase-separated execution with exact
row metadata. Output-demand elimination has a smaller, already bounded budget.

## Artifacts and validation

- Combined derived-data/script archive:
  `/private/tmp/lunaflux-fine-diagnosis-20260906-r1.tar.gz`, SHA-256
  `48ee536c5b973bfb000e6b201963f0d0a9938ccec05af8e67881b2e5eff2e795`.
  Raw Nsight traces remain in the previous four-engine archive; this smaller
  archive contains the new analyses and QKV results. Final GPU process query
  was empty after all diagnostic runs completed.
- Derived trace distributions and step CSV/SQL:
  `/private/tmp/lunaflux-fine-trace-20260906-r1`.
- New QKV controls and summary:
  `/private/tmp/lunaflux-fine-qkv-20260906-r2` (same basename under remote
  `/dev/shm`). Initial r1 used the older probe lacking compact CLI support and
  exited 2 before a GPU result; preserved, not counted. r2 uses existing
  `probe-v3` without recompilation.
- Scripts: `lunaflux-fine-trace-20260906.mbtx`,
  `lunaflux-fine-step-windows-20260906.mbtx`,
  `lunaflux-fine-qkv-20260906.mbtx`, and
  `lunaflux-fine-qkv-summary-20260906.mbtx`, under `/private/tmp`.
- Trace grouping checks: every LunaFlux step has exactly 28 QKV layer
  launches, one LM head and one greedy launch. No warmup enters these windows.
- Analysis and orchestration follow the MoonBit skill's `.mbtx` requirement.
  No production implementation changed; no full suite was needed for these
  diagnostic/documentation changes. Unrelated working-tree edits are preserved.
