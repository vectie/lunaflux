# DeepSeek original first-layer component replay — 2026-10-11

The final retained-state discrepancy is not localized by agreeing greedy tokens.
The bounded offline replay starts earlier, with the same 144 literal prompt IDs
and original first-layer checkpoint weights on Spark `.179`. The installed
original `Block.hc_pre`, `kernel.hc_split_sinkhorn`, and `RMSNorm` execute without
replacement arithmetic. No complete model or state dictionary is instantiated.
Original output files retain every operand consumed by the exported native
hyper-connection envelope. Production execution is unchanged.

| Matched original first-layer boundary | Values | Exact values | Relative L2 | Maximum absolute error |
| --- | --- | --- | --- | --- |
| Collapsed BF16 residual | 589,824 | 589,824 | 0 | 0 |
| Learned BF16 normalized input | 589,824 | 589,824 | 0 | 0 |
| F32 post controls | 576 | 0 | 0.00059055% | 4.09782e-7 |
| F32 Sinkhorn combination | 2,304 | 145 | 0.00001397% | 5.36442e-7 |

This rules out the **initial** collapsed/normalized input as the source of the
large observed pre-head difference. It does not rule out later hyper-connection
states, quantify propagation, or prove query/attention/MLP correct. Initially
duplicated embedding streams also make this easier than later distinct residual
streams; subsequent matched boundaries remain necessary. No tolerance or
model-specific arithmetic policy is introduced.

The original unit/container exits zero with `OOMKilled=false`, and the native
replay plus CUDA memcheck/leak, racecheck and synccheck complete with zero errors.
Exact user-unit retirement is awaited before each GPU launch, including imports
that precede CUDA ownership. Both paths release GPU owners. Original replay is
bounded by 8-GiB/no-swap limits, native by 2 GiB; these are external host limits,
not measurements of complete unified-memory CUDA residency.

Artifacts: `/tmp/lunaflux-deepseek-original-hyper-20261011-v1`.

```sh
moon run scripts/replay-deepseek-original-hyper.mbtx prepare NEW_ROOT ORIGINAL_IDS_TEXT
moon run scripts/replay-deepseek-original-hyper.mbtx run ROOT
moon run scripts/replay-deepseek-original-hyper.mbtx collect ROOT
```

Two runner tests and three affected-package tests pass. The original adapter
compiles with `-Wall -Wextra -Werror`; native checking passes with existing
toolchain-migration warning exclusions. No serving dependency, hashing,
authentication, or deployment changes are part of this diagnostic.
