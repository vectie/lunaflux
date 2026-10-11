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

## Query/KV prefix on the literal original normalized input

The next replay calls the installed original `linear`, its activation quantizer
and FP8 matrix kernels, original learned `RMSNorm`, and the original BF16
per-head normalization expression. Checkpoint weights and UE8M0 scale bytes are
unchanged. All 144 original rows and all 64 heads are retained. Each native
operator receives the literal original intermediate in **matched** mode; a
second **propagated** replay executes the complete native six-stage prefix.

| Boundary | Values | Matched exact BF16 | Matched relative L2 | Propagated relative L2 |
| --- | --- | --- | --- | --- |
| Q-A projection | 147,456 | 147,456 | 0 | 0 |
| Learned Q normalization | 147,456 | 147,456 | 0 | 0 |
| Q-B projection | 4,718,592 | 4,718,452 | 0.000151050% | 0.000151050% |
| BF16 head normalization | 4,718,592 | 4,718,592 | 0 | 0.0000112431% |
| KV projection | 73,728 | 73,728 | 0 | 0 |
| Learned KV normalization | 73,728 | 73,728 | 0 | 0 |

Q-B maximum error is 0.0001220703125; propagated head-normalization maximum
error is 0.000244140625, with 139 differing BF16 values. These tiny discrepancies
do not identify the cause of the approximately 18–21% later retained-state
discrepancy. The next boundary is rotary/cache simulation/window attention,
not an unmeasured change to the already agreeing normalization contracts.

Artifacts: `/tmp/lunaflux-deepseek-original-query-20261011-v2`.
Original exit is zero, `OOMKilled=false`; matched and propagated native runs
exit zero and release CUDA allocations. Full-geometry memcheck with leak
checking, racecheck and synccheck all report zero errors. The runner has one
passing test and the source package has four. The shared test-only tensor I/O
header removes duplicate literal-input/checkpoint loading code from the hyper
and query adapters; it performs no model arithmetic.

The preserved `v1` replay failed because the adapter had not selected the
original generator's BF16 framework default dtype. The original FP8 output
allocation then had an incompatible F32 dtype. `v2` fixes this **test setup**;
it is not a production source bug or a model-numerical repair.

```sh
moon run scripts/replay-deepseek-original-query.mbtx prepare NEW_ROOT ORIGINAL_NORMALIZED_BF16 144
moon run scripts/replay-deepseek-original-query.mbtx run ROOT
moon run scripts/replay-deepseek-original-query.mbtx sanitize ROOT
moon run scripts/replay-deepseek-original-query.mbtx collect ROOT
```

This is an original TP1 full-head **component** comparison. It is not a new
full-model or two-rank DeepSeek correctness pass. No production dependency or
hardening is added.

## Rotary, cache simulation and window attention on original operands

The installed original rotary implementation, UE8M0 cache simulation,
window-index builder and sparse-attention kernel now run on the literal
144-row Q/KV outputs above. All 64 heads are retained; matching each operator
independently distinguishes its own rounding from propagated differences.

| Boundary | Values | Matched exact BF16 | Matched relative L2 | Propagated relative L2 |
| --- | --- | --- | --- | --- |
| Query rotary | 4,718,592 | 4,718,570 | 0.000858670% | 0.000858670% |
| KV rotary | 73,728 | 73,728 | 0 | 0 |
| KV cache quantization simulation | 73,728 | 73,728 | 0 | 0 |
| Window attention | 4,718,592 | 4,717,516 | 0.00627383% | 0.00633559% |
| Inverse rotary | 4,718,592 | 4,718,570 | 0.000359987% | 0.00728139% |

Maximum matched errors are 0.015625 for query rotary and attention, and
0.00390625 for inverse rotary. These first-layer differences still do not
explain the approximately 18–21% later retained-state discrepancy. Output
projections, distinct-stream residual mixing, expert execution and later
compressed layers remain to be traced on original inputs. No tolerance or
production arithmetic has been changed to hide that discrepancy.

Artifacts: `/tmp/lunaflux-deepseek-original-window-20261011-v1`.
Original and native replays exit zero; the original container reports
`OOMKilled=false`. Full-geometry CUDA memcheck/leak, racecheck and synccheck
report zero errors. GPU allocations are released. The source package has
three passing tests and the runner one. This is a TP1 first-layer component
comparison, not a full-model numerical pass.

```sh
moon run scripts/replay-deepseek-original-window.mbtx prepare NEW_ROOT ORIGINAL_QUERY_DIRECTORY 144
moon run scripts/replay-deepseek-original-window.mbtx run ROOT
moon run scripts/replay-deepseek-original-window.mbtx sanitize ROOT
moon run scripts/replay-deepseek-original-window.mbtx collect ROOT
```

## Grouped output projections and quantization propagation

The next replay consumes the literal original inverse-rotated attention output.
Original output-A weights undergo the original checkpoint converter's block
dequantization and BF16 publication; the adapter executes the original grouped
`einsum` and original FP8 output-B `linear`. Native execution uses the exported
production grouped-output lowering with all eight groups and 144 rows.

| Boundary | Values | Matched exact BF16 | Matched relative L2 | Propagated relative L2 |
| --- | --- | --- | --- | --- |
| Grouped output-A | 1,179,648 | 1,178,284 | 0.0102101% | 0.0102101% |
| Quantized output-B | 589,824 | 589,810 | 0.0000826156% | 0.0467903% |

Output-A maximum error is 0.015625. Matched output-B maximum error is
0.0009765625; propagated output-B maximum error is 0.0625. The quantizer
amplifies a small preceding difference, but this two-stage result still does
not account for the later 18–21% state discrepancy. The result excludes
two-rank collective rounding and later-layer propagation.

Artifacts: `/tmp/lunaflux-deepseek-original-output-20261011-v4`.
Original and both native modes exit zero; the original container is not
OOM-killed. Full-geometry CUDA memcheck/leak, racecheck and synccheck report
zero errors, and GPU allocations are released. Three source-package tests
and one runner test pass. Earlier diagnostic setup failures are retained:
`v1` lacked the async test import, `v2` had an incorrectly renamed diagnostic
GPU-query option, and `v3` attempted execution before preparation retired.
None was a production runtime or numerical defect.

```sh
moon run scripts/replay-deepseek-original-output.mbtx prepare NEW_ROOT ORIGINAL_INVERSE_BF16 144
moon run scripts/replay-deepseek-original-output.mbtx run ROOT
moon run scripts/replay-deepseek-original-output.mbtx sanitize ROOT
moon run scripts/replay-deepseek-original-output.mbtx collect ROOT
```

## FFN mixing on distinct original residual streams

The FFN replay first calls the installed original block's attention `hc_post`
with the original output-B branch and original attention mixing coefficients.
It then compares native FFN `hc_pre` and normalization against the original
on those literal, distinct residual streams. This avoids the easier initial
embedding case, where all four streams are identical.

| Boundary | Values | Exact BF16 | Relative L2 |
| --- | --- | --- | --- |
| FFN collapse | 589,824 | 589,716 | 0.003552047% |
| FFN normalization | 589,824 | 589,701 | 0.005719365% |
| FFN post coefficients (F32) | 576 | — | 0.000227973% |
| FFN combination coefficients (F32) | 2,304 | — | 0.000429354% |

Maximum BF16 error is 0.00390625 at both boundaries. Artifacts are retained at
`/tmp/lunaflux-deepseek-original-hyper-ffn-20261011-v1`. Original execution,
native execution and memcheck/leak, racecheck and synccheck all exit zero;
the original container reports `OOMKilled=false` and allocations are released.
This tests native FFN pre-mixing, **not native attention post-mixing** or a
full two-rank block. Production arithmetic and tolerances are unchanged.

```sh
moon run scripts/replay-deepseek-original-hyper.mbtx prepare NEW_ROOT ORIGINAL_IDS_TEXT --ffn-input ORIGINAL_HYPER_DIRECTORY ORIGINAL_OUTPUT_B_BF16
moon run scripts/replay-deepseek-original-hyper.mbtx run ROOT
moon run scripts/replay-deepseek-original-hyper.mbtx collect ROOT
```
