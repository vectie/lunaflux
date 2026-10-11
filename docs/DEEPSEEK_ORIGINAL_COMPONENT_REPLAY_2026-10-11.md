# DeepSeek original first-layer component replay — 2026-10-11

## Original routing on all 144 literal FFN rows

The routed-expert replay supplied original routing weights, so it did not test
the native router. The separate routing replay now executes installed original
`Gate` and the production numeric routing lowering on all 144 original
first-layer FFN-normalized rows and their literal token IDs. Layer-0 hash
selection is the actual first-layer input. Layer-3 learned selection uses that
same corpus with its own checkpoint weights/bias: this is a component test,
**not the actual propagated layer-3 input**.

Both paths select identical expert IDs in every slot, **864/864 for each
router**. Identity-aware comparison also reports zero changed expert sets in
144 rows. This comparison does not assume that original unsorted top-k and
native descending publication must always use the same slot order.

| Original router | Logit relative L2 | Score relative L2 | Selected-weight relative L2 | Weight maximum error |
| --- | --- | --- | --- | --- |
| Layer-0 token hash | 0.0000599616% | 0.0000231966% | 0.0000438517% | 0.000000566244 |
| Layer-3 learned, first-layer corpus | 0.0000570585% | 0.0000227907% | 0.0000192749% | 0.000000238419 |

These results do not explain or resolve the 18–21% retained-state difference.
They rule out a large router error on this matched corpus; later actual router
inputs, shared-expert addition and two-rank/full-model propagation remain open.
No production arithmetic or acceptance tolerance changed.

Artifacts: `/tmp/lunaflux-deepseek-original-router-20261011-v2`. Original
containers exit zero with `OOMKilled=false`. Both native executions and all
six memcheck/leak, racecheck and synccheck runs complete without errors, with
GPU owners released. Two runner regressions and four source-package tests
pass. The preserved v1 failed during diagnostic source concatenation because
two kernels lacked a separating newline; it did not execute on the GPU.

```sh
moon run scripts/replay-deepseek-original-router.mbtx prepare NEW_ROOT ORIGINAL_FFN_BF16 ORIGINAL_IDS_I32
moon run scripts/replay-deepseek-original-router.mbtx run ROOT
moon run scripts/replay-deepseek-original-router.mbtx sanitize ROOT
moon run scripts/replay-deepseek-original-router.mbtx collect ROOT
```

## Learned compression corpus qualification

The `pool-cache-20261011-v1` incremental replay now continues through the
original compressor's YaRN rotary and UE8M0 cache simulation. Its matched
transform consumes the **literal original normalized tensors**, not native
normalizations. Both ratios are bitwise exact: 18,432/18,432 ratio-4 values
and 512/512 ratio-128 values. Propagating native projections, pooling and
normalization leaves one ratio-4 transformed value different (relative L2
0.000418077%, maximum 0.00048828125); ratio-128 remains exact. All six
incremental sanitizer runs report zero errors with deterministic release.
The separate full-prompt `pool-cache-full-20261011-v1` produces the same
native propagated transformed bytes for both ratios. This closes the cache
transform portion of this corpus replay, **not full compressed-layer or
two-rank model correctness**. It does not explain the retained-state gap.

The follow-up `pool-incremental-20261011-v1` runs the same 144 literal rows as
144 one-row submissions, retaining original and native compression state.
Matched pools are exact for both ratios; ratio-4 matched norm still differs at
one value. Propagated ratio-4 pool/norm relative L2 is 0.000112815% /
0.001677099%; ratio-128 pool/norm is exact. The **native** propagated final
normalizations are bitwise identical between full-prompt and incremental
execution for both ratios. Original projection shape changes are not replaced
with native values. All six incremental sanitizer runs report zero errors.
The runner now has two passing regressions. Use `prepare ... --incremental`
to select this separate diagnostic without changing corpus bytes.

The installed original `Compressor` and the production learned-pool CUDA
lowering replay the same literal 144-row original FFN-normalized corpus with
original layer-2 ratio-4 and layer-3 ratio-128 parameters. Forward hooks observe
the original pooled and normalized values without replacing arithmetic.
These are **not the actual inputs at layers 2 and 3** of a full execution:
this qualifies component geometry and arithmetic, not later-layer alignment.

| Boundary | Ratio 4 exact / values | Ratio 4 relative L2 | Ratio 128 exact / values |
| --- | --- | --- | --- |
| Pool, original projected operands | 18,432 / 18,432 | 0 | 512 / 512 |
| Norm, original pooled operands | 18,431 / 18,432 | 0.001673019% | 512 / 512 |
| Pool, native projections propagated | 18,430 / 18,432 | 0.000438643% | 512 / 512 |
| Norm, native projections propagated | 18,429 / 18,432 | 0.001725301% | 512 / 512 |

Both projection comparisons have F32 relative L2 below 0.000065%; the largest
absolute difference is 0.00001192093. The original full-prompt replay stopped
before rotary/cache quantization; the follow-ups above cover those operations
and incremental submissions. Full two-rank propagation remains excluded.
This does not explain the retained 18–21% state gap.

Artifacts: `/tmp/lunaflux-deepseek-original-pool-20261011-v1`. Original and
native runs exit zero, original containers report `OOMKilled=false`, and both
ratios pass memcheck/leak, racecheck and synccheck with deterministic release.
One runner regression and three source-package tests pass. No production
arithmetic, acceptance tolerance or runtime hardening changed.

```sh
moon run scripts/replay-deepseek-original-pool.mbtx prepare NEW_ROOT ORIGINAL_144_ROW_BF16
moon run scripts/replay-deepseek-original-pool.mbtx run ROOT
moon run scripts/replay-deepseek-original-pool.mbtx sanitize ROOT
moon run scripts/replay-deepseek-original-pool.mbtx collect ROOT
```

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

## Actual six routed FP4 experts on a literal original FFN row

The installed original hash gate routes token ID 0 using the literal original
post-attention FFN-normalized row. Its six selected checkpoint experts and F32
routing weights are retained in original slot order. Native execution uses the
production precision plan (FP4/E8M0 weights, dynamic K128 FP8 activations,
clamp 10, F32 SwiGLU product and routing before the down-input BF16 round).
Compact expert IDs are only an offline slot remapping of those actual weights.

All **12,288 gate/up activation values are bitwise identical**. Four down
outputs are completely identical; two have one differing value each. The worst
matched down relative L2 is **0.0150414%** (slot 4); slot 2 is 0.00000306464%.
Propagated results agree with matched results because all preceding activation
values match. This does not test shared-expert addition, all 144 prompt rows,
two-rank summation or later-layer experts.

Artifacts: `/tmp/lunaflux-deepseek-original-expert-20261011-v3`. Original and
both native executions exit zero, the original is not OOM-killed, all three
CUDA sanitizers report zero errors and GPU allocations are released. Earlier
setup failures remain preserved: v1 omitted the quantizer error argument and
complete block grid; v2 attempted typed shell-dtype copying unsupported by the
installed framework. v3 copies literal checkpoint bytes through uint8 views.
These were test-adapter errors, not production numerical fixes.

```sh
moon run scripts/replay-deepseek-original-expert.mbtx prepare NEW_ROOT EXACT_ORIGINAL_FFN_BF16 ORIGINAL_TOKEN_ID
moon run scripts/replay-deepseek-original-expert.mbtx run ROOT
moon run scripts/replay-deepseek-original-expert.mbtx sanitize ROOT
moon run scripts/replay-deepseek-original-expert.mbtx collect ROOT
```

## Attention post-publication on matched original operands

The FFN runner additionally replays the native attention `hc_post` using the
original BF16 branch, original BF16 residual streams and original F32 post/
combination coefficients. Its output is compared with the installed original
`Block.hc_post` result, before either FFN pre-mixer consumes it.

All **2,359,296 BF16 values match bit-for-bit**, including all 144 rows and four
streams. The v2 FFN pre-mixing results remain unchanged from v1. Artifacts are
at `/tmp/lunaflux-deepseek-original-hyper-ffn-20261011-v2`; native, original,
memcheck/leak, racecheck and synccheck all exit zero. This rules out an error
in this matched publication boundary, not the accumulated whole-model gap.

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
