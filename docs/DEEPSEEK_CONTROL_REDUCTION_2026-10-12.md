# Matched DeepSeek control reduction — 2026-10-12

This follows the output-A correction and residual-publication attribution in
[the layer-stream report](DEEPSEEK_ORIGINAL_STREAM_REPLAY_2026-10-11.md).
It is a numerical investigation, not a throughput or full-model quality result.

The original bridge observes `mixes` and `rsqrt` at the return of unmodified
`Block.hc_pre`. All 144 literal predecessor rows use the original one-row GEMM
geometry. The observer changes no arithmetic; post/combination outputs remain
bitwise identical to the previous no-internals bridge. Original root:
`/tmp/lunaflux-deepseek-original-hyper-layer0-rows1-20261012-v2` on `.179`.

## Isolated causes

The diagnostic actual-renderer production control reproduces both native F32
control arrays bitwise. Its inverse RMS differs on **all 144 rows**, relative
L2 **0.000320213%**. Projection results after normalization differ on 3,455/3,456
values, relative L2 **0.000365954%**. Post-control error is **0.000590706%**.

| Intervention, identical original input | Post-control relative L2 |
| --- | ---: |
| Actual ordered production source | 0.000590706% |
| Original inverse only; native ordered projection | 0.000194052% |
| 32 striped projection partials; native inverse | 0.000388130% |
| Original normalized mixes; native nonlinear controls | 0.000021391% |
| Compensated square sum; ordered projection | 0.000194147% |
| Compensated square sum + 32-partition projection, actual new lowering | **0.000021539%** |

Original-value substitutions are **offline causal controls**, never serving
data. Fixing projection alone leaves normalization error. A more accurate
square sum alone leaves projection error. Even exact original mixes leave a
smaller nonlinear-control/Sinkhorn difference, which is not waived.

Compensated F32 square summation makes **141/144 inverse values bitwise exact**,
relative L2 **0.000001076%**; three differ by at most 9.54e-7. A 256-partition
square tree matches 140/144 and is not more accurate in this capture. F32 FMA
changes projection details, but has no meaningful additional post-control norm
benefit over separated products here. No intrinsic-exp production selection.

## Actual compiler/decoder implementation

The existing `SquareReductionLaw` and `DotReductionLaw` now flow through pure
`HyperConnectionPrecision` into private CUDA lowering. No extra numerical IR,
model-specific kernel branch or vendor library is introduced. The refined
candidate uses compensated square sums and `StripedTreeFloat32(partitions=32)`
with separate F32 product rounding. Both attention and FFN envelopes receive
the policy; tests cover four window/compressed layers and all eight consumers.
The numerical source contract explicitly carries the selected laws instead of
mislabeling refined code as ordered. Workspace and weight budgets stay equal.

Ordinary callers retain ordered control laws. Their generated source is
byte-identical to the retained source, independently checked with `cmp`.
The offline `prepare-refined`/`replay-layer-refined` route selects the candidate
at both export and execution. No literal reference or Python/Torch enters it.

Actual source probe root:
`/tmp/lunaflux-deepseek-hyper-arithmetic-20261012-v4`.
It observes **141/144** exact inverse values, **1,025/3,456** exact normalized
mixes, **15/576** exact post controls and **572/2,304** exact combinations.
Post norm error decreases about **27.4×**, mix norm about **35.8×**; combination
error is **0.000004574%**. These F32 arrays are not yet bitwise parity.

All twelve diagnostic routes and the actual refined source finish memcheck
with zero errors/leaks, racecheck with zero hazards/errors/warnings and
synccheck with zero errors. These all-144-row control probes use 2-GiB/no-swap
units and release CUDA. The earlier `v1` compile failure from a diagnostic
helper-name replacement is retained; `v2` corrects it and adds a regression.
`v3` isolates square reductions; `v4` uses the actual new renderer.

## Commit-pinned complete layer

The `0be5ea5b` ARM release build passes on the clean source, then the same 144
original one-row layer-0 replay completes, zero exit, empty application stderr,
no swap and CUDA closed. Logs explicitly record compensated control squares
and striped control dots in addition to tiled attention/complex rotary/striped
output-A. It is not the default policy. Root:
`/tmp/lunaflux-deepseek-production-layer-0-refined-rows1-20261012-v1`;
reports: `/tmp/lunaflux-deepseek-layer-0-refined-frontiers-20261012-v1`.

| Propagated boundary | Before control refinement | After control refinement |
| --- | ---: | ---: |
| Attention input | Bitwise | Bitwise |
| Sparse attention, inverse, output-A | Bitwise | Bitwise |
| Output-B differing BF16 / 589,824 | 11 | 11 |
| Residual differing BF16 / 2,359,296 | 2,386 | **64** |
| Residual relative L2 | 0.009148849% | **0.000704269%** |
| FFN-input differing BF16 / 589,824 | 2,131 | **56** |
| FFN-input relative L2 | 0.021675558% | **0.001278160%** |
| FFN-output relative L2 | 0.198810072% | **0.031858964%** |
| Complete-layer relative L2 | 0.153870184% | **0.025374703%** |

Complete-layer error falls **83.5%** from the previous candidate, about **93.2%**
from the first 0.3749400% matched baseline. This proves propagated numerical
benefit, not a speedup. It retains 12,377 differing complete-output BF16 values;
remaining nonlinear controls, FFN local arithmetic and propagated drift are
not waived. Affected tests pass 114/114 plus the new metadata/unsupported-law
package tests 12/12; the offline mode tests pass 2/2. Native check succeeds with
unchanged toolchain exclusions. The whole-module clean TLS blocker remains.

The residual publisher itself is unchanged. Speed, compressed-path parity,
full-model 18–21% drift and MiniMax media/encoder differences remain open.

The aggregate improvement is not a per-row parity claim. Position 143 improves
from **0.324717%** complete-output error to about **9.54e-8%**. But row 47 moves
from **0.246268% to 0.253464%**, a small regression despite FFN-input error
decreasing from **0.010825% to 0.001572%**. Its FFN-output error increases from
**0.322967% to 0.336369%**. A matched latest-input FFN intervention is required
to distinguish local arithmetic from quantization/routing sensitivity. These
rows remain visible; the new candidate is not a blanket serving selection.

The same committed executable also completes the original-input **compressed
layer 3**, one-row geometry, in
`/tmp/lunaflux-deepseek-production-layer-3-refined-rows1-20261012-v1`.
Complete-output relative L2 is **0.155833791%**, versus earlier ordered
**0.231644100%** and tiled-only **0.232746659%**. This is about **32.7% less
error** than ordered, not parity. Only **1,210,863/2,359,296 BF16 values** match;
all are finite, maximum absolute error 0.018493652. The candidate combines
control, rotary and output changes, so this is not an isolated attribution to
one of those changes. It exits zero and closes CUDA; default selection stays
unchanged and a dedicated compressed-attention intervention remains necessary.
