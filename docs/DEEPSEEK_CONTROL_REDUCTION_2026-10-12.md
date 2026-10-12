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

Complete-layer propagation and speed remain to be measured before any default
policy change. The residual publisher itself is unchanged. Full-model 18–21%
drift, compressed-path parity and MiniMax media/encoder differences remain open.
