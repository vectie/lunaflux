# DeepSeek retained target comparison — 2026-10-11

This comparison uses the installed original DeepSeek-V4-Flash-0731 implementation,
not another LunaFlux execution as the numerical reference. Both implementations
consume the same 144 literal prompt IDs in one-logical-row prefill calls. Native
AOT capacity remains six physical rows; the compared prompt frontier is row zero.

Native source: `b0dceaa0cc742ba524a147bb55bf0ddf79d708f3`.
Native binary/AOT replay: `/tmp/lunaflux-dspark-run-target-capture-20261011-v2`.
Original MP2 replay: `/tmp/lunaflux-deepseek-upstream-reference-sampler-20261011-v2`.
Original GB10 sparse-attention lowering uses the previously documented 16-head
tiles; this does not change the original numerical contract. Both original ranks'
129,280 logits and all three retained means are bitwise identical to each other.

## Actual samples, not a reporting marker

The observer records the original generation loop's actual sampler result and
its forward frontier. Intermediate prompt samples and predictor-head samples do
not count as generated output. At positions 143 and 144 the original model samples
`33310` (`hello`) and `1` (EOS), respectively. The final CLI's appended EOS is not
used as evidence of EOS sampling.

The native one-token logit capture emits `33310`. Its independent eight-token
ordinary-target replay is separate and still required to check EOS agreement;
the logit capture is not an eight-token generation result.

## Difference is already present before the vocabulary head

The decoder already retains target-layer means 40/41/42 for DSpark. The diagnostic
reads those allocations only after normal execution retires; no new scan or
readback is added to production token steps. The original observer captures the
same `h.mean(dim=2)` outputs before the head.

| Matched retained mean | Values | Exact BF16 values | Relative L2 | Cosine similarity | Maximum absolute error |
| --- | --- | --- | --- | --- | --- |
| Layer 40 | 4,096 | 26 | 21.0982% | 0.977631 | 2.453125 |
| Layer 41 | 4,096 | 33 | 21.1090% | 0.977745 | 2.437500 |
| Layer 42 | 4,096 | 37 | 17.9389% | 0.983963 | 3.250000 |

The final head reproduces the earlier matched-one-row difference: 12.4704%
relative L2, RMSE 0.686261, maximum absolute error 4.995758. Both head argmaxes
are 33310, scoring 34.619720 versus 34.679176.

These observations rule out a **head-only** explanation. They do not identify
the earliest divergent block or distinguish projection, compression, routing,
attention and residual accumulation contributions. They do not justify a
full-distribution equivalence claim or changing an acceptance threshold.

Reports: `/tmp/lunaflux-deepseek-target-layers-20261011-v3` and
`/tmp/lunaflux-deepseek-target-head-20261011-v3`.

Reproduction:

```sh
moon run scripts/compare-deepseek-target-captures.mbtx NATIVE_ROOT ORIGINAL_ROOT NEW_REPORT_ROOT 6 0
moon run scripts/compare-checkpoint-logits.mbtx NATIVE_ROOT/logits.f32 ORIGINAL_ROOT/prefill-logits-rank0.f32 NEW_HEAD_REPORT 129280 0 0
```
