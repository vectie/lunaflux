# DeepSeek online probability boundary — 2026-10-11

The original MP2 full-vocabulary comparison confirms a native/reference
differential. Source inspection found a missing numerical boundary, not a new
authentication or deployment requirement.

The original sparse-attention kernel runs online KV64 blocks. F32 scores and
denominators remain unrounded, but unnormalized P is cast to BF16 before P×V.
Previous native window/compressed attention instead used a global F32 softmax
and unrounded P×V, with sink participating in the maximum. These are different
floating-point algorithms even when they represent the same real-valued formula.

`compiler/precision_ir/AttentionProbabilityLaw` now declares either `GlobalF32`
or `OnlineBf16(block_tokens)`. Existing unrelated callers keep GlobalF32. The
DeepSeek builder selects the original KV64 law; the shared CUDA lowering
contains no model-family branch. Maxima, rescaling, denominator and value sums
remain F32; each block's P rounds once to BF16. Sink enters after all blocks.
Empty masked blocks preserve the recurrence without NaN. Logical window slots
stay padded before the compressed read set, preserving block boundaries rather
than moving compressed values into future-token holes.

Actual generated window, learned-compressed and all-causal kernels pass
**10,416** component outputs on Spark `.179`: early causal rows, 64-block crossing,
window rollover, retained/current KV, masked compressed slots, large sink,
inactive rows and invalid-frame clearing. The independent host recurrence uses
the same explicit numerical law, with a one-BF16-ULP comparison allowance; it
does not claim bitwise Tensor Core reduction equivalence. CUDA memcheck/leak,
racecheck and synccheck report zero errors under 2-GiB/no-swap limits.
Artifacts: `/tmp/lunaflux-online-probability-20261011-v1`.

This component test fixes a demonstrable precision-contract omission, but
cannot attribute the previous **11.4684%** full-logit relative L2 to it alone.
Native six-row streaming and original whole-prompt prefill also have different
floating-point partitioning, and scalar versus matrix reductions can differ.
The fresh full-model retest used commit `4ecd2040`, newly exported AOT kernels,
and the identical 144 literal token IDs as the original MP2 reference. Both
native ranks retired successfully, with GPU contexts released. Evidence:
`/tmp/lunaflux-dspark-run-probability-20261011-v2`; the reference remains
`/tmp/lunaflux-deepseek-upstream-reference-20261011-v9`.

| Matched final-prefill row, 129,280 logits | Previous native | Corrected native | Original reference |
| --- | --- | --- | --- |
| Greedy token | 19923 (Hello) | 33310 (hello) | 33310 (hello) |
| Token 33310 score | 31.68308 | 32.25790 | 32.38558 |
| Logit RMSE | 0.604927 | 0.595892 | — |
| Relative L2 | 11.4684% | 11.2971% | — |
| Maximum absolute error | 3.86580 | 4.36326 | — |

The single observed generated-token mismatch is fixed, but the complete logit
distribution is still materially different; maximum error actually increased.
This is **one matched greedy token**, not EOS agreement, multi-prompt acceptance,
or full-model numerical equivalence. The original CLI always appends a reporting
EOS, so that marker alone does not prove sampling. The preserved original
command requests eight new tokens with a 144-token prompt and 256-token
capacity. Stopping after one content token therefore implies a sampled EOS on
the following step under the original generation loop, rather than exhaustion
of either limit. The native run requested only one generated token and cannot
establish EOS agreement. Streaming partition and
matrix/scalar reduction differences remain to be isolated.

## Completed valid one-token prefill ablation

Both original MP2 ranks completed
`/tmp/lunaflux-deepseek-upstream-reference-token-prefill-20261011-v1` with exit
zero, `OOMKilled=false`, all 36,943 tensors loaded, and identical rank logit
outputs. The 144 literal prompt tokens match the native prompt. Only the
initial prompt is partitioned into 144 calls of the original one-token forward;
the original numerical kernels and subsequent generation remain unchanged.
Both the whole-prompt and one-token references generate content token `33310`.

| Final-prefill vocabulary comparison | RMSE | Relative L2 | Maximum absolute difference |
| --- | --- | --- | --- |
| Corrected native vs original whole-prompt | 0.595892 | 11.2971% | 4.36326 |
| Corrected native vs original one-token calls | 0.647957 | 11.7744% | 3.86141 |
| Original one-token calls vs original whole-prompt | 0.763808 | 14.4805% | 4.01242 |

This establishes that the original model's own legal execution partition changes
its floating-point output substantially. It does **not** establish that the
native discrepancy is solely partitioning, prove native numerical equivalence,
or justify accepting a mismatching token. Native still uses six-row prompt
frames, not either matched reference geometry. Partition-sensitive projection,
compression and attention arithmetic must be attributed before declaring a
full correctness pass. In particular, six-row *original* continuation is not a
valid comparison: its decode read-set helpers assume one query row. The valid
one-row experiment must not be presented as an exact native-shape reference.

Reports `NATIVE_VS_TOKEN_PREFILL.txt` and `TOKEN_VS_WHOLE_PREFILL.txt` preserve all
129,280-value comparisons; no production execution or acceptance threshold was
changed for this experiment.

The native frontend replay now supports an explicit `--prefill-rows 1..6`
diagnostic partition. This changes only logical prompt frames: the unchanged
six-row AOT allocation/execution capacity still accommodates DSpark's five-row
verification block. It must not be described as a newly compiled one-row kernel.
The one-row native campaign completed at
`/tmp/lunaflux-dspark-run-token-prefill-20261011-v1` with the same corrected
`4ecd2040` binary/AOT and all 144 literal IDs. Both ranks exit zero and release
their GPU owners. Logical row count is one, while physical AOT capacity remains
six. The captured frontier is therefore physical row **zero**, not row five.

| Native one-row vs original one-row, 129,280 logits | Result |
| --- | --- |
| Native / original greedy token | 33310 / 33310 |
| Native / original winning score | 34.619720 / 34.679176 |
| RMSE | 0.686261 |
| Relative L2 | **12.4704%** |
| Maximum absolute difference | 4.995758 |

The matched logical partition does **not** close the full-logit discrepancy.
The earlier native-six-row versus original-one-row comparison is a different
experiment and remains preserved. Matching the first token is not distribution
equivalence. This diagnostic requests one generated token to capture logits;
the prepared eight-token/EOS envelope still needs its separate native replay.
Full outputs and the exact row selection are recorded in
`MATCHED_ONE_ROW_LOGITS.txt`; no correctness threshold is relaxed.

The complete local native suite after this change passed **4,997/4,997**, using
the existing toolchain-migration warning exclusions `-20-25-29-35-79-92`.

The later [retained-target comparison](DEEPSEEK_RETAINED_TARGET_COMPARISON_2026-10-11.md)
directly observes the original sampled EOS and localizes a material difference
before the vocabulary head. Native EOS agreement and full-distribution parity
must still be distinguished from the agreeing first content token. The latest
full local native suite passes **5,004/5,004**, with the same migration warning
exclusions; this software result is not physical numerical equivalence.
