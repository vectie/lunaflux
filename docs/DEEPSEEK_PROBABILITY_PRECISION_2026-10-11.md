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
The fresh committed full-model logit retest must report the remaining difference
without relabeling internal agreement as independent correctness.
