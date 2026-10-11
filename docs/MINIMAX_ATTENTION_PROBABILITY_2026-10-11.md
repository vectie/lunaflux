# MiniMax encoder attention precision — 2026-10-11

The installed original H3 encoder evaluates separate Q/K/V projections,
BF16 Q/K normalization and rotary stages, then library causal SDPA. It explicitly
enables cuDNN SDPA but does not force a backend. The independent ATen bridge now
mirrors that preference and records the actual selected backend. The 140-row,
64-head, dimension-128 run on Spark `.179` selected **Flash (backend 1)**, not
cuDNN; changing this flag alone cannot be claimed as a native numerical fix.
Reference: `/tmp/lunaflux-minimax-aten-reference-original-sdpa-20261011-v2`.

Native text attention previously only represented per-key online F32 softmax,
without an intermediate BF16 probability boundary. The precision IR now
distinguishes `OrderedOnlineF32`, `GlobalF32`, and `OnlineBf16(block_tokens)`.
The text plan explicitly carries the law into its shared causal/GQA CUDA
lowering. This preserves the functional compiler boundary: immutable numerical
semantics → source lowering; neither the generic lowering nor precision IR
contains a model-family or prompt-specific branch.

The existing **ordered-online default remains unchanged**. The offline exporter
can explicitly select `bf16-64` or `bf16-128` after independent square/dot laws.
Their block maxima/rescaling/denominators remain F32, while P rounds to BF16
before F32 fused multiply-add with BF16 V. This is a different floating-point
algorithm, not an alleged bitwise-equivalent optimization or a claim that the
library uses one of these exact scalar schedules.

Actual exported kernels on `.179` pass **26,880** independently evaluated
component outputs across both block sizes, 140 causal rows, grouped heads, zero
scores, ordinary scores and large scores. The comparison allowance is one BF16
ULP, not bitwise matrix-reduction equivalence. Memcheck/leak, racecheck and
synccheck report zero errors under 2-GiB/no-swap limits. Evidence:
`/tmp/lunaflux-minimax-probability-20261011-v1`.

The component result does not establish fifty-layer encoder parity or audiovisual
quality. Diverse original-weight encoder ablations must determine whether this
boundary helps propagated alignment before any default selection changes.
