# MiniCPM-V 4.7 decoder operators

This builder turns a validated selected-model specification into reusable
numeric plans. It selects centered gamma for decoder/query/key norms, packed
query/gate projection views, staged BF16 SiLU-gated recurrent norms, full-softmax
top-eight routing, scalar per-head recurrent controls and grouped Q/K state.
Convolution uses the BF16 projection boundary from upstream's unfused fallback.
The routed plan reuses `ExpertMlpPrecision` with the full 256-expert BF16 bank and
BF16 rounding after every contribution. Routing values round to BF16 and widen
exactly into the existing expert ABI's F32 buffer. Existing dense SwiGLU handles
the shared expert, with a separate staged sigmoid gate/combination operator.
Router/shared-gate projections reuse the existing matrix lowering.

Full attention now binds all four BF16 projections, prefix/split-half MRoPE and
the raw projection's strided output gate. The first 64 of 256 head channels
rotate; frequency axes interleave T/H/W with counts 11/11/10. Frequency constants
follow FP32 `1/pow(base,exponent)` rather than substituting a negative exponent.
Coefficients, products and sums preserve their BF16 rounds. Canvas coordinates
and causal KV token positions remain distinct inputs to the enclosing runtime.

`expert_uploads` describes direct checkpoint byte ranges into the per-expert
gate/up/down bank. A layer occupies 1,610,612,736 bytes; all 40 routed banks total
64,424,509,440 bytes. This extent is part of the existing checkpoint payload,
not another full persistent weight copy. Whole-model admission remains separate.

The compiler IR and CUDA source packages contain numeric/layout semantics;
model names, vocabulary and checkpoint choices stay here. The command-line
preparer consumes this builder rather than recreating its assumptions.

`integration/minicpm_v47_moe` assembles these MoE plans into a prepared device
frame with caller-owned workspace. This builder remains an operator subset.
`integration/minicpm_v47_attention` now owns the full-attention ingress and
prepared frame using the existing paged-attention source pipeline. The separate
`model/minicpm_v47_positions` owns canvas positions and continuation. Persistent
state, vision and the enclosing decoder session still need assembly and
validation; source generation does not certify execution.

`linear_projection` binds all five recurrent matrices. The separate
`integration/minicpm_v47_recurrent` composes those plans with existing
convolution/control/delta/gated-normalization kernels in nine ordered launches,
using caller-owned persistent state and scratch.
