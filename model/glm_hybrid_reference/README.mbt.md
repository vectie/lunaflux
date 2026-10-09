# GLM hybrid correctness reference

This package is a bounded, allocation-owning correctness oracle for tiny GLM
hybrid fixtures. It covers DSA score/index state and Flash k-pooling, KDA
causal short convolution and recurrent delta updates, vision patch/downsample/
project shapes, per-head sigmoid-gated RMSNorm, terminal KDA row-major dense
projection, raw no-bias MoE router logits, safe-bound F32 decay control, and
exact plan schedule traversal.
It also exposes separate tiny F32 oracles for sigmoid/correction and grouped
expert choice, preserving corrected choice scores and uncorrected selected
weights as distinct streams.
Production APIs depend only
on family-neutral hybrid plan vocabulary; official GLM-5.3 builders occur only
in tests.

The end-to-end tiny block oracle covers RMSNorm, low-rank Q/K/V projection,
causal DSA over explicit selected indices, output projection, dense or routed
MoE SwiGLU, F32 router correction, residual or multi-stream mHC composition,
and a bidirectional vision attention/MLP/merge/project block. `-1` is the sole
allowed padding sentinel for an unfilled sparse-selection slot.

It is intentionally unsuitable for production inference: it uses scalar
`Double` loops, owns intermediate arrays, performs validation on each call,
and has no cache, scheduler, device, kernel, or materialization authority.
