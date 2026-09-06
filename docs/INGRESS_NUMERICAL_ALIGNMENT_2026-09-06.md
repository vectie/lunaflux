# Fusion-cut numerical alignment

Full projection fusion and partial postprocessing must use the same numerical
schedule on either side of the materialization boundary. The compiler now owns
an immutable `AttentionIngressNumerics`: head width, strided pairwise reduction
width, Float epsilon, and AOT-materialized Float rotary basis. Head count and
fusion placement cannot change it. Normalization rounds to BF16 before paired
rotary evaluation, which rounds again before the KV commit.

A shared CUDA lowering consumes this plan in both serving artifact generators.
The partial kernel retains its 128-thread ABI but a complete subgroup performs
the pure epilogue, using exactly the full path's reduction tree. Other threads
participate in the surrounding block loads/stores and barriers. This removes
per-token `powf` and the previously different 128-lane reduction tree. Planning
does not contain CUDA names or model-family selection.

This aligns the epilogue, not the preceding projection's GEMV/GEMM accumulation.
Physical differential testing and serving measurements must distinguish these
two boundaries. Partial fusion remains an explicit offline evaluation choice
until its complete execution path has been checked; no throughput or
batch-invariance claim follows merely from source alignment.
