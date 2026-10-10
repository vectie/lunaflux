# Checkpoint-backed F32 hyper connections

The adapter binds `hc_attn_*` / `hc_ffn_*` as F32, and each branch's norm as BF16.
Pure precision contracts distinguish projection-then-normalization and
source/destination-transposed residual reduction from the existing GLM path.
The residual sum is accumulated independently in F32, then the branch product
is added before a single BF16 publication. The same CUDA lowering serves both
contracts; neither model names nor checkpoint paths enter it.

Preparation streams four checkpoint weights and owns six reusable buffers.
Three prefix launches expose normalized branch input; the caller writes the
borrowed branch output before the one-launch suffix. They join the decoder's
queue and add no completion. Close that queue, this owner, then borrowed inputs.
This is an executable component, not complete DeepSeek/DSpark generation.
