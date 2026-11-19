# Hybrid MoE geometry

Shared geometry for Qwen3.5-family repeating gated-delta/full-attention decoders.
The pinned Qwen FP8 profile and BF16 MiniCPM-V 4.7 profile consume this owner.
Vocabulary associations, weight precision, vision and kernel admission remain
with their model/plan packages. Geometry does not imply executable model support.

The memory helpers report logical payload only: BF16 K/V for full-attention
layers and FP32 recurrent matrices for linear layers. Pages, convolution history,
workspace and allocator padding are separate.
