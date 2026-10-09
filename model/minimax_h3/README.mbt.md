# MiniMax H3 model-family boundary

This package records the exact startup requirements for MiniMax H3 and builds
an immutable joint audio/video diffusion plan without reinterpreting the model
as LunaFlux's decoder-only token-to-logits graph.

The official [MiniMax H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)
defines FL2VA and Ref2VA joint audio/video diffusion checkpoints. Their
transformer stack is BF16 while input/output projections and timestep modules
remain F32. The
official [architecture description](https://www.minimax.io/blog/minimax-h3)
describes the H3 VAE and Omni Transformer system. The released configuration
is normalized and validated by `model/minimax_h3_spec`.

`build` now produces exact FL2VA or Ref2VA workflow stages, the released dual
rectified-flow schedules (video shift 12, audio shift 3), component roles,
conditioning limits, deterministic identity, and bounded request/latent shape
resolution through `model/joint_diffusion_plan`. It also binds the official
frame-major patch-vector and channel-major audio-row layouts.

This is an execution-plan foundation, not physical runtime readiness. Metadata
binding now has exact complete-manifest joins for the denoiser and Qwen3-VL
conditioner; VAE complete vocabularies and every component's payload/device
materialization still fail closed. Conditioning execution, transformer/VAE
kernels, device-worker ownership, and bounded audio/video request/result
protocols do not yet exist.
