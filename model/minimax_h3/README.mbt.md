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

The audio shape law explicitly preserves 800 output samples per rounded latent.
For 124 resolved video frames, the nominal duration is 165,333 samples, while
the actual decoder emits 165,600 per channel. These values are distinct;
canonical result metadata and output budgets must use `audio_output_samples()`.
The model adapter selects this law once; neither the generic result protocol
nor device dispatch branches on a MiniMax name.

Checkpoint-backed text conditioning, denoising and both VAE decoders now execute
through explicit bounded device owners. Actual-caption execution at small
geometry is recorded in
[the current report](../../docs/MINIMAX_REAL_CAPTION_EXECUTION_2026-10-10.md).
That result is not independent numerical/quality validation, production serving
or an optimized realistic-resolution benchmark.
