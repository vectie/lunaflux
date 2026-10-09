# H3 Qwen3-VL layer-50 feature extractor

This module emits and binds 651 ordered CUDA launches: one token embedding and
13 operations for each of language layers 0–49. The result is BF16 `[rows,5120]`
at workspace offset zero: **unnormalized hidden_states[50]**. It deliberately
omits layer 50–63, final language RMSNorm, LM head, decode cache and sampling.
It is separate from the DiT's two-layer token refiner.

`model/minimax_h3_text_encoder` owns the pure plan, exact layer weight packing,
and workspace regions. `kernels/text_encoder_cuda_source` owns causal grouped
query attention and Qwen staged-BF16 normalization/rotary. The integration
reuses existing dense projections, embedding, residual and staged SwiGLU.

`parse_config` reads the encoder's Qwen3-VL architecture and default full-head
rotary contract, including its theta and epsilon. It rejects scaled variants.
`EncoderWeights::prepare` owns/upload all consumed layer and embedding storage;
`EncoderRequest::prepare` owns counts, IDs, BF16 cosine/sine tables, workspace,
functions and the execution queue. Submit once, poll or wait, then use
`prepare_refiner` to connect the completed `[rows,5120]` allocation to the actual
H3 context projection and two-layer refiner. Cancel drains, and failed cleanup
retains owners for retry. Close downstream queues first. No runtime JIT.
`execute()` composes submit plus one blocking completion wait and implements
the shared `PreparedFrame` interface (`drain` cancels, `release` closes).
`prebind_refiner()` and `with_destination()` permit constructing the entire
dependent pipeline before submission. They expose uninitialized destinations,
not completed results. The refiner also implements `PreparedFrame`; execute
encoder, then refiner, then dependent denoising, and release in reverse order.

For media, `placement_from_grids` computes canonical timestamp-aware THW
coordinates and exact visual-token mapping. Video temporal patches consume
separately bracketed frame blocks. Supply the vision encoder's four contiguous
BF16 feature regions. The main region replaces visual token embeddings before
layer 0; deepstack regions are added **after language layers 0,1,2**, not after
vision extraction layer numbers 8,16,24. This adds four launches (655 total).
The owner checks every visual placeholder is covered once. Pixel processing and
vision execution are separate modules; tables use Qwen interleaved mRoPE.

Scope: single unpadded sequence; no multi-request batching or tensor parallelism.
The config/position plan permits the official 262,144 positions, but the current
source bundle supports **at most 83,886 input rows**, because the shared staged
MLP renderer requires `rows * 25,600 <= Int::max_value`. Larger requests fail
source construction; this module does not silently truncate or bypass that guard.
These correctness-grade scalar kernels are **not a fast tiled implementation**.
CUDA compilation, numerical parity, sanitizers and performance are unverified.
Exact host region resolution and bounded layer/embedding upload are implemented
in the existing host materialization and block upload packages. The current
host component still materializes the full checkpoint before selecting these
consumed tensors; skipping unused upper layers during file streaming remains.
Physical qualification and fast tiled kernels remain open. Host trigonometric
rounding has not been compared with CUDA. Source numerical contracts follow
[Qwen3-VL](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py);
dimensions/config were checked against the official Qwen3-VL-32B checkpoint.
