# H3 visual conditioner

This package executes the Qwen3-VL-32B visual tower consumed by H3, not the
VideoVAE. It includes patch projection, interpolated learned positions, axial
rotary positions, 27 frame-local noncausal transformer blocks, the final merger,
and deepstack mergers after blocks 8, 16 and 24. Ordinary blocks use tanh GELU;
mergers use exact GELU. The final merger normalizes width 1152 before grouping;
deepstack mergers normalize the grouped width 4608.

`VisionPlan` receives processor grid counts. `preprocess_rgb` accepts decoded
RGB8 frames, performs bounded smart resize, antialiased bicubic resampling,
normalization and BF16 patch grouping. One image repeats its frame into both
temporal patch slots; odd-length videos repeat the final frame. Codec decoding,
frame selection, timestamps and text chat-template construction belong above
this numeric component. Processor resampling/transcendental rounding still
requires numerical comparison with the checkpoint's processor implementation.

`VisionWeights.prepare` resolves exact named tensor shapes from the admitted
TextConditioner host component and uploads bounded chunks. `VisionRequest`
owns scratch, patches, function handles, four output regions and the ordered
executor. Partial preparation/submission errors retain owners for `close`;
completed outputs are accessible only after `poll` returns true. Close text
consumer queues before the vision request, then close vision weights.

The output allocation is four contiguous BF16 `[merged_rows,5120]` regions:
main merger, deepstack 8, deepstack 16, deepstack 24. Main features replace
visual-token embeddings; deepstack features are added after text layers 0, 1,
2. `VisionPlan.merged_position` supplies grid/temporal/spatial coordinates for
the text adapter's token-stream rotary-position construction.

The emitted scalar CUDA functions and 284-launch recipe are executable AOT
source/bindings, not measured optimized kernels. Host/native tests do not prove
GPU numerical correctness, sanitizer cleanliness, or efficient H3 serving.

Semantics were checked against SGLang's local `minimax_h3_qwen3vl.py`,
`qwen3vl.py`, the published Qwen3-VL-32B config/processor config, and Hugging Face
Transformers' Qwen3-VL vision model and Qwen2-VL fast image processor.
