# MiniMax H3 weight boundary

The official release contains separate denoiser, text-conditioner, video-VAE,
and audio-VAE weights. A partial-shard binding classifies each present tensor
and retains authenticated content plus canonical joint-diffusion plan identity.
It is deliberately not a completeness claim.

`join_complete_component` is the stronger boundary. Schema v1 owns exact
638-tensor denoiser, 1,058-tensor Qwen3-VL conditioner, 703-tensor VideoVae,
and 1,087-tensor AudioVae vocabularies, rejects duplicates and identity mixing,
and only then constructs a complete binding. FL2VA and Ref2VA denoisers remain
separate authenticated partitions.

The denoiser vocabulary also owns the released mixed-precision schema. Video
and audio input/output projections plus the timestep MLP are F32; the context
path, refiner, transformer blocks, and output normalization are BF16. A tensor
with the right name and shape but the wrong dtype fails semantic admission.

Refiner source names are `token_refiner.refiner_blocks.{0,1}.*`, matching the
actual converted checkpoint index. Semantic binding and packing use these same
names. The nonexistent `token_refiner.blocks.*` spelling is not an alias.

The VideoVae schema binds its exact three-shard index and the AudioVae schema
binds its exact single-file header; both accept only their official F32 names,
shapes, dtypes, and payload sizes. No binding reads raw payload bytes, executes
remote code, or allocates a device tensor. Encode/decode execution remains a
separate typed fail-closed boundary.

## Original VideoVAE source layout

The original `AutoencoderKLLegacy` single-file checkpoint has 560 F32 tensors;
the converted layout has 703. Both complete vocabularies are explicit and
cannot be mixed. `MiniMaxH3OriginalVideoVaeSchema` describes original names
and exact shapes without changing the execution graph or kernel ABI.

`original_video_vae_copies` translates canonical decoder packing once at
startup. Original QKV rows are **head-interleaved**, as upstream reshapes its
projection to `[batch,sequence,heads,3*head_dimension]` before chunking Q/K/V.
The adapter emits bounded per-head slices, not three whole-matrix slices.
Mask-token bytes remain authenticated but are unused by inference. Encoder
aliases use the same source mapping for reference-media execution.
