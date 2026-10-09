# H3 component-by-component weight startup

`WeightStartup` bridges admitted safetensors manifests to actual packed device
owners. `prepare_component` inspects checkpoint metadata and streams packed
tensor slices through bounded scratch directly into device allocations. Text,
vision, denoiser, decoder and reference-encoder paths no longer build full host
weight arenas. This avoids retaining a second component-sized weight copy on
unified-memory Spark hardware. Scratch, transfer bytes and incremental hashing
still consume bounded startup memory; modules/workspaces/KV need separate budgets.

Use the pure `conditioner_packings`, `denoiser_packings`,
`video_decoder_packings`, and `audio_decoder_packings` factories. The resulting
allocation indices preserve factory order. Conditioner entries are the real
`EncoderWeights` and `VisionWeights`; other entries are packed device allocations
using the existing model-owned layouts and upload routines.

Device accounting is cumulative across components. Preflight is subtraction
based and happens before device allocation. A partial failure retains all
acquired owners, blocks further preparation, and permits retryable reverse-order
close. `reserved_bytes` conservatively reserves the entire typed encoder group
even if only part uploaded. All components must have the same model identity.

Acquire `WeightLease` after loading, borrow via `with_encoder`, `with_vision` or
`with_allocation`, and retain that lease for the lifetime of all dependent
programs/queues. Release it only after downstream queues drain and close. Closing
with a live lease is rejected. No request-path IO or cryptography is introduced.

This package loads raw weights, not AOT modules or program working memory.
Audio normalized-weight caches remain owned and budgeted by `AudioDecoder`.
Callers must budget modules, program workspace and those caches separately.
For reference audio/video encoders, append `reference_encoder_packing` to that
VAE component's decoder list. It derives the existing segmented layout from
manifest metadata and streams the checkpoint into it, retaining `WeightManifest`
metadata without host arenas. The
`with_reference` callback returns both the segmented owner and immutable
manifest for media encoder `prepare_resolved` after host release. The existing
reference ABI addresses the complete VAE tensor table, so this allocation is
separate from packed decoder weights; both copies count against the aggregate
budget. Its reservation includes a conservative 64-byte alignment bound.

Tests exercise pure budget limits, wrong-component rejection, factory coverage,
ordering, and lease closure. Physical GPU upload and full H3 inference are not
claimed by these CPU tests.
