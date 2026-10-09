# MiniMax H3 host materialization

This startup-only package binds a digest-pinned, exact component manifest to
approved safetensors shard files and copies mixed F32/BF16 payloads directly into final
bounded host arenas. It admits only the complete 638-tensor FL2VA/Ref2VA
denoiser manifests, the complete 1,058-tensor Qwen3-VL conditioner manifest,
the 703-tensor VideoVae manifest, and the 1,087-tensor AudioVae manifest.

The manifest proves model identity, exact names, official shapes and dtypes,
shard/index membership, payload ranges, artifact digests, and deterministic
arena placement. The result remains component-scoped and explicitly releasable.

VAE materialization remains component-scoped; this package never marks the
complete MiniMax H3 model executable. It provides no VAE encode/decode, device
upload, kernel execution, scheduler, codec, media preprocessing, or request
protocol.

`model/materialize` is intentionally not reused: its public contract is bound
to Llama weights and a single in-memory byte view. File inspection, bounded
header parsing, incremental authentication, replay checks, and segmented direct
copy belong to the family-neutral `model/streaming_safetensors` package. It
keeps 64-bit source offsets, never forms whole-file `Bytes`, and authenticates
each shard once while filling all final MiniMax-owned arenas.
