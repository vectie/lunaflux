# MiniMax H3 host materialization

This startup-only package binds an immutable, exact component manifest to
approved safetensors shard files and copies mixed F32/BF16 payloads directly into final
bounded host arenas. It admits only the complete 638-tensor FL2VA/Ref2VA
denoiser manifests, the complete 1,058-tensor Qwen3-VL conditioner manifest,
the 703-tensor VideoVae manifest, and the 1,087-tensor AudioVae manifest.

The manifest records model/plan association, exact names, official shapes and
dtypes, shard/index membership, payload ranges, declared artifact labels, and
deterministic arena placement. Loading reuses its construction-time label;
it does not reserialize and rehash the manifest or authenticate its payloads.
Bounds remain checked before allocation. The result remains component-scoped
and explicitly releasable.

VAE materialization remains component-scoped; this package never marks the
complete MiniMax H3 model executable. It provides no VAE encode/decode, device
upload, kernel execution, scheduler, codec, media preprocessing, or request
protocol.

`model/materialize` is intentionally not reused: its public contract is bound
to Llama weights and a single in-memory byte view. File inspection, bounded
header parsing, same-file checks, and segmented direct
copy belong to the family-neutral `model/streaming_safetensors` package. It
keeps 64-bit source offsets and never forms whole-file `Bytes` or hashes weight
payloads. Loading reads only selected slices into final MiniMax-owned arenas.
Declared inventory labels do not prove payload integrity.
