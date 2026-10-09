# Adapter migration notes

These migrations are deliberately deferred until each family package is no
longer under active development. The neutral package must not import any of
them.

## DeepSeek V4

Register these exact dtype rules:

- `BF16` → `bfloat16()`
- `F32` → `float32()`
- `I64` → `opaque_fixed_width("I64", 8)`
- `I8`, `F8_E4M3`, and `F8_E8M0` → `opaque_one_byte(...)`

The adapter should translate each authority-free streaming tensor into the
existing `DeepSeekV4SafetensorsTensor` constructor using the shard ordinal,
payload-relative offsets, exact dtype tag, and shape, then run the existing
semantic binder unchanged. The binder remains responsible for distinguishing
packed FP4-in-I8, FP8 parameters, and UE8M0 scale roles. Copy requests should
come from accepted bindings and target the family-owned bounded host regions.
No FP8/FP4 interpretation belongs in this adapter.

## GLM 5.3

Register `BF16` and `F32` with the built-in rules and `F8_E4M3` as an opaque
one-byte rule. Translate metadata to `Glm53ObservedTensor` and retain the
existing exact manifest-order/name/dtype/shape binding. The current
`glm53_host_materialize` source/digest/header loop can then be replaced by
`inspect_shards` plus manifest-derived copy requests, while its numeric-plan,
arena-layout, identity, and release contracts remain family-owned. Do not map
FP8 payload bytes to BF16 or claim scale interpretation.

## MiniMax H3

The currently admitted component manifests are BF16-only, so register only
`bfloat16()`. Translate streaming metadata to the existing BF16 safetensors
metadata view, run the component semantic binder and exact shard-index
membership checks, then derive copy requests from the already bounded arena
layout. Preserve separate FL2VA, Ref2VA, text-conditioner, video-VAE, and
audio-VAE ownership; do not combine them into a synthetic complete model.

For every family, the expected complete-file digest and approved relative
locator become `StreamingSafetensorsShardInput`s. A successful adapter must
publish family metadata or host ownership only after `inspect_shards` or
`copy_ranges` returns successfully; destination spans are discarded on error.
