# Qwen3 numeric-weight converter

This offline native command converts the pinned single-file Qwen3-0.6B BF16
safetensors artifact into LunaFlux `lunaflux.numeric-safetensors.v1` ordinal
order. It accepts only the exact pinned configuration and 311-tensor source,
proves that `lm_head.weight` and `model.embed_tokens.weight` are byte-identical,
and emits the 310 unique storage tensors without the redundant LM-head copy.

```sh
moon run --target native cmd/lunaflux_qwen3_weight_convert -- \
  MODEL_ROOT CONFIG_LOCATOR CONFIG_SHA256 MODEL_CONTENT_SHA256 \
  SOURCE_LOCATOR SOURCE_SHA256 MAX_BATCH_ROWS OUTPUT_LOCATOR
```

All three locators are direct-child filenames under the same canonical absolute
model root. The output is create-new and is never overwritten. Reads and copies
use one MiB bounded buffers; the command performs no device or token-step work.
On success it reports the authenticated source artifact/manifest digests and
the admitted output artifact/route-manifest digests. `MAX_BATCH_ROWS` is part of
the authenticated model-plan identity: use `1` for the c1 correctness profile
or `32` for the c32 benchmark profile. An artifact converted for one capacity
cannot be substituted into the other.
