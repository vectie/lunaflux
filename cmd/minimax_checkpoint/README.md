# H3 checkpoint entry point

`config VARIANT_ROOT` parses the actual pipeline and transformer config.
`weight-plan VARIANT_ROOT` reports exact packed component weight bytes from
that config; it excludes execution workspace, outputs and transfer staging.
`inspect-component VARIANT_ROOT COMPONENT INVENTORY COMPONENT_LIMIT DEVICE_LIMIT`
binds all real component tensors and resolves complete device packing without
opening CUDA. Inventories live under the variant root; their shard names are
relative to the selected component directory (`video_vae/source` for VideoVAE).

`prepare-component` uses the same arguments, streams that component into an
explicit aggregate GPU owner with one-MiB staging, then releases it. It is a
weight-bootstrap test, not a generated media request or a performance result.
Whole-request preparation must additionally budget execution workspaces,
conditioning/latent transfer and actual decoded outputs before device upload.
