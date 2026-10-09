# H3 offline source export

`build` calls the real source generators from the admitted model metadata,
resolved request, packed sequence, parsed audio VAE configuration and text plan.
It walks the complete denoising schedule and exports every distinct timestep
count. Outputs cover text, optional Qwen-VL vision, context/refiner, DiT input,
block and heads, both RF updates, latent permutation, all seven audio
upsample/AMP stages and tail, and tiled video decode/canonical output.
`with_reference_audio` and `with_reference_video` add real reference encoders.

The parsed text configuration is mandatory: text epsilon must match it and
the row count must fit its position bound. Packed target video/audio extents
and hidden width must match the resolved request/model. Optional vision output
rows must match unique in-range text-row overrides. This checks allocation and
schedule consistency, not token placement: no input IDs are supplied here.

Every `SourceUnit` is a separate CUDA compilation unit. Fixed ABI symbols such
as the permutation and RF update symbols intentionally recur between modules;
do not concatenate the sources. `SOURCES.v1` records deterministic source hashes,
symbols, shape metadata and required timestep variants. It is not a compiled
artifact, an admission receipt or a physical validation claim.

`SourceExport.write(existing_empty_directory)` writes exclusive read-only `.cu`
files and writes the inventory last. Partial failures preserve partial files.
No runtime JIT or `nvcc` invocation is included.

The executable `integration/minimax_h3_source_export/cli` provides the ordinary
text-to-video/audio export path. Arguments are:

```text
EXPORT_DIR TRANSFORMER_JSON AUDIO_VAE_JSON TEXT_CONFIG_JSON CONTENT_SHA256
HEIGHT WIDTH FRAMES STEPS TEXT_ROWS WORKSPACE_BYTES
```

The digest must come from the caller's admitted model inventory. This command
does not authenticate the inventory itself. Reference-media and visual-text
exports currently use the public library APIs rather than CLI flags.
The text source remains bounded to 83,886 rows by the generic MLP renderer.
These are correctness-oriented generated kernels, not performance claims.
