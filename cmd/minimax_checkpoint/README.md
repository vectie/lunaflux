# H3 checkpoint entry point

`config VARIANT_ROOT` parses the actual pipeline and transformer config.
`weight-plan VARIANT_ROOT` reports exact packed component weight bytes from
that config; it excludes execution workspace, outputs and transfer staging.
`audio-config` and `video-config` resolve the original checkpoint's wrapper and
source data documents to native numeric contracts without executing Python.
`inspect-component VARIANT_ROOT COMPONENT INVENTORY COMPONENT_LIMIT DEVICE_LIMIT`
binds all real component tensors and resolves complete device packing without
opening CUDA. Inventories live under the variant root; their shard names are
relative to the selected component directory (`video_vae/source` for VideoVAE).

`prepare-component` uses the same arguments, streams that component into an
explicit aggregate GPU owner with one-MiB staging, then releases it. It is a
weight-bootstrap test, not a generated media request or a performance result.
Whole-request preparation must additionally budget execution workspaces,
conditioning/latent transfer and actual decoded outputs before device upload.

`export-audio VARIANT_ROOT HEIGHT WIDTH FRAMES STEPS EMPTY_OUTPUT_DIRECTORY`
exports the shared decoder AOT sources. `audio-memory` uses the same first five
arguments followed by a workspace-byte limit and reports aggregate retained
scratch/cache/input/output storage. `decode-audio VARIANT_ROOT INVENTORY HEIGHT
WIDTH FRAMES STEPS AOT_DIRECTORY INPUT_F32 OUTPUT_F32 DEVICE_BYTE_LIMIT` runs
the complete checkpoint-backed AudioVAE from an explicit destandardized
channel-major latent input. It does not run text conditioning or the denoiser.

`export-video VARIANT_ROOT HEIGHT WIDTH FRAMES STEPS TILE_SIZE OVERLAP MAX_TILES
WORKSPACE_LIMIT EMPTY_OUTPUT_DIRECTORY` exports the same VideoVAE module as the
full request and reports its exact workspace/input/output byte counts. Tiling
and memory policy are explicit inputs, not model- or machine-hidden defaults.

`export-request VARIANT_ROOT HEIGHT WIDTH FRAMES STEPS TEXT_ROWS TILE_SIZE OVERLAP
MAX_TILES WORKSPACE_LIMIT EMPTY_OUTPUT_DIRECTORY` reads the original audio,
video and Qwen3-VL metadata and exports the complete text-only request module
set: text encoder, refiner, timestep variants, joint denoiser, input/output,
flow updates and both media decoders. Optional image/reference conditioning
still uses the typed source-export API. This command reads metadata only and
opens no CUDA context; compilation does not establish checkpoint execution.

`encode-text VARIANT_ROOT INVENTORY TEXT_ROWS INPUT_IDS_I32 AOT_DIRECTORY
OUTPUT_BF16 COMPONENT_LIMIT DEVICE_LIMIT` runs the original Qwen3-VL checkpoint
through H3's selected unnormalized `hidden_states[50]`. Input contains exactly
TEXT_ROWS signed I32 little-endian token IDs relative to the variant root;
media placeholders require the separate visual-conditioning route. It uses
`text.cubin` from the same full-request export and streams the complete text
component into the explicit weight owner. Weights, request workspace, rotary
tables, metadata and module reserve must fit one aggregate device budget.
Output is exactly `[TEXT_ROWS,5120]` BF16 and is created without overwriting.
Finite-output diagnostics do not establish independent numerical equivalence
or complete text-to-media generation.

`decode-video VARIANT_ROOT INVENTORY HEIGHT WIDTH FRAMES STEPS TILE_SIZE OVERLAP
MAX_TILES AOT_DIRECTORY INPUT_F32 OUTPUT_F32 DEVICE_BYTE_LIMIT` streams the actual
VideoVAE checkpoint and executes all 36 decoder blocks, temporal/spatial assembly
and canonical pixel conversion. Input is packed standardized denoiser-layout F32,
not channel-major destandardized data. It uses the same immutable decoder plan
and AOT source as whole-request export. Weights, input, normalization tables,
request workspace/output and a module/queue reserve share one explicit budget.
The output is channel-major F32 `[1,3,T,H,W]`. An explicit latent fixture is only
a decoder test, not text-to-media inference or independent numerical validation.
