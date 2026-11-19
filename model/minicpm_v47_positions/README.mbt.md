# MiniCPM-V 4.7 canvas positions

`Sequence::build` prepares one unpadded prompt's row-major T/H/W coordinates and
`rope_delta`. It follows the Transformers 5.18.0 MiniCPM-V 4.7 canvas structure:
visual runs become thumbnail/detail crops, marker-adjacent frames share groups,
thumbnail coordinates stretch over their slice canvas, slice markers pin crop
corners, and row newlines sit beyond the canvas's right edge. Image and video
crop grids have independent occurrence-ordered streams. The per-call downsample
setting determines whether each LLM token covers 2x2 or 4x4 patches.

Modality labels come from the processor. The builder does not guess modality
from token IDs, execute vision, resize images or treat visual tokens as ordinary
text. Callers compact padding first and retain the mapping to their scheduler's
CSR rows. Missing grids and mismatched placeholder counts are errors.

The immutable prompt is prepared once. `write_prompt_into` stages a chunk and
`write_decode_into` stages appended text into preallocated caller buffers without
allocating. They write causal/cache addresses separately from the three rotary
coordinates. Decode uses `causal_position + rope_delta` for all three rotary
axes; causal addresses always count actual tokens. The selected rotary capacity
is 262144; cache-context admission is a separate caller responsibility.

Analytic fixtures cover text, thumbnails, nearest-even half coordinates, detail
slices/newlines, image/video groups, separate groups, chunking and continuation.
Grid resampling preserves the upstream FP32 linspace step, symmetric endpoint
calculation and nearest-even rounding. Exact rational arithmetic is different:
15 samples over extent118 put sample7 at59 in ATen, versus58 with rational
rounding. The offline LibTorch 2.9.1 CPU oracle matches 533,500 coordinates across
8,196 cases through the public prompt API, on both spatial axes. This qualifies
that primitive and carrier, not full upstream grouping/processor or CUDA parity.
No Python or LibTorch is required by this production package.

A separate full-sequence oracle executes the unchanged Transformers 5.18.0
grouping/canvas methods with CPU Torch 2.9.1. All 154 unpadded batch-one cases match
751,830 prompt coordinates, 1,848 decode coordinates, every rope delta and 32-row
staging. It includes adjacent/separate mixed image/video frames, detail slices,
six processor-derived image geometries and both compression modes. Crop grids
are shared inputs, so this does not qualify geometry/pixel processing, masked/
batched padding or actual CUDA/model execution. See
`tests/minicpm_v47_canvas_oracle/README.md` for the separate offline dependency
and reproducible command.

Run the optional offline oracle with an explicit LibTorch installation:

```sh
moon run scripts/verify-minicpm-v47-positions.mbtx ABSOLUTE_REPO LIBTORCH_ROOT NEW_OUTPUT
```

The installation needs `include/` and `lib/`. Only the separate test executable
links ATen; `comparison/report.json` records the actual version and scoped result.

`integration/minicpm_v47.prepare_text_prompt` joins the actual tokenizer and
initial text chat template with this position owner. Visual prompt construction
and whole-model execution remain separate integration work.
