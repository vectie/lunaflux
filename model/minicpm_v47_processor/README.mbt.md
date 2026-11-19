# MiniCPM-V 4.7 image preparation

This package owns decoded RGB8 preparation for the selected **MiniCPM-V 4.7
35B-A3B**, independently of tokenization, device execution, cameras and E1 control.
`Plan::build` supplies an immutable thumbnail/detail layout, encoder patch grids,
compressed visual-token count and typed placeholder body. `prepare_rgb` applies
the selected PIL resize profile, normalization and BF16 packing once before
prefill. It has no Python, Torch or native runtime dependency.

```mbt check
///|
async test "plan image input without allocating pixel payloads" {
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let plan = @minicpm_v47_processor.Plan::build(spec, height=896, width=448)
  assert_eq(plan.slice_grid(), (2, 1))
  assert_eq(plan.crops().length(), 3)
  assert_eq(plan.visual_tokens(), 194)
  let fragment = plan.placeholder()
  assert_eq(fragment.tokens().length(), 201)
}
```

The initial profile follows Transformers v5.18.0's
[PIL image processor](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/minicpmv4_6/image_processing_pil_minicpmv4_6.py),
which the pinned MiniCPM-V 4.7 model family reuses, and its
[replacement framing](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/minicpmv4_7/processing_minicpmv4_7.py).
Geometry uses resolution448, patch14, nearest-even multiples of56, up to nine
detail slices, and row-major slice order. Both compression profiles keep this
resize geometry: 16x yields one embedding per4x4 patches; 4x yields one per2x2.

The pixel payload has shape `[1,3,14,sum(crop_height*crop_width/14)]`, stored as
little-endian BF16. Crops concatenate **within each channel and intra-patch
row**, along the last axis. It is neither ordinary CHW nor a concatenation of
complete crop tensors. Each detail crop indexes one globally resized canvas;
the crop is not independently resized from the original image.

`placeholder()` supplies the image body, explicit Image/Text modalities and
matching grids for `minicpm_v47_positions.Sequence`. Ingress prepends local
`<image_id>N</image_id>` tokens as Text. `placeholder_text(image_id=N)` renders
the matching replacement string. Real-tokenizer integration checks both forms.
The caller accounts for framing/text tokens when checking its context ceiling.

RGB8 bicubic uses antialiasing, coefficient quantization to22 bits and clipping
after each pass, following [Pillow12's resize implementation](https://github.com/python-pillow/Pillow/blob/12.0.0/src/libImaging/Resample.c).
The numeric-payload budget includes construction plus immutable output, a single
crop's horizontal RGB8 scratch and coefficient arrays. It excludes input and
small MoonBit object headers; it is not a total-process RSS limit.

Run from LunaFlux:

```sh
moon test --target native --deny-warn --warn-list '-79-29-25-20' model/minicpm_v47_processor
moon run scripts/verify-minicpm-v47-processor.mbtx ABSOLUTE_REPO NEW_ABSOLUTE_OUTPUT
```

The offline oracle downloads the six unchanged Pillow12 C source/header files
and its license, compiles its RGB8 resampler and compares every prepared BF16
word for five nonuniform images (12,305,664 words). Its local adapter only
provides image memory and no-op section callbacks. Opaque Python declarations
allow upstream headers to compile; no Python interpreter/library is loaded.
Downloaded sources and the oracle never enter production builds.

The separate [full PIL pipeline reference](../../tests/minicpm_v47_pil_pipeline_oracle/README.md)
now invokes unchanged Transformers5.18.0 geometry, `_preprocess`, backend and
transform methods with Pillow12/NumPy. Twelve decoded-RGB8 cases match
22,682,688 BF16 words, slice grids, target sizes, crop counts and output shapes.
This independently checks the original thumbnail/detail decisions and packing,
in addition to the lower-level Pillow C resize check above.

The separate canvas reference covers unpadded position groups. PNG/JPEG ingress,
vision execution and embedding insertion also have software implementations;
their own fixtures do not establish physical CUDA or model quality. Torchvision
profile parity, full JPEG/color-profile parity, video sampling, masked/batched padding,
full-model numerical behavior and target execution remain open.

The separate image-load reference now verifies all eight EXIF orientations
over126 cases, with72 full PNG cases exact against Pillow. The54 JPEG cases
retain a maximum sampled channel difference3 after the progressive entropy
correction; this does not establish full-model numerical equivalence. See the
[ingress reference](../../tests/minicpm_v47_image_load_oracle/README.md).
