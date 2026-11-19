# MiniCPM-V 4.7 packed vision layout

`Layout` owns crop-isolated encoder spans, learned absolute position IDs,
2x2 early-window order and final merger order. It accepts the processor's
patch grids and prepares immutable integer metadata once before vision
execution. It has no device handles, actor tick, camera or codec dependency.

```mbt check
///|
async test "encoder metadata from decoded-image geometry" {
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let image = @processor.Plan::build(spec, height=896, width=896)
  let layout = @minicpm_v47_vision.Layout::build(
    spec,
    image.target_sizes()[:],
    maximum_patches=16384,
    maximum_crop_patches=4096,
  )
  assert_eq(layout.raw_spans().rows(), 5120)
  assert_eq(layout.encoded_spans().rows(), 1280)
  assert_eq(layout.output_rows(), 320)
}
```

In default16x mode, all27 encoder blocks run, with an additional attention/MLP
window merger after block index6. It gathers raster patches into groups of
four, attends inside each group and emits one raster-ordered row per2x2
window. Remaining blocks use the reduced crop spans. The final MLP merger
gathers another2x2 group and projects4608 channels into2048 LLM channels.
4x mode skips the early merger, retaining all encoder rows until the final MLP.
The image processor retains the same resize geometry for either mode.

Full attention never crosses a crop boundary. Thumbnails and detail slices are
separate crops. Learned positional lookup uses70x70 nearest buckets and resets
per crop; it is separate from decoder canvas MRoPE. The explicit F32
bucketize-right profile needs upstream CPU/CUDA arange comparison at ties
before claiming exact upstream position equivalence.

The model uses16 attention heads of width72. `attention_plan` supplies the
generic segmented-attention precision contract without paged KV, rotary or
causal assumptions. Layout construction rejects incomplete spatial merges and
over-capacity input; it never truncates or substitutes a smaller model.

References: pinned Transformers v5.18.0
[vision model](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/minicpmv4_7/modeling_minicpmv4_7.py)
and [vision utilities](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/vision_utils.py).
