# MiniCPM-V 4.7 image-to-decoder bridge

This package owns image prompt binding and one prefill insertion stage. It
reuses the full 40-layer decoder; it imports no camera/HTTP/robot implementation.

`prepare_image_prompt` follows the pinned stateless system/user template:
content-list parts join with newlines, local image IDs start at zero, outer
content is trimmed and the no-thinking assistant prefix is closed. The complete
prompt is tokenized once. Explicit image plans supply grids/counts; extra literal
image placeholders in text are rejected. `prepare_image_tokens` accepts native,
already-templated IDs using the same binding law.

```mbt check
///|
async test "bind image features and canvas positions without a device" {
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let image = @processor.Plan::build(spec, height=448, width=448)
  let tokens = [100] + image.placeholder().tokens()[:].to_owned()
  let prompt = @minicpm_v47_multimodal.prepare_image_tokens(
    spec,
    tokens[:],
    [image],
    maximum_tokens=4096,
  )
  assert_eq(prompt.feature_rows(), 64)
  assert_eq(prompt.feature_map()[2], 0)
  assert_eq(prompt.feature_map()[65], 63)
  assert_true(prompt.prompt().positions().rope_delta() < 0)
}
```

Feature rows follow image/crop/placeholder occurrence order. The map uses
absolute causal token positions, independently of canvas MRoPE, so insertion
works across prefill chunks. Vision `Input::from_images` concatenates NaViT
columns in this order, retaining isolated crop attention.

`Program` emits `image_scatter.cu`. `Frame` borrows text launches/storage and the
vision feature allocation, owns one function/two metadata buffers and inserts
raw BF16 feature copying immediately after token embedding. Prefill has 925
launches; decode keeps its original 924. At context4096, map/header storage is
16,388 bytes. Binding a completed FeatureView makes two request-level metadata
uploads and no feature readback. Request steps make no extra copies/allocations;
decode does not recompute or reinsert images.

Use `Runner::prepare_launches` with this frame's two launch spans. The original
`Runner::prepare` remains the text-frame convenience path; its host progress,
cache, greedy and stop laws stay unchanged.

Wait for vision completion and retire the prior text request before
`bind_prompt`. Finish consuming feature storage before reusing vision. Call
`clear_prompt` before a text-only request. Close Runner before this Frame, then
the borrowed text/vision owners.

The scripted native fixture joins vision output to the existing text runner,
covering three prefill chunks/decode, two mapping uploads, zero-allocation
request steps, text reuse and cancellation. It computes no model arithmetic.
The generated-source CPU ASan/UBSan probe checks raw BF16 copy, causal indexing
and text/decode preservation, not NVCC or GPU behavior. The separate
[image runtime](../minicpm_v47_image_runtime/README.mbt.md) now composes both
branches and enforces feature lifetime. Executable/media serving and upstream
numerical/quality checks remain pending.

Reference: pinned
[chat template](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B/blob/98e79ba972b586ac6146d3505771ce1467e199ba/chat_template.jinja)
and [feature insertion](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/minicpmv4_7/modeling_minicpmv4_7.py).
