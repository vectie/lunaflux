///|
async fn selected_spec() -> @model.Spec {
  @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
}

///|
async test "packed crops isolate attention and retain both spatial merge orders" {
  let spec = selected_spec()
  let layout = @minicpm_v47_vision.Layout::build(
    spec,
    [{ height: 4, width: 4, }, { height: 4, width: 8, }],
    maximum_patches=64,
    maximum_crop_patches=32,
  )
  assert_eq(layout.raw_spans().boundaries(), [0, 16, 48])
  assert_eq(layout.encoded_spans().boundaries(), [0, 4, 12])
  assert_eq(layout.output_boundaries(), [0, 1, 3])
  assert_eq(layout.output_rows(), 3)
  assert_eq(layout.insert_after(), Some(6))
  assert_eq(layout.layer_rows(6), 48)
  assert_eq(layout.layer_rows(7), 12)
  assert_eq(layout.layer_rows(26), 12)
  assert_eq(layout.raw_spans().starts()[16], 16)
  assert_eq(layout.raw_spans().ends()[15], 16)
  assert_eq(layout.window_indices()[:16].to_owned(), [
    0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15,
  ])
  assert_eq(layout.window_indices()[16:20].to_owned(), [16, 17, 24, 25])
  assert_eq(layout.window_spans().starts()[:8].to_owned(), [
    0, 0, 0, 0, 4, 4, 4, 4,
  ])
  assert_eq(layout.window_spans().ends()[:8].to_owned(), [
    4, 4, 4, 4, 8, 8, 8, 8,
  ])
  assert_eq(layout.merger_indices(), [0, 1, 2, 3, 4, 5, 8, 9, 6, 7, 10, 11])
  assert_eq(layout.position_ids()[:4].to_owned(), [0, 17, 35, 52])
  assert_eq(layout.position_ids()[4], 17 * 70)
  assert_eq(layout.position_ids()[16], 0)
}

///|
async test "processor target sizes join actual selected encoder and head72" {
  let spec = selected_spec()
  let image = @processor.Plan::build(spec, height=896, width=896)
  let layout = @minicpm_v47_vision.Layout::build(
    spec,
    image.target_sizes()[:],
    maximum_patches=8192,
    maximum_crop_patches=2048,
  )
  assert_eq(layout.raw_spans().rows(), 5120)
  assert_eq(layout.encoded_spans().rows(), 1280)
  assert_eq(layout.output_rows(), image.visual_tokens())
  let attention = @minicpm_v47_vision.attention_plan(
    spec,
    maximum_patches=8192,
    maximum_crop_patches=2048,
  )
  assert_eq(attention.geometry(), (8192, 2048, 16, 72))
  assert_eq(layout.position_ids().length(), 5120)
  let largest = @processor.Plan::build(spec, height=1792, width=1792)
  let all_crops = @minicpm_v47_vision.Layout::build(
    spec,
    largest.target_sizes()[:],
    maximum_patches=16384,
    maximum_crop_patches=4096,
  )
  assert_eq(largest.crops().length(), 10)
  assert_eq(all_crops.raw_spans().rows(), 10240)
  assert_eq(all_crops.output_rows(), 640)
}

///|
async test "4x keeps all encoder rows and skips early window attention" {
  let config = @utf8.decode(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary(),
  )
  let spec = @model.Spec::parse(
    @utf8.encode(config.replace(old="\"16x\"", new="\"4x\""))[:],
  )
  let layout = @minicpm_v47_vision.Layout::build(
    spec,
    [{ height: 4, width: 4, }],
    maximum_patches=16,
    maximum_crop_patches=16,
  )
  assert_eq(layout.insert_after(), None)
  assert_eq(layout.layer_rows(26), 16)
  assert_eq(layout.encoded_spans().rows(), 16)
  assert_eq(layout.output_rows(), 4)
  assert_eq(layout.window_spans().rows(), 0)
  assert_true(layout.window_indices().is_empty())
  assert_eq(layout.merger_indices(), [
    0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15,
  ])
}

///|
async test "layout rejects incomplete merge grids and capacity overflow" {
  let spec = selected_spec()
  let cases : Array[Array[@positions.Grid]] = [
    [],
    [{ height: 2, width: 4, }],
    [{ height: 4, width: 0, }],
    [{ height: 8, width: 8, }],
  ]
  for grids in cases {
    assert_true(
      (try? @minicpm_v47_vision.Layout::build(
        spec,
        grids[:],
        maximum_patches=32,
        maximum_crop_patches=32,
      ))
      is Err(_),
    )
  }
  let layout = @minicpm_v47_vision.Layout::build(
    spec,
    [{ height: 4, width: 4, }],
    maximum_patches=16,
    maximum_crop_patches=16,
  )
  assert_true((try? layout.layer_rows(27)) is Err(_))
}
