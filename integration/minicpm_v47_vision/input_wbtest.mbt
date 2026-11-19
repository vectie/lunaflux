///|
async test "multiple images concatenate NaViT columns rather than complete blobs" {
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let program = Program::new(
    spec,
    maximum_patches=2048,
    maximum_crop_patches=1024,
  )
  let plan = @processor.Plan::build(spec, height=56, width=56)
  let dark = plan.prepare_rgb(
    Bytes::from_array(Array::make(56 * 56 * 3, b'\x00'))[:],
    max_bytes=10000000L,
  )
  let light = plan.prepare_rgb(
    Bytes::from_array(Array::make(56 * 56 * 3, b'\xff'))[:],
    max_bytes=10000000L,
  )
  let grids = plan.target_sizes()[:].to_owned() +
    plan.target_sizes()[:].to_owned()
  let layout = @vision.Layout::build(
    spec,
    grids[:],
    maximum_patches=2048,
    maximum_crop_patches=1024,
  )
  let input = Input::from_images(program, layout, [dark, light])
  assert_eq(input.patch_rows(), 2048)
  assert_eq(input.feature_rows(), 128)
  let data = input.uploads[3].bytes
  let single = plan.pixel_columns() * 2
  for plane in 0..<42 {
    let first = plane * single * 2
    assert_eq(data[first], b'\x80')
    assert_eq(data[first + 1], b'\xbf')
    assert_eq(data[first + single - 1], b'\xbf')
    assert_eq(data[first + single], b'\x80')
    assert_eq(data[first + single + 1], b'\x3f')
    assert_eq(data[first + single * 2 - 1], b'\x3f')
  }
}
