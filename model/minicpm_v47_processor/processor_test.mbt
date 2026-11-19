///|
async fn selected_spec() -> @model.Spec {
  @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
}

///|
async test "thumbnail upscale and exact slice geometry" {
  let spec = selected_spec()
  let small = @minicpm_v47_processor.Plan::build(spec, height=224, width=224)
  assert_eq(small.slice_grid(), (0, 0))
  assert_eq(small.crops().length(), 1)
  assert_eq(small.crops()[0].size.height, 448)
  assert_eq(small.crops()[0].size.width, 448)
  assert_eq(small.visual_tokens(), 64)
  assert_eq(small.pixel_columns(), 14336)
  assert_eq(small.pixel_bytes(), 1204224L)
  let wide = @minicpm_v47_processor.Plan::build(spec, height=448, width=896)
  assert_eq(wide.slice_grid(), (1, 2))
  assert_eq(wide.crops()[0].size.height, 336)
  assert_eq(wide.crops()[0].size.width, 616)
  assert_eq(wide.crops()[2].left, 448)
  assert_eq(wide.crops()[2].top, 0)
  assert_eq(wide.crops()[2].canvas.width, 896)
  assert_eq(wide.visual_tokens(), 194)
  let tall = @minicpm_v47_processor.Plan::build(spec, height=896, width=448)
  assert_eq(tall.slice_grid(), (2, 1))
  assert_eq(tall.crops()[0].size.height, 616)
  assert_eq(tall.crops()[0].size.width, 336)
  assert_eq(tall.crops()[2].top, 448)
  assert_eq(tall.crops()[2].left, 0)
  let single = @minicpm_v47_processor.Plan::build(
    spec,
    height=896,
    width=896,
    max_slices=1,
  )
  assert_eq(single.slice_grid(), (0, 0))
  assert_eq(single.visual_tokens(), 64)
}

///|
async test "placeholder newline layout feeds the existing canvas position owner" {
  let spec = selected_spec()
  let plan = @minicpm_v47_processor.Plan::build(spec, height=896, width=896)
  assert_eq(plan.slice_grid(), (2, 2))
  assert_eq(plan.visual_tokens(), 320)
  let fragment = plan.placeholder()
  assert_eq(fragment.tokens().length(), 331)
  assert_eq(fragment.grids().length(), 5)
  assert_eq(
    fragment.modalities().filter(m => m == @positions.Modality::Image).length(),
    320,
  )
  assert_eq(
    fragment.tokens().filter(t => t == spec.markers().newline).length(),
    1,
  )
  // Two adjacent slices have no newline; only detail row boundaries do.
  assert_eq(fragment.tokens()[132], spec.markers().slice_start)
  assert_eq(fragment.tokens()[198], spec.markers().newline)
  assert_eq(fragment.tokens()[199], spec.markers().slice_start)
  let sequence = @positions.Sequence::build(
    spec,
    fragment.tokens()[:],
    fragment.modalities()[:],
    image_grids=fragment.grids()[:],
    video_grids=[],
  )
  assert_eq(sequence.prompt_tokens(), 331)
  assert_true(sequence.rope_delta() < 0)
  let text = plan.placeholder_text(image_id=3)
  assert_true(text.has_prefix("<image_id>3</image_id><image>"))
  assert_true(text.has_suffix("</slice>"))
}

///|
fn bf16(data : Bytes, offset : Int32) -> UInt32 {
  data[offset * 2].to_uint() | (data[offset * 2 + 1].to_uint() << 8)
}

///|
async test "RGB channel normalization survives thumbnail and detail packing" {
  let spec = selected_spec()
  let plan = @minicpm_v47_processor.Plan::build(spec, height=56, width=56)
  let input = FixedArray::make(56 * 56 * 3, b'\x00')
  for pixel in 0..<(56 * 56) {
    input[pixel * 3] = b'\x00'
    input[pixel * 3 + 1] = b'\x80'
    input[pixel * 3 + 2] = b'\xff'
  }
  let pixels = plan.prepare_rgb(
    Bytes::from_array(input[:])[:],
    max_bytes=10000000L,
  )
  assert_eq(pixels.data().length().to_int64(), plan.pixel_bytes())
  let plane = 14 * plan.pixel_columns()
  for i in 0..<plane {
    assert_eq(bf16(pixels.data(), i), 0xbf80U)
    assert_eq(bf16(pixels.data(), plane + i), 0x3b81U)
    assert_eq(bf16(pixels.data(), plane * 2 + i), 0x3f80U)
  }
}

///|
async test "identity resize retains NaViT channel and intra-patch ordering" {
  let spec = selected_spec()
  let plan = @minicpm_v47_processor.Plan::build(spec, height=448, width=448)
  let input = Bytes::from_array(
    Array::makei(448 * 448 * 3, i => (i % 256).to_byte()),
  )
  let pixels = plan.prepare_rgb(input[:], max_bytes=10000000L)
  for y in [0, 1, 13, 14, 27, 223, 447] {
    for x in [0, 1, 13, 14, 27, 223, 447] {
      for channel in 0..<3 {
        let patch = y / 14 * 32 + x / 14
        let index = (channel * 14 + y % 14) * plan.pixel_columns() +
          patch * 14 +
          x % 14
        let value = input[(y * 448 + x) * 3 + channel].to_int().to_double() *
          (1.0 / 255.0)
        let normalized = (value.to_float() - 0.5F) / 0.5F
        let bits = normalized.reinterpret_as_uint()
        let expected = (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16
        assert_eq(bf16(pixels.data(), index), expected)
      }
    }
  }
}

///|
async test "invalid geometry and numeric payload budget fail before processing" {
  let spec = selected_spec()
  assert_true(
    (try? @minicpm_v47_processor.Plan::build(spec, height=0, width=224))
    is Err(_),
  )
  assert_true(
    (try? @minicpm_v47_processor.Plan::build(
      spec,
      height=448,
      width=448,
      max_slices=10,
    ))
    is Err(_),
  )
  let plan = @minicpm_v47_processor.Plan::build(spec, height=56, width=56)
  assert_true((try? plan.prepare_rgb(b"", max_bytes=10000000L)) is Err(_))
  let input = Bytes::from_array(Array::make(56 * 56 * 3, b'\x00'))
  assert_true((try? plan.prepare_rgb(input[:], max_bytes=1L)) is Err(_))
  assert_true((try? plan.placeholder_text(image_id=-1)) is Err(_))
}

///|
async test "fourfold compression changes placeholder count without changing pixel geometry" {
  let text = @utf8.decode(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary(),
  )
  let four = @model.Spec::parse(
    @utf8.encode(text.replace(old="\"16x\"", new="\"4x\""))[:],
  )
  let sixteen = selected_spec()
  let a = @minicpm_v47_processor.Plan::build(four, height=896, width=896)
  let b = @minicpm_v47_processor.Plan::build(sixteen, height=896, width=896)
  assert_eq(a.pixel_bytes(), b.pixel_bytes())
  assert_eq(a.visual_tokens(), b.visual_tokens() * 4)
  assert_eq(a.placeholder().tokens().length(), 1291)
}
