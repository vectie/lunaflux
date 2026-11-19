///|
async test "all vision operands bind actual headers and use one bounded transfer session" {
  @fixture.with_checkpoint((spec, checkpoint, root) => {
    let program = @minicpm_v47_vision.Program::new(
      spec,
      maximum_patches=16384,
      maximum_crop_patches=4096,
    )
    assert_eq(program.storage_bytes(), 429178928L)
    let plan = @minicpm_v47_vision.WeightPlan::new(spec, program, checkpoint)
    assert_eq(plan.entries().length(), 459)
    assert_eq(plan.byte_count(), 1105972064L)
    assert_eq(
      plan.entries()[0].name(),
      "model.vision_tower.embeddings.patch_embedding.weight",
    )
    let seen = Ref(0)
    let valid = Ref(false)
    assert_true(
      try
        plan.transfer(
          root,
          scratch=FixedArray::make(3, b'\xff'),
          limits=@fixture.limits(),
          consume=(ordinal, offset, bytes, count) => {
            valid.val = ordinal == 0 &&
              offset == 0L &&
              count == 3 &&
              bytes[0] == b'\x00' &&
              bytes[1] == b'\x00' &&
              bytes[2] == b'\x00'
            seen.val += 1
            false
          },
        )
      catch {
        _ => true
      } noraise {
        _ => false
      },
    )
    assert_eq(seen.val, 1)
    assert_true(valid.val)
  })
}

///|
async test "input preparation joins pixels and model layout before native work" {
  let spec = selected_spec()
  let program = @minicpm_v47_vision.Program::new(
    spec,
    maximum_patches=1024,
    maximum_crop_patches=1024,
  )
  let pixel_plan = @processor.Plan::build(spec, height=56, width=56)
  let pixels = pixel_plan.prepare_rgb(
    Bytes::from_array(Array::make(56 * 56 * 3, b'\x80'))[:],
    max_bytes=10000000L,
  )
  let layout = @vision.Layout::build(
    spec,
    pixel_plan.target_sizes()[:],
    maximum_patches=1024,
    maximum_crop_patches=1024,
  )
  let input = @minicpm_v47_vision.Input::new(program, layout, pixels)
  assert_eq(input.patch_rows(), 1024)
  assert_eq(input.feature_rows(), 64)
  let too_small = @minicpm_v47_vision.Program::new(
    spec,
    maximum_patches=512,
    maximum_crop_patches=512,
  )
  assert_true(
    try @minicpm_v47_vision.Input::new(too_small, layout, pixels) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  let short_segment = @minicpm_v47_vision.Program::new(
    spec,
    maximum_patches=2048,
    maximum_crop_patches=512,
  )
  assert_true(
    try @minicpm_v47_vision.Input::new(short_segment, layout, pixels) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}
