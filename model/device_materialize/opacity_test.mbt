///|
test "opaque weight layout and limits preserve checked read-only views" {
  let plan = numeric_layout_plan(3)
  let layout = try! @device_materialize.plan_numeric_layout(
    plan,
    alignment_bytes=64L,
    max_arena_bytes=1024L,
  )
  assert_eq(layout.identity(), plan.identity())
  assert_eq(layout.alignment_bytes(), 64L)
  assert_eq(layout.length(), plan.tensor_count())
  guard layout.region(try! @plan.TensorRef::new(2)) is Some(region) else {
    fail("opaque layout lost its final region")
  }
  assert_eq(region.reference().as_int(), 2)
  assert_eq(region.byte_length(), 32L)
  assert_eq(region.end_offset_bytes(), layout.arena_bytes())

  let safetensors = try! @safetensors.HeaderLimits::new(
    max_header_bytes=4096,
    max_tensor_count=32,
    max_rank=2,
    max_name_length=128,
    max_payload_bytes=8192L,
    max_materialized_tensor_bytes=8192L,
  )
  let limits = try! @device_materialize.DeviceWeightFileLimits::new(
    max_file_bytes=16384L,
    transfer_chunk_bytes=4096,
    safetensors~,
  )
  assert_eq(limits.max_file_bytes(), 16384L)
  assert_eq(limits.transfer_chunk_bytes(), 4096)
  assert_eq(limits.safetensors(), safetensors)
}

///|
test "opaque file limits retain hostile validation" {
  let safetensors = try! @safetensors.HeaderLimits::new(
    max_header_bytes=4096,
    max_tensor_count=32,
    max_rank=2,
    max_name_length=128,
    max_payload_bytes=8192L,
    max_materialized_tensor_bytes=8192L,
  )
  try
    @device_materialize.DeviceWeightFileLimits::new(
      max_file_bytes=8L,
      transfer_chunk_bytes=4096,
      safetensors~,
    )
  catch {
    InvalidFileLimit(FileBytes) => ()
    _ => fail("unexpected file-limit error")
  } noraise {
    _ => fail("invalid file bound was accepted")
  }
}
