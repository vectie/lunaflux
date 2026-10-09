///|
test "exact alias-free activation geometry and distinct symbols" {
  let source = try! @alias_free_activation_cuda_source.f32_alias_free_snake_beta_source(
    batch_count=2,
    channel_count=512,
    time_count=1035,
    ratio=2,
    kernel_size=12,
    upsample_symbol="upsample",
    activation_symbol="snake_beta",
    downsample_symbol="downsample",
  )
  assert_eq(source.upsample_time(), 2070)
  assert_eq(source.input_element_count(), 1059840L)
  assert_eq(source.intermediate_element_count(), 2119680L)
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("t+15"))
  assert_true(text.contains("t*2+k-5"))
}

///|
test "alias-free activation rejects hostile geometry and symbols" {
  for symbol in ["", "7bad", "bad-name"] {
    try
      @alias_free_activation_cuda_source.f32_alias_free_snake_beta_source(
        batch_count=1,
        channel_count=1,
        time_count=1,
        ratio=2,
        kernel_size=12,
        upsample_symbol=symbol,
        activation_symbol="act",
        downsample_symbol="down",
      )
    catch {
      InvalidFunctionSymbol => ()
      _ => fail("wrong symbol error")
    } noraise {
      _ => fail("hostile symbol admitted")
    }
  }
  try
    @alias_free_activation_cuda_source.f32_alias_free_snake_beta_source(
      batch_count=1,
      channel_count=1,
      time_count=1073741824,
      ratio=2,
      kernel_size=12,
      upsample_symbol="up",
      activation_symbol="act",
      downsample_symbol="down",
    )
  catch {
    DimensionOverflow => ()
    _ => fail("wrong overflow error")
  } noraise {
    _ => fail("overflow admitted")
  }
}
