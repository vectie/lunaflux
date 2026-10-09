///|
test "F32 pointwise Conv1d binds exact channel-major geometry" {
  let source = try! f32_pointwise_conv1d_source(
    batch_count=2,
    input_channels=32,
    output_channels=2048,
    time_count=207,
    function_symbol="audio_dec_in_proj",
  )
  assert_eq(source.operation(), F32PointwiseConv1d)
  assert_eq(source.numerical_contract(), OrderedF32BiasAccumulationF32OutputV1)
  assert_eq(source.input_element_count(), 13248L)
  assert_eq(source.output_element_count(), 847872L)
  let text = @utf8.decode(source.source_bytes()) catch {
    _ => abort("invalid UTF-8")
  }
  assert_true(text.contains("channel < 32"))
  assert_true(text.contains("__fmul_rn"))
  assert_true(text.contains("__fadd_rn"))
}

///|
test "F32 pointwise Conv1d rejects hostile geometry and symbols" {
  try
    f32_pointwise_conv1d_source(
      batch_count=0,
      input_channels=32,
      output_channels=2048,
      time_count=207,
      function_symbol="x",
    )
  catch {
    InvalidBatchCount => ()
    _ => fail("wrong batch rejection")
  } noraise {
    _ => fail("zero batch admitted")
  }
  try
    f32_pointwise_conv1d_source(
      batch_count=2,
      input_channels=0,
      output_channels=2048,
      time_count=207,
      function_symbol="x",
    )
  catch {
    InvalidChannelCount => ()
    _ => fail("wrong channel rejection")
  } noraise {
    _ => fail("zero channel admitted")
  }
  try
    f32_pointwise_conv1d_source(
      batch_count=2,
      input_channels=32,
      output_channels=2048,
      time_count=0,
      function_symbol="x",
    )
  catch {
    InvalidTimeCount => ()
    _ => fail("wrong time rejection")
  } noraise {
    _ => fail("zero time admitted")
  }
  try
    f32_pointwise_conv1d_source(
      batch_count=2147483647,
      input_channels=2147483647,
      output_channels=2048,
      time_count=2147483647,
      function_symbol="x",
    )
  catch {
    DimensionOverflow => ()
    _ => fail("wrong overflow rejection")
  } noraise {
    _ => fail("overflow admitted")
  }
  try
    f32_pointwise_conv1d_source(
      batch_count=2,
      input_channels=32,
      output_channels=2048,
      time_count=207,
      function_symbol="7bad",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("wrong symbol rejection")
  } noraise {
    _ => fail("leading digit admitted")
  }
}
