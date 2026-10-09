///|
test "weight-normalized Conv1d binds exact staged geometry" {
  let source = try! f32_weight_normalized_conv1d_source(
    batch_count=2,
    input_channels=2048,
    output_channels=1024,
    input_time=207,
    kernel_size=7,
    padding=3,
    stride=1,
    normalize_symbol="normalize_conv_pre",
    convolution_symbol="execute_conv_pre",
  )
  assert_eq(source.operation(), F32WeightNormalizedConv1d)
  assert_eq(source.numerical_contract(), OrderedF32PerOutputWeightNormConv1dV1)
  assert_eq(source.input_element_count(), 847872L)
  assert_eq(source.weight_element_count(), 14680064L)
  assert_eq(source.output_element_count(), 423936L)
  assert_eq(source.output_time(), 207)
  let text = @utf8.decode(source.source_bytes()) catch {
    _ => abort("invalid UTF-8")
  }
  assert_true(text.contains("sqrtf(sum_squares)"))
  assert_true(text.contains("input_channel < 2048"))
  assert_true(text.contains("kernel_index < 7"))
}

///|
test "weight-normalized Conv1d rejects hostile geometry and symbols" {
  try
    f32_weight_normalized_conv1d_source(
      batch_count=0,
      input_channels=2,
      output_channels=2,
      input_time=3,
      kernel_size=3,
      padding=1,
      stride=1,
      normalize_symbol="n",
      convolution_symbol="c",
    )
  catch {
    InvalidBatchCount => ()
    _ => fail("wrong batch rejection")
  } noraise {
    _ => fail("zero batch admitted")
  }
  try
    f32_weight_normalized_conv1d_source(
      batch_count=1,
      input_channels=2,
      output_channels=2,
      input_time=2,
      kernel_size=7,
      padding=0,
      stride=1,
      normalize_symbol="n",
      convolution_symbol="c",
    )
  catch {
    InvalidTimeCount => ()
    _ => fail("wrong convolution geometry rejection")
  } noraise {
    _ => fail("negative output geometry admitted")
  }
  try
    f32_weight_normalized_conv1d_source(
      batch_count=1,
      input_channels=2147483647,
      output_channels=2147483647,
      input_time=2147483647,
      kernel_size=2147483647,
      padding=0,
      stride=1,
      normalize_symbol="n",
      convolution_symbol="c",
    )
  catch {
    DimensionOverflow => ()
    _ => fail("wrong overflow rejection")
  } noraise {
    _ => fail("overflow admitted")
  }
  try
    f32_weight_normalized_conv1d_source(
      batch_count=1,
      input_channels=2,
      output_channels=2,
      input_time=3,
      kernel_size=3,
      padding=1,
      stride=1,
      normalize_symbol="same",
      convolution_symbol="same",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("wrong duplicate symbol rejection")
  } noraise {
    _ => fail("duplicate symbols admitted")
  }
}
