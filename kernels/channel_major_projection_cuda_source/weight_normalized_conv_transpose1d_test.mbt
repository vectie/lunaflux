///|
test "exact first BigVGAN upsampler geometry is rendered" {
  let source = try! f32_weight_normalized_conv_transpose1d_source(
    batch_count=2,
    input_channels=1024,
    output_channels=512,
    input_time=207,
    kernel_size=9,
    padding=2,
    stride=5,
    normalize_symbol="normalize_first_upsample",
    convolution_symbol="execute_first_upsample",
  )
  assert_eq(source.operation(), F32WeightNormalizedConvTranspose1d)
  assert_eq(
    source.numerical_contract(),
    OrderedF32PerInputWeightNormConvTranspose1dV1,
  )
  assert_eq(source.output_time(), 1035)
  assert_eq(source.input_element_count(), 423936L)
  assert_eq(source.weight_element_count(), 4718592L)
  assert_eq(source.output_element_count(), 1059840L)
}

///|
test "transposed convolution source rejects geometry overflow and hostile symbols" {
  for
    hostile in [
      (0, 1024, 512, 207, 9, 2, 5),
      (2, 1024, 512, 1, 1, 1, 1),
      (2, 2147483647, 2147483647, 207, 9, 2, 5),
    ] {
    let (batch, input, output, time, kernel, padding, stride) = hostile
    try
      f32_weight_normalized_conv_transpose1d_source(
        batch_count=batch,
        input_channels=input,
        output_channels=output,
        input_time=time,
        kernel_size=kernel,
        padding~,
        stride~,
        normalize_symbol="normalize_first_upsample",
        convolution_symbol="execute_first_upsample",
      )
    catch {
      InvalidBatchCount | InvalidTimeCount | DimensionOverflow => ()
      _ => fail("wrong hostile geometry rejection")
    } noraise {
      _ => fail("hostile geometry admitted")
    }
  }
  try
    f32_weight_normalized_conv_transpose1d_source(
      batch_count=2,
      input_channels=1024,
      output_channels=512,
      input_time=207,
      kernel_size=9,
      padding=2,
      stride=5,
      normalize_symbol="9bad",
      convolution_symbol="execute_first_upsample",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("wrong hostile symbol rejection")
  } noraise {
    _ => fail("leading-digit symbol admitted")
  }
}
