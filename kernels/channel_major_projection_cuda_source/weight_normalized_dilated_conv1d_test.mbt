///|
test "dilated weight-normalized Conv1d pins same-length ordered F32 geometry" {
  let source = @channel_major_projection_cuda_source.f32_weight_normalized_dilated_conv1d_source(
    batch_count=2,
    channel_count=512,
    time_count=1035,
    kernel_size=11,
    dilation=5,
    padding=25,
    normalize_symbol="normalize_k11_d5",
    convolution_symbol="conv_k11_d5",
  )
  assert_eq(source.element_count(), 1059840L)
  assert_eq(source.weight_elements(), 2883584L)
  assert_true(source.source_bytes().length() > 0)
}

///|
test "dilated Conv1d rejects wrong same-length padding and symbols" {
  try
    @channel_major_projection_cuda_source.f32_weight_normalized_dilated_conv1d_source(
      batch_count=2,
      channel_count=512,
      time_count=1035,
      kernel_size=11,
      dilation=5,
      padding=24,
      normalize_symbol="normalize",
      convolution_symbol="conv",
    )
  catch {
    InvalidTimeCount => ()
    _ => fail("wrong geometry error")
  } noraise {
    _ => fail("wrong padding admitted")
  }
  try
    @channel_major_projection_cuda_source.f32_weight_normalized_dilated_conv1d_source(
      batch_count=2,
      channel_count=512,
      time_count=1035,
      kernel_size=11,
      dilation=5,
      padding=25,
      normalize_symbol="same",
      convolution_symbol="same",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("wrong symbol error")
  } noraise {
    _ => fail("duplicate symbols admitted")
  }
}

///|
test "dilated Conv1d scalar oracle preserves padding and ordered accumulation" {
  let input : Array[Float] = [1.0F, 2.0F, 3.0F]
  let weight_g : Array[Float] = [1.0F]
  let weight_v : Array[Float] = [1.0F, 0.0F, 0.0F]
  let bias : Array[Float] = [0.0F]
  let output = try! @channel_major_projection_cuda_source.f32_weight_normalized_dilated_conv1d_reference(
    batch_count=1,
    channel_count=1,
    time_count=3,
    kernel_size=3,
    dilation=1,
    padding=1,
    input,
    weight_g,
    weight_v,
    bias,
  )
  assert_eq(output, [0.0F, 1.0F, 2.0F])
}
