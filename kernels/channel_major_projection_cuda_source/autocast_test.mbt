///|
test "autocast pointwise convolution rounds operands bias and publication but retains F32 storage" {
  let candidate = @channel_major_projection_cuda_source.f32_pointwise_conv1d_source(
    batch_count=1,
    input_channels=24,
    output_channels=24,
    time_count=4,
    function_symbol="probe",
    autocast=Some(Float16),
  )
  assert_eq(
    candidate.numerical_contract(),
    AutocastPointwiseF32AccumulationV1(Float16),
  )
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("const float *input"))
  assert_true(
    source.contains(
      "__half2float(__float2half_rn(weight[weight_base + channel]))",
    ),
  )
  assert_true(source.contains("float sum = 0.0f;"))
  assert_true(
    source.contains("output[flat] = __half2float(__float2half_rn(sum));"),
  )
}
