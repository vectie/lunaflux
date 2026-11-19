///|
test "causal convolution preserves oldest-to-newest channel history" {
  let convolution = @gated_delta_reference.Convolution::new(2, 4)
  let weights = [1.0F, 2.0F, 3.0F, 4.0F, -1.0F, -2.0F, -3.0F, -4.0F]
  let out = FixedArray::make(2, 0.0F)
  convolution.run_into([1.0F, 1.0F], weights, out)
  assert_true((out[0] - 3.928055F).abs() < 0.00001F)
  assert_true((out[1] - -0.071945F).abs() < 0.00001F)
  convolution.run_into([2.0F, 2.0F], weights, out)
  assert_true((out[0] - 10.999816F).abs() < 0.00001F)
  let saved = FixedArray::make(6, 0.0F)
  convolution.copy_into(saved)
  assert_eq(saved[:].to_owned(), [0.0F, 1.0F, 2.0F, 0.0F, 1.0F, 2.0F])
  let continued = @gated_delta_reference.Convolution::new(2, 4)
  continued.restore(saved[:])
  let other = FixedArray::make(2, 0.0F)
  convolution.run_into([3.0F, 3.0F], weights, out)
  continued.run_into([3.0F, 3.0F], weights, other)
  assert_eq(out[0], other[0])
  assert_eq(out[1], other[1])
}

///|
test "unfused BF16 convolution rounds its projection before SiLU" {
  let rounded = @gated_delta_reference.Convolution::new(
    1,
    2,
    projection_boundary=Bf16BeforeSilu,
  )
  let fused = @gated_delta_reference.Convolution::new(1, 2)
  let actual = FixedArray::make(1, 0.0F)
  let other = FixedArray::make(1, 0.0F)
  rounded.run_into([1.0234375F], [0.0F, 1.5F], actual)
  fused.run_into([1.0234375F], [0.0F, 1.5F], other)
  assert_eq(@bf16.round_bf16(actual[0]), 1.2578125F)
  assert_eq(@bf16.round_bf16(other[0]), 1.265625F)
}
