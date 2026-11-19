///|
test "normalization source binds centered gamma and packed projection views" {
  let plan = @precision.RmsNormPrecision::new(
    vector_width=256,
    vectors_per_row=16,
    maximum_rows=32,
    epsilon=0.000001,
    mode=Weighted(OnePlusGamma),
    input_vector_stride=512,
    input_row_stride=8192L,
  )
  let source = @rms_norm_execution_cuda_source.rms_norm_source(
    plan,
    symbol="qnorm",
  )
  assert_eq(source.launch(), (32, 16, 128))
  assert_eq(source.parameter_count(), 4)
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("row*8192ULL"))
  assert_true(text.contains("vector*512ULL"))
  assert_true(text.contains("__fadd_rn"))
  let gated = @rms_norm_execution_cuda_source.rms_norm_source(
    @precision.RmsNormPrecision::new(
      vector_width=128,
      vectors_per_row=32,
      maximum_rows=32,
      epsilon=0.000001,
      mode=Bf16NormBf16WeightSiluGate,
    ),
    symbol="gated_norm",
  )
  assert_eq(gated.parameter_count(), 5)
  assert_true(
    @utf8.decode(gated.source_bytes()).contains("gate[out_base+column]"),
  )
}
