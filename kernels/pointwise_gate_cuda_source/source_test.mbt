///|
test "strided BF16 output gate preserves sigmoid round and packed output" {
  let plan = @precision.PointwiseGatePrecision::new(
    vector_width=256,
    vectors_per_row=16,
    maximum_rows=32,
    gate_vector_stride=512,
    gate_row_stride=8192L,
    gate_element_offset=256,
  )
  let source = @pointwise_gate_cuda_source.pointwise_gate_source(
    plan,
    symbol="gate",
  )
  let text = @utf8.decode(source.source_bytes()[:])
  assert_eq(source.launch(), (32, 1, 128))
  assert_eq(source.parameter_count(), 4)
  assert_eq(plan.output_bytes(), 262144L)
  assert_true(text.contains("row*8192ULL"))
  assert_true(text.contains("*512ULL+256ULL+channel%256"))
  assert_true(
    text.contains("const __nv_bfloat16 activated=__float2bfloat16_rn("),
  )
  assert_true(
    @precision.Bf16GateExpression::Product.program().root()
    is Convert(
      BFloat16,
      Binary(MultiplyRN, Convert(Float32, Input(0)), Convert(Float32, Input(1)))
    ),
  )
}
