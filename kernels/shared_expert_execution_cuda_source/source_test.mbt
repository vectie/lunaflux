///|
test "shared gate and product round before adding routed output" {
  let plan = @precision.SharedExpertPrecision::new(
    hidden_width=2048,
    maximum_rows=32,
  )
  let source = @shared_expert_execution_cuda_source.shared_expert_source(
    plan,
    symbol="shared_combine",
  )
  let text = @utf8.decode(source.source_bytes()[:])
  assert_eq(source.launch(), (32, 1, 128))
  assert_eq(source.parameter_count(), 5)
  assert_true(plan.expression(Sigmoid).root() is Convert(BFloat16, _))
  assert_true(plan.expression(Weight).root() is Convert(BFloat16, _))
  assert_true(
    plan.expression(Combine).root()
    is Convert(BFloat16, Binary(AddRN, Input(0), Convert(Float32, Input(1)))),
  )
  assert_true(text.contains("const float *routed"))
  assert_true(text.contains("__shared__ __nv_bfloat16 gate"))
}
