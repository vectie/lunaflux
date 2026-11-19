///|
test "router retains full softmax and final BF16 boundary" {
  let plan = @precision_ir.SoftmaxTopKPrecision::new(
    expert_count=256,
    experts_per_token=8,
    maximum_rows=32,
  )
  assert_eq(plan.geometry(), (256, 8, 32))
  assert_eq(plan.weight_bytes(), 512L)
  assert_eq(plan.index_bytes(), 1024L)
  assert_true(
    plan.expression(Probability).root() is Binary(DivideRN, Input(0), Input(1)),
  )
  assert_true(
    plan.expression(SelectedWeight).root()
    is Convert(BFloat16, Binary(DivideRN, Input(0), Input(1))),
  )
}

///|
test "F32 route storage widens BF16 values without changing their round" {
  let plan = @precision_ir.SoftmaxTopKPrecision::new(
    expert_count=256,
    experts_per_token=8,
    maximum_rows=32,
    weight_storage=Float32CarryingBf16,
  )
  assert_eq(plan.weight_bytes(), 1024L)
  assert_true(
    plan.expression(SelectedWeight).root()
    is Convert(Float32, Convert(BFloat16, Binary(DivideRN, Input(0), Input(1)))),
  )
}
