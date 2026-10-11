///|
test "compensation is explicit and retains the F32 domain" {
  let (init, update, finish) = @float32_reduction_cuda_source.dot_accumulator_source(
    CompensatedFloat32,
    accumulator="sum",
  )
  assert_true(init.contains("sum_correction"))
  assert_true(update.contains("fabsf(sum) >= fabsf(product)"))
  assert_true(update.contains("isfinite(next)"))
  assert_true(finish.contains("__fadd_rn(sum, sum_correction)"))
  let (_, ordered, close) = @float32_reduction_cuda_source.dot_accumulator_source(
    OrderedFloat32,
    accumulator="sum",
  )
  assert_eq(ordered, "sum = __fadd_rn(sum, product);")
  assert_eq(close, "")
}
