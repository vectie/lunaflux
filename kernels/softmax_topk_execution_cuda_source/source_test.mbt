///|
test "router emits full expert probability and selected renormalization stages" {
  let plan = @precision.SoftmaxTopKPrecision::new(
    expert_count=256,
    experts_per_token=8,
    maximum_rows=32,
  )
  let source = @softmax_topk_execution_cuda_source.softmax_topk_source(
    plan,
    symbol="router",
  )
  let text = @utf8.decode(source.source_bytes()[:])
  assert_eq(source.launch(), (32, 1, 32))
  assert_eq(source.parameter_count(), 4)
  assert_true(text.contains("float probabilities[256], top[8]"))
  assert_true(text.contains("probabilities[expert]>probabilities[best]"))
  assert_true(text.contains("selected_total"))
  assert_true(text.contains("__float2bfloat16_rn"))
}

///|
test "existing expert ABI receives exactly widened BF16 route weights" {
  let plan = @precision.SoftmaxTopKPrecision::new(
    expert_count=256,
    experts_per_token=8,
    maximum_rows=32,
    weight_storage=Float32CarryingBf16,
  )
  let source = @softmax_topk_execution_cuda_source.softmax_topk_source(
    plan,
    symbol="widened_router",
  )
  let text = @utf8.decode(source.source_bytes()[:])
  assert_true(text.contains("float *weights"))
  assert_true(
    text.contains("weights[out+rank]=__bfloat162float(__float2bfloat16_rn("),
  )
}
