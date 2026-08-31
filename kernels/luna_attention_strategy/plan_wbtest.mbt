///|
test "attention planning separates query-tiled prefill and split decode" {
  let shape = AttentionShape::new(
    query_heads=16,
    kv_heads=4,
    head_dimension=128,
    tokens_per_page=16,
  )
  let plan = plan_attention_dispatch(shape, 8192, 256, 8)
  assert_eq(plan.prefill().phase(), Prefill)
  assert_eq(plan.prefill().traversal(), QueryTile)
  assert_eq(plan.prefill().query_tile(), 16)
  assert_eq(plan.decode().phase(), Decode)
  assert_eq(plan.decode().traversal(), KeyValueSplit)
  assert_eq(plan.decode().reduction(), SplitOnlineSoftmaxMerge)
  assert_eq(plan.decode().key_value_splits(), 8)
}
