///|
test "bit-copy packing uses existing StepCounts live token field" {
  let plan = @precision.PackedQkvPrecision::new(
    query_width=4096,
    kv_width=512,
    maximum_rows=32,
  )
  assert_eq(plan.geometry(), (4096, 512, 32))
  assert_eq(plan.output_bytes(), 327680L)
  let source = @utf8.decode(
    @packed_qkv_cuda_source.packed_qkv_source(plan, symbol="pack_qkv")[:],
  )
  assert_true(source.contains("const uint16_t *query"))
  assert_true(source.contains("counts[3]>32"))
  assert_true(source.contains("column<5120ULL"))
  assert_true(source.contains("column-4608ULL"))
  assert_true(!source.contains("__float2bfloat16"))
}
