///|
test "grouped parameter layout matches H3 checkpoint matrix and compact tables" {
  let plan = ParameterProjectionLayout::new(
    rows=4,
    input_width=2688,
    width=5376,
    groups=3,
    fields=6,
  )
  assert_eq(plan.input_bytes, 4L * 2688L * 2L)
  assert_eq(plan.weight_bytes, 96768L * 2688L * 2L)
  assert_eq(plan.bias_bytes, 96768L * 2L)
  assert_eq(plan.output_bytes, 6L * 12L * 5376L * 2L)
}
