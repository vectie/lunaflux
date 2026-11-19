///|
test "full attention gate reads per-head second halves without using Q values" {
  let plan = @precision.PointwiseGatePrecision::new(
    vector_width=2,
    vectors_per_row=2,
    maximum_rows=2,
    gate_vector_stride=4,
    gate_row_stride=8L,
    gate_element_offset=2,
  )
  let input = [2.0F, 4.0F, 8.0F, 16.0F, 2.0F, 4.0F, 8.0F, 16.0F]
  let raw_projection = [
    999.0F,
    999.0F,
    0.0F,
    0.0F,
    999.0F,
    999.0F,
    100.0F,
    -100.0F,
    999.0F,
    999.0F,
    1.0F,
    1.0F,
    999.0F,
    999.0F,
    0.0F,
    0.0F,
  ]
  let output = FixedArray::make(8, 0.0F)
  @bf16_operator_reference.pointwise_gate_into(
    plan,
    input,
    raw_projection,
    output,
    rows=2,
  )
  assert_eq(output[:].to_owned(), [
    1.0F, 2.0F, 8.0F, 0.0F, 1.4609375F, 2.921875F, 4.0F, 8.0F,
  ])
}
