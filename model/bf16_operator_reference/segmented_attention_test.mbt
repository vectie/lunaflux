///|
test "BF16 normalized probabilities differ from F32 and unnormalized rounding" {
  let plan = @precision.SegmentedAttentionPrecision::new(
    maximum_rows=3,
    maximum_segment=3,
    head_count=1,
    head_width=1,
  )
  let attention = @bf16_operator_reference.SegmentedAttention::new(plan)
  let output = FixedArray::make(3, 0.0F)
  attention.run_into(
    [0, 0, 0],
    [3, 3, 3],
    [0.0F, 0.0F, 0.0F],
    [0.0F, 0.0F, 0.0F],
    [1.0F, 2.0F, 2.0F],
    output,
    rows=3,
  )
  // BF16(1/3)=0.333984375; BF16(5*BF16(1/3))=1.671875.
  // BF16(5/3)=1.6640625, which would be a different numeric program.
  assert_eq(output[:].to_owned(), [1.671875F, 1.671875F, 1.671875F])
}

///|
test "crop attention isolation works at the selected width72" {
  let plan = @precision.SegmentedAttentionPrecision::new(
    maximum_rows=4,
    maximum_segment=2,
    head_count=2,
    head_width=72,
  )
  let attention = @bf16_operator_reference.SegmentedAttention::new(plan)
  let input = Array::make(4 * 2 * 72, 0.0F)
  let values = Array::makei(input.length(), i => {
    [1.0F, 3.0F, 10.0F, 20.0F][i / 144]
  })
  let output = FixedArray::make(input.length(), 0.0F)
  attention.run_into(
    [0, 0, 2, 2],
    [2, 2, 4, 4],
    input[:],
    input[:],
    values[:],
    output,
    rows=4,
  )
  for i in 0..<output.length() {
    assert_eq(output[i], if i < 288 { 2.0F } else { 15.0F })
  }
  assert_true(
    (try? attention.run_into(
      [0, 0, 2, 2],
      [3, 2, 4, 4],
      input[:],
      input[:],
      values[:],
      output,
      rows=4,
    ))
    is Err(_),
  )
}
