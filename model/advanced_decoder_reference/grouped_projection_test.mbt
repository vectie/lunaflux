///|
test "grouped output-A reference never crosses group-local slices" {
  let output = try! grouped_output_a_projection_v1(
    [1.0F, 2.0F, 10.0F, 20.0F],
    [1.0F, 0.0F, 0.0F, 1.0F, 2.0F, 0.0F, 0.0F, 3.0F],
    row_count=1,
    group_count=2,
    input_width=2,
    output_width=2,
    limits=try! ReferenceLimits::new(
      max_values=64,
      max_width=8,
      max_iterations=1,
    ),
  )
  assert_eq(output, [1.0F, 2.0F, 20.0F, 60.0F])
}

///|
test "grouped output-A reference rejects transposed or truncated weights" {
  try
    grouped_output_a_projection_v1(
      [1.0F, 2.0F],
      [1.0F],
      row_count=1,
      group_count=1,
      input_width=2,
      output_width=1,
      limits=try! ReferenceLimits::new(
        max_values=8,
        max_width=4,
        max_iterations=1,
      ),
    )
  catch {
    InvalidDimension => ()
    _ => fail("wrong truncated-weight rejection")
  } noraise {
    _ => fail("truncated weights admitted")
  }
}
