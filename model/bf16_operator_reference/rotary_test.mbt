///|
test "multi-axis rotary selects per-frequency coordinates and rotates split prefix halves" {
  let plan = @precision.MultiAxisRotaryPrecision::new(
    head_count=1,
    head_width=6,
    rotary_width=4,
    maximum_rows=1,
    maximum_positions=10,
    axis_for_pair=[0, 1],
    inverse_frequencies=[1.0F, 1.0F],
  )
  let cosine = FixedArray::make(2, 0.0F)
  let sine = FixedArray::make(2, 0.0F)
  @bf16_operator_reference.rotary_coefficients_into(
    plan,
    [0, 1, 9],
    cosine,
    sine,
    rows=1,
  )
  assert_eq(cosine[:].to_owned(), [1.0F, 0.5390625F])
  assert_eq(sine[:].to_owned(), [0.0F, 0.83984375F])
  let output = FixedArray::make(6, 0.0F)
  @bf16_operator_reference.multi_axis_rotary_into(
    plan,
    [1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F],
    cosine[:],
    sine[:],
    output,
    rows=1,
  )
  assert_eq(output[:].to_owned(), [1.0F, -2.28125F, 3.0F, 3.84375F, 5.0F, 6.0F])
  // In-place use must read both halves before publishing either.
  @bf16_operator_reference.multi_axis_rotary_into(
    plan,
    output[:],
    [1.0F, 1.0F],
    [0.0F, 0.0F],
    output,
    rows=1,
  )
  assert_eq(output[:].to_owned(), [1.0F, -2.28125F, 3.0F, 3.84375F, 5.0F, 6.0F])
}

///|
test "BF16 rotary products round before addition rather than only at final output" {
  let plan = @precision.MultiAxisRotaryPrecision::new(
    head_count=1,
    head_width=2,
    rotary_width=2,
    maximum_rows=1,
    maximum_positions=10,
    axis_for_pair=[0],
    inverse_frequencies=[1.0F],
  )
  let output = FixedArray::make(2, 0.0F)
  @bf16_operator_reference.multi_axis_rotary_into(
    plan,
    [1.0F, 1.0078125F],
    [0.5390625F],
    [0.83984375F],
    output,
    rows=1,
  )
  // Omitting the product rounds would give -0.306640625 for the real part.
  assert_eq(output[:].to_owned(), [-0.30859375F, 1.3828125F])
}
