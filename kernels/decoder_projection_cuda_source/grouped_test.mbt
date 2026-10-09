///|
test "grouped BF16 projection fixes layout and ordered arithmetic" {
  let rendered = try! grouped_projection_bf16_source(
    group_count=2,
    input_width=4,
    output_width=3,
    maximum_rows=8,
    function_symbol="grouped_projection_test",
  )
  assert_eq(rendered.group_count(), 2)
  assert_eq(rendered.input_width(), 4)
  assert_eq(rendered.output_width(), 3)
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("blockIdx.y"))
  assert_true(source.contains("__fmul_rn"))
  assert_true(source.contains("__fadd_rn"))
  assert_true(source.contains("__float2bfloat16_rn"))
  assert_false(source.contains("bias"))
}

///|
test "grouped BF16 projection rejects hostile geometry and symbols" {
  try
    grouped_projection_bf16_source(
      group_count=0,
      input_width=4,
      output_width=3,
      maximum_rows=8,
      function_symbol="valid",
    )
  catch {
    InvalidInputWidth => ()
    _ => fail("wrong zero-group rejection")
  } noraise {
    _ => fail("zero groups admitted")
  }
  try
    grouped_projection_bf16_source(
      group_count=2,
      input_width=4,
      output_width=3,
      maximum_rows=8,
      function_symbol="bad-name",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("wrong symbol rejection")
  } noraise {
    _ => fail("bad symbol admitted")
  }
}
