///|
test "grouped projection directly emits compact field-major tables" {
  let source = @utf8.decode(
    grouped_parameter_projection_source(
      rows=2,
      input_width=8,
      width=4,
      groups=3,
      fields=6,
      function_symbol="params",
    ),
  )
  assert_true(source.contains("if (i >= 144ULL) return;"))
  assert_true(source.contains("group = column / 24ULL"))
  assert_true(
    source.contains("(field * 6ULL + row * 3ULL + group) * 4ULL + channel"),
  )
  assert_true(source.contains("__fadd_rn(sum, __bfloat162float(bias[column]))"))
  assert_false(source.contains("expf"))
  try
    grouped_parameter_projection_source(
      rows=2147483647,
      input_width=8,
      width=4,
      groups=3,
      fields=6,
      function_symbol="params",
    )
  catch {
    DimensionOverflow => ()
    _ => fail("wrong parameter extent error")
  } noraise {
    _ => fail("unlaunchable parameter output accepted")
  }
}
