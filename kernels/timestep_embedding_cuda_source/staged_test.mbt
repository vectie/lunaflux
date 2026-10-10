///|
test "staged timestep MLP computes first projection once per hidden element" {
  let text = @utf8.decode(
    staged_dense_silu_dense_f32_rows_source(
      rows=3,
      input_width=256,
      hidden_width=5376,
      output_width=2688,
      function_prefix="time",
    ),
  )
  assert_true(text.contains("void time_hidden("))
  assert_true(text.contains("void time_output("))
  assert_true(text.contains("if (i >= 16128ULL) return;"))
  assert_true(text.contains("float first = bias[hidden];"))
  assert_false(text.contains("for (int hidden"))
  assert_false(text.contains("}#include"))
  assert_true(text.has_suffix("\n"))
}
