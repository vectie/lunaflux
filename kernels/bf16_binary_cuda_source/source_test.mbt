///|
test "BF16 add source rejects output alias and rounds once" {
  let source = bf16_add_source(
    width=4096,
    maximum_rows=32,
    function_symbol="add_rows",
  )
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("if (output == left || output == right) return"))
  assert_true(text.contains("__fadd_rn"))
  assert_true(text.contains("__float2bfloat16_rn"))
}
