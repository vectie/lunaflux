///|
test "ungated residual performs one BF16-rounded F32 add" {
  let text = @utf8.decode(
    residual_add_bf16_source(elements=16L, function_symbol="residual"),
  )
  assert_true(text.contains("if (i >= 16ULL) return;"))
  assert_true(text.contains("__float2bfloat16_rn(__fadd_rn"))
  assert_false(text.contains("__fmul_rn"))
}
