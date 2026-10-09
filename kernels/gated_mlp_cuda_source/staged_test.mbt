///|
test "staged packed MLP writes one product per row and intermediate channel" {
  let source = @utf8.decode(
    staged_packed_swiglu_bf16_source(
      rows=3,
      hidden_size=4,
      intermediate_size=7,
      function_prefix="mlp",
    ),
  )
  assert_true(source.contains("mlp_gate_up("))
  assert_true(source.contains("mlp_down("))
  assert_true(source.contains("if (i >= 21ULL) return;"))
  assert_true(source.contains("if (i >= 12ULL) return;"))
  assert_true(source.contains("product[i] = __float2bfloat16_rn"))
  assert_true(source.contains("weight[(inner + 7ULL) * 4ULL + c]"))
  assert_false(source.contains("down_sum"))
  try
    staged_packed_swiglu_bf16_source(
      rows=2147483647,
      hidden_size=1,
      intermediate_size=2,
      function_prefix="mlp",
    )
  catch {
    DimensionOverflow => ()
    _ => fail("wrong overflow error")
  } noraise {
    _ => fail("unlaunchable intermediate accepted")
  }
}
