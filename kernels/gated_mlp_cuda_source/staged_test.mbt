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

///|
test "compensated MLP retains BF16 activation and product boundaries" {
  let source = @utf8.decode(
    staged_packed_swiglu_bf16_source(
      rows=1,
      hidden_size=8,
      intermediate_size=3,
      function_prefix="compensated_mlp",
      dot_reduction_law=CompensatedFloat32,
    ),
  )
  for sum in ["up_sum", "gate_sum", "sum"] {
    assert_true(source.contains(sum + "_correction"))
  }
  assert_true(source.contains("const __nv_bfloat16 activated"))
  assert_true(source.contains("const float product_value"))
  if @env.get_env_var("LUNA_EXPORT_COMPENSATED_MLP") is Some("1") {
    println("LUNA_MLP_BEGIN")
    println(source)
    println("LUNA_MLP_END")
  }
}
