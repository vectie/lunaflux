///|
test "conditioning activation computes in F32 before a single BF16 cast" {
  let source = @utf8.decode(
    f32_silu_to_bf16_source(elements=10752L, function_symbol="activate"),
  )
  assert_true(source.contains("const float *input"))
  assert_true(source.contains("if (i >= 10752ULL) return;"))
  assert_true(source.contains("__float2bfloat16_rn(activated)"))
  assert_false(source.contains("__bfloat162float"))
  for elements in [0L, -1L, 2147483648L] {
    try f32_silu_to_bf16_source(elements~, function_symbol="activate") catch {
      InvalidRowCount => ()
      _ => fail("wrong activation extent error")
    } noraise {
      _ => fail("unlaunchable activation admitted")
    }
  }
}
