///|
test "indexed modulation uses compact parameter rows and explicit BF16 round trips" {
  let source = @utf8.decode(
    indexed_bf16_modulation_source(
      rows=64,
      width=4,
      parameter_rows=6,
      function_prefix="modulate",
    ),
  )
  assert_true(source.contains("modulate_affine("))
  assert_true(source.contains("modulate_residual("))
  assert_true(source.contains("indices[i / 4ULL] * 4ULL + i % 4ULL"))
  assert_true(source.contains("if (i >= 256ULL) return;"))
  assert_true(source.contains("__bfloat162float(shift[p])"))
  assert_true(source.contains("__bfloat162float(gate[p])"))
  assert_false(source.contains("fmaf"))
  for count in [0, -1] {
    try
      indexed_bf16_modulation_source(
        rows=64,
        width=4,
        parameter_rows=count,
        function_prefix="modulate",
      )
    catch {
      InvalidRowCount => ()
      _ => fail("wrong parameter table error")
    } noraise {
      _ => fail("empty parameter table admitted")
    }
  }
}
