///|
test "selected final modulation keeps BF16 stages before widening" {
  let source = @utf8.decode(
    @affine_modulation_cuda_source.indexed_bf16_shift_scale_f32_source(
      rows=3,
      width=8,
      parameter_rows=2,
      function_symbol="final_rows",
    ),
  )
  assert_true(source.contains("if (i >= 24ULL) return;"))
  assert_true(source.contains("indices[i / 8ULL]"))
  assert_true(source.contains("const __nv_bfloat16 h = __float2bfloat16_rn"))
  assert_true(source.contains("output[i] = __bfloat162float(h);"))
}
