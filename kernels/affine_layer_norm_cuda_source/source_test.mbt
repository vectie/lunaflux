///|
test "affine LayerNorm centers inputs and preserves explicit bias" {
  let source = @affine_layer_norm_cuda_source.affine_layer_norm_bf16_source(
    width=4,
    maximum_rows=8,
    epsilon=0.00001,
    function_symbol="index_layer_norm",
  )
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("const float mean = __fdiv_rn(sum, 4.0f)"))
  assert_true(
    text.contains("__fsub_rn(__bfloat162float(input[base + c]), mean)"),
  )
  assert_true(text.contains("__bfloat162float(bias[c])"))
  assert_eq(source.launch(), (8, 1))
}
