///|
test "text lowering preserves mean and normalized BF16 boundaries" {
  let plan = @precision.TextIoPrecision::new(
    vocabulary_size=17,
    hidden_width=8,
    stream_count=4,
    maximum_rows=32,
    maximum_output_rows=1,
    normalization_epsilon=1.0e-5,
    maximum_workspace_bytes=84L,
  )
  let source = @text_io_execution_cuda_source.text_io_source(
    plan,
    symbol_prefix="text_a",
  )
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("ids[i/(4ULL*8ULL)]"))
  assert_true(
    text.contains(
      "const float mean=__bfloat162float(__float2bfloat16_rn(__fdiv_rn(sum,4.0f)))",
    ),
  )
  assert_true(
    text.contains(
      "const float normalized=__bfloat162float(__float2bfloat16_rn(__fmul_rn(mean,inverse)))",
    ),
  )
  assert_true(
    text.contains("__fmul_rn(normalized,__bfloat162float(weight[c]))"),
  )
  assert_true(text.contains("const int input_row = selected[row]"))
  assert_true(
    text.contains("logits[i]=__bfloat162float(__float2bfloat16_rn(sum))"),
  )
  assert_true(text.contains("if(token<0 || value>best)"))
  assert_true(text.contains("tokens[row]=-1"))
  assert_false(text.contains("hc_head_fn"))
  for stage in 0..<4 {
    assert_true(text.contains("void " + source.symbol(stage) + "("))
  }
  let other = @text_io_execution_cuda_source.text_io_source(
    plan,
    symbol_prefix="text_b",
  )
  assert_false(@utf8.decode(other.source_bytes()).contains("text_a"))
}
