///|
test "learned lowering retains controls and consumes one collapsed stream" {
  let plan = @precision.LearnedTextIoPrecision::new(
    vocabulary_size=17,
    hidden_width=8,
    stream_count=4,
    maximum_rows=5,
    maximum_output_rows=2,
    normalization_epsilon=1.0e-5,
    control_epsilon=1.0e-6,
    control_normalization_epsilon=1.0e-5,
    maximum_workspace_bytes=248L,
  )
  let source = @utf8.decode(
    @text_io_execution_cuda_source.learned_text_io_source(
      plan,
      symbol_prefix="learned_fixture",
    ),
  )
  assert_true(source.contains("void learned_fixture_stream_reduce("))
  assert_true(source.contains("function_weights[weight_base"))
  assert_true(source.contains("controls[stream], value"))
  assert_true(source.contains("stream<1"))
  assert_false(source.contains("stream<4"))
  assert_true(source.contains("= __float2bfloat16_rn(NAN); return;"))
  assert_true(source.contains("tokens[row]=-1"))
  if @env.get_env_var("LUNA_EXPORT_LEARNED_TEXT_IO") == Some("1") {
    println("LUNA_LEARNED_TEXT_SOURCE_BEGIN")
    println(source)
    println("LUNA_LEARNED_TEXT_SOURCE_END")
  }
}
