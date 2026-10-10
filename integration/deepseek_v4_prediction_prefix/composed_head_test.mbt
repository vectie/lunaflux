///|
test "base and predictor text heads compose without private helper collisions" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let base = @base_text.DeepSeekTextIo::new(
    spec,
    maximum_rows=3,
    maximum_output_rows=1,
    maximum_workspace_bytes=16777216L,
  )
  defer base.close()
  let prediction = @deepseek_v4_prediction_prefix.DeepSeekPredictionHead::new(
    spec,
    temperature=0.0,
    maximum_workspace_bytes=16777216L,
  )
  defer prediction.close()
  let code = @utf8.decode(base.source_bytes()) +
    @utf8.decode(prediction.source_bytes())
  for
    name in [
      "deepseek_text_io_stream_reduce", "deepseek_prediction_head_stream_reduce",
    ] {
    assert_eq(code.split("float " + name + "_sigmoid(float value)").length(), 2)
    assert_true(code.contains("__fadd_rn(" + name + "_sigmoid(logit),"))
  }
  assert_false(code.contains("lf_hc_head_sigmoid"))
  if @env.get_env_var("LUNA_EXPORT_COMPOSED_HEADS") == Some("1") {
    println("LUNA_COMPOSED_HEADS_BEGIN")
    println(code)
    println("LUNA_COMPOSED_HEADS_END")
  }
}
