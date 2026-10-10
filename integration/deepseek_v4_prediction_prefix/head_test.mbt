///|
test "DSpark head plans all five rows and excludes independent greedy sampling" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let text = @deepseek_v4_prediction_prefix.prediction_text_head_plan(
    spec,
    maximum_workspace_bytes=4194304L,
  )
  assert_eq(text.text().geometry(), (spec.vocabulary_size(), 4096, 1, 5, 5))
  let chain = @deepseek_v4_prediction_prefix.prediction_markov_plan(
    spec,
    temperature=0.7,
    maximum_workspace_bytes=4194304L,
  )
  assert_eq(chain.geometry(), (spec.vocabulary_size(), 4096, 256, 5))
  let head = @deepseek_v4_prediction_prefix.DeepSeekPredictionHead::new(
    spec,
    temperature=0.0,
    maximum_workspace_bytes=4194304L,
  )
  defer head.close()
  let source = @utf8.decode(head.source_bytes())
  assert_true(source.contains("deepseek_prediction_head_stream_reduce"))
  assert_true(source.contains("deepseek_prediction_markov_confidence"))
  try head.launches() catch {
    _ => ()
  } noraise {
    _ => fail("unprepared head executed")
  }
  if @env.get_env_var("LUNA_EXPORT_PREDICTION_HEAD") == Some("1") {
    println("LUNA_HEAD_BEGIN")
    println(source)
    println("LUNA_HEAD_END")
  }
  head.close()
  head.close()
}
