///|
test "whole prediction source joins all three stages and distinct main/draft ports" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let owner = @deepseek_v4_prediction_prefix.DeepSeekPredictor::new(
    spec,
    maximum_main_rows=8,
    maximum_positions=256,
    maximum_workspace_bytes=16777216L,
    maximum_expert_bytes=8589934592L,
    temperature=0.7,
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
  )
  defer owner.close()
  let code = @utf8.decode(owner.source_bytes())
  for
    name in [
      "deepseek_prediction_main", "deepseek_prediction_seed_seed", "deepseek_prediction_embedding_embed",
      "deepseek_prediction_head_stream_reduce", "deepseek_prediction_markov_confidence",
    ] {
    assert_true(code.contains(name))
  }
  for stage in 0..<3 {
    assert_true(code.contains("deepseek_prediction_s\{stage}_hyper_ffn"))
  }
  assert_true(code.contains("ids[row]=row==0?token:128799"))
  assert_true(code.contains("positions[row]=valid?start[0]+n+row:0"))
  assert_false(code.contains("deepseek_l0_"))
  try owner.prime_launches() catch {
    _ => ()
  } noraise {
    _ => fail("unprepared prime exposed")
  }
  try owner.prediction_launches() catch {
    _ => ()
  } noraise {
    _ => fail("unprepared prediction exposed")
  }
  if @env.get_env_var("LUNA_EXPORT_PREDICTOR") == Some("1") {
    println("LUNA_PREDICTOR_BEGIN")
    println(code)
    println("LUNA_PREDICTOR_END")
  }
  owner.close()
  owner.close()
}
