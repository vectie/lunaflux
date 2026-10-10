///|
test "three DSpark blocks share base arithmetic without borrowing base namespaces" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let source = StringBuilder()
  for stage in 0..<3 {
    let block = @deepseek_v4_prediction_prefix.DeepSeekPredictionBlock::new(
      spec,
      stage~,
      maximum_main_rows=8,
      maximum_positions=256,
      maximum_workspace_bytes=16777216L,
      maximum_expert_bytes=8589934592L,
      accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
    )
    defer block.close()
    let code = @utf8.decode(block.source_bytes())
    for
      name in [
        "hyper_attn", "hyper_ffn", "routed", "shared", "router", "output_inverse",
      ] {
      assert_true(code.contains("deepseek_prediction_s\{stage}_" + name))
    }
    assert_false(code.contains("deepseek_l0_"))
    assert_true(block.bank_bytes() > 0L)
    try block.prime_launches() catch {
      @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
      e => raise e
    } noraise {
      _ => fail("unprepared prediction block executed")
    }
    try block.output() catch {
      @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
      e => raise e
    } noraise {
      _ => fail("unprepared prediction output exposed")
    }
    source.write_string(code)
    source.write_string("\n")
    block.close()
    block.close()
  }
  if @env.get_env_var("LUNA_EXPORT_PREDICTION_BLOCK") == Some("1") {
    println("LUNA_BLOCK_BEGIN")
    println(source.to_string())
    println("LUNA_BLOCK_END")
  }
}
