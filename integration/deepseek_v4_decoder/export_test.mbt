///|
test "export all complete base decoder block geometries" {
  if @env.get_env_var("LUNA_EXPORT_DEEPSEEK_BLOCKS") != Some("1") {
    return
  }
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  for ratio in [0, 4, 128] {
    let layer = for layer in 0..<spec.layer_count() {
      if spec.compression_ratio(layer) == Some(ratio) {
        break layer
      }
    } nobreak {
      fail("missing compression profile")
    }
    let owner = @deepseek_v4_decoder.DeepSeekDecoderBlock::new(
      spec,
      layer~,
      maximum_rows=3,
      maximum_context=256,
      maximum_state_bytes=16777216L,
      maximum_workspace_bytes=16777216L,
      maximum_expert_bytes=8589934592L,
      accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
    )
    println("LUNA_BLOCK_\{ratio}_BEGIN")
    println(@utf8.decode(owner.source_bytes()))
    println("LUNA_BLOCK_\{ratio}_END")
    owner.close()
  }
}
