///|
test "export complete checkpoint geometry compressed sublayers" {
  if @env.get_env_var("LUNA_EXPORT_COMPRESSED_SUBLAYER") != Some("1") {
    return
  }
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  for ratio in [4, 128] {
    for layer in 0..<spec.layer_count() {
      if spec.compression_ratio(layer) != Some(ratio) {
        continue
      }
      let owner = @deepseek_v4_query_kv.DeepSeekCompressedSublayer::new(
        spec,
        layer~,
        maximum_rows=3,
        maximum_context=256,
        maximum_state_bytes=16777216L,
        maximum_workspace_bytes=16777216L,
      )
      println("LUNA_COMPRESSED_SUBLAYER_\{ratio}_SOURCE_BEGIN")
      println(@utf8.decode(owner.source_bytes()))
      println("LUNA_COMPRESSED_SUBLAYER_\{ratio}_SOURCE_END")
      owner.close()
      break
    }
  }
}
