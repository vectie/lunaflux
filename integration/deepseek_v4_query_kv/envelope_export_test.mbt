///|
test "export all complete attention envelope geometries" {
  if @env.get_env_var("LUNA_EXPORT_ATTENTION_ENVELOPES") != Some("1") {
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
    let owner = @deepseek_v4_query_kv.DeepSeekAttentionEnvelope::new(
      spec,
      layer~,
      maximum_rows=3,
      maximum_context=256,
      maximum_state_bytes=16777216L,
      maximum_workspace_bytes=16777216L,
    )
    println("LUNA_ENVELOPE_\{ratio}_BEGIN")
    println(@utf8.decode(owner.source_bytes()))
    println("LUNA_ENVELOPE_\{ratio}_END")
    owner.close()
  }
}
