///|
test "all compressed layers map to the exact learned or all-causal read set" {
  for
    model in [
      "deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-ai/DeepSeek-V4-Pro",
    ] {
    let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(model)
    for layer in 0..<spec.layer_count() {
      if spec.compression_ratio(layer) == Some(0) {
        continue
      }
      let plan = @deepseek_v4_query_kv.compressed_attention_plan(
        spec,
        layer~,
        maximum_rows=3,
        maximum_context=16,
        maximum_state_bytes=1048576L,
      )
      assert_eq(plan.window().geometry().2, 16)
      assert_eq(plan.coordinates(), CacheRelative)
      match spec.compression_ratio(layer) {
        Some(4) => {
          assert_true(
            plan.read_set() is @precision.CompressedRowReadSet::Learned(top_k=_),
          )
          assert_eq(plan.cache_capacity(), 4)
        }
        Some(128) => {
          assert_true(
            plan.read_set() is @precision.CompressedRowReadSet::AllCausal,
          )
          assert_eq(plan.cache_capacity(), 0)
        }
        _ => fail("unhandled official compression ratio")
      }
    }
    for layer in [2, 3] {
      let owner = @deepseek_v4_query_kv.DeepSeekCompressedAttention::new(
        spec,
        layer~,
        maximum_rows=3,
        maximum_context=16,
        maximum_state_bytes=16777216L,
        maximum_workspace_bytes=16777216L,
      )
      let text = @utf8.decode(owner.source_bytes())
      assert_true(text.contains("joint_attention_attention"))
      assert_false(text.contains("offsets[row]"))
      if spec.compression_ratio(layer) == Some(4) {
        assert_true(text.contains("index_selection"))
      }
      owner.close()
    }
  }
}
