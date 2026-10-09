///|
test "learned indexer cache connects the separate width rotation and FP4 contract" {
  for
    model in [
      "deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-ai/DeepSeek-V4-Pro",
    ] {
    let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(model)
    for layer in 0..<spec.layer_count() {
      if spec.compression_ratio(layer) != Some(4) {
        try
          @deepseek_v4_query_kv.indexer_cache_transform_plan(
            spec,
            layer~,
            maximum_rows=5,
            maximum_workspace_bytes=1048576L,
          )
        catch {
          @precision.PrecisionPlanError::NoImplementation => ()
          error => raise error
        } noraise {
          _ => fail("non-indexer layer accepted")
        }
        continue
      }
      let plan = @deepseek_v4_query_kv.indexer_cache_transform_plan(
        spec,
        layer~,
        maximum_rows=5,
        maximum_workspace_bytes=1048576L,
      )
      assert_eq(
        plan.rotary().geometry(),
        (
          1,
          spec.index_head_dimension(),
          spec.rope_head_dimension(),
          2,
          spec.max_position_embeddings(),
        ),
      )
      assert_eq(plan.block_width(), 32)
      let cache = @deepseek_v4_query_kv.compressed_cache_plan(
        spec,
        layer~,
        for_indexer=true,
        maximum_rows=5,
        maximum_context=64,
        maximum_state_bytes=16777216L,
        maximum_workspace_bytes=1048576L,
      )
      assert_eq(cache.geometry(), (spec.index_head_dimension(), 4, 2, 16))
      let owner = @deepseek_v4_query_kv.DeepSeekCompressedKvCache::new(
        spec,
        layer~,
        for_indexer=true,
        maximum_rows=5,
        maximum_context=64,
        maximum_state_bytes=16777216L,
        maximum_workspace_bytes=1048576L,
      )
      let text = @utf8.decode(owner.source_bytes())
      assert_true(text.contains("_indexer_kv_hadamard"))
      assert_true(text.contains("_indexer_compressed_history_copy"))
      assert_false(text.contains("__nv_cvt_float_to_fp8"))
      owner.close()
      owner.close()
    }
  }
}
