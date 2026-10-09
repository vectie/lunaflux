///|
test "compressor geometry follows each layer and the separate indexer width" {
  for
    model in [
      "deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-ai/DeepSeek-V4-Pro",
    ] {
    let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(model)
    for layer in 0..<spec.layer_count() {
      let ratio = spec.compression_ratio(layer).unwrap()
      for indexer in [false, true] {
        if ratio == 0 || (indexer && ratio != 4) {
          try
            @deepseek_v4_query_kv.compressor_pool_plan(
              spec,
              layer~,
              for_indexer=indexer,
              maximum_rows=5,
              maximum_state_bytes=16777216L,
              maximum_workspace_bytes=1048576L,
            )
          catch {
            @precision.PrecisionPlanError::NoImplementation => ()
            error => raise error
          } noraise {
            _ => fail("unsupported layer silently received a compressor")
          }
        } else {
          let plan = @deepseek_v4_query_kv.compressor_pool_plan(
            spec,
            layer~,
            for_indexer=indexer,
            maximum_rows=5,
            maximum_state_bytes=16777216L,
            maximum_workspace_bytes=1048576L,
          )
          let width = if indexer {
            spec.index_head_dimension()
          } else {
            spec.head_dimension()
          }
          assert_eq(
            plan.geometry(),
            (
              spec.hidden_size(),
              width,
              ratio,
              ratio == 4,
              5,
              spec.max_position_embeddings(),
            ),
          )
          let owner = @deepseek_v4_query_kv.DeepSeekCompressorPool::new(
            spec,
            layer~,
            for_indexer=indexer,
            maximum_rows=5,
            maximum_state_bytes=16777216L,
            maximum_workspace_bytes=1048576L,
          )
          assert_true(!owner.source_bytes().is_empty())
          try owner.output() catch {
            _ => ()
          } noraise {
            _ => fail("unprepared output escaped")
          }
          owner.close()
          owner.close()
        }
      }
    }
  }
}
