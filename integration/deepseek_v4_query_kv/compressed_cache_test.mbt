///|
test "compressed cache bounds every layer including contexts shorter than a group" {
  for
    model in [
      "deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-ai/DeepSeek-V4-Pro",
    ] {
    let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(model)
    for layer in 0..<spec.layer_count() {
      let ratio = spec.compression_ratio(layer).unwrap()
      if ratio == 0 {
        try
          @deepseek_v4_query_kv.compressed_cache_plan(
            spec,
            layer~,
            maximum_rows=5,
            maximum_context=64,
            maximum_state_bytes=16777216L,
            maximum_workspace_bytes=1048576L,
          )
        catch {
          @precision.PrecisionPlanError::NoImplementation => ()
          error => raise error
        } noraise {
          _ => fail("window layer has compressed storage")
        }
      } else {
        let plan = @deepseek_v4_query_kv.compressed_cache_plan(
          spec,
          layer~,
          maximum_rows=5,
          maximum_context=64,
          maximum_state_bytes=16777216L,
          maximum_workspace_bytes=1048576L,
        )
        assert_eq(
          plan.geometry(),
          (spec.head_dimension(), ratio, (5 + ratio - 1) / ratio, 64 / ratio),
        )
        let pool = @deepseek_v4_query_kv.compressor_pool_plan(
          spec,
          layer~,
          maximum_rows=5,
          maximum_context=64,
          maximum_state_bytes=16777216L,
          maximum_workspace_bytes=1048576L,
        )
        let (_, _, _, _, _, positions) = pool.geometry()
        assert_eq(positions, 64)
        let owner = @deepseek_v4_query_kv.DeepSeekCompressedKvCache::new(
          spec,
          layer~,
          maximum_rows=5,
          maximum_context=64,
          maximum_state_bytes=16777216L,
          maximum_workspace_bytes=1048576L,
        )
        let source = @utf8.decode(owner.source_bytes())
        assert_true(source.contains("compressed_history_copy"))
        assert_true(source.contains("live > 64 - base"))
        try owner.values() catch {
          _ => ()
        } noraise {
          _ => fail("unprepared cache")
        }
        try owner.append_descriptors() catch {
          _ => ()
        } noraise {
          _ => fail("unprepared errors")
        }
        owner.close()
        owner.close()
      }
    }
  }
}
