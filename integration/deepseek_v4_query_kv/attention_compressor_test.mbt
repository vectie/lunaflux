///|
test "attention compressor uses compact group rows and compressed YaRN at all ratios" {
  for
    model in [
      "deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-ai/DeepSeek-V4-Pro",
    ] {
    let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(model)
    for layer in 0..<spec.layer_count() {
      let ratio = spec.compression_ratio(layer).unwrap()
      if ratio == 0 {
        try
          @deepseek_v4_query_kv.compressor_cache_plan(
            spec,
            layer~,
            maximum_input_rows=5,
            maximum_workspace_bytes=1048576L,
          )
        catch {
          @precision.PrecisionPlanError::NoImplementation => ()
          error => raise error
        } noraise {
          _ => fail("window layer silently compressed")
        }
      } else {
        let plan = @deepseek_v4_query_kv.compressor_cache_plan(
          spec,
          layer~,
          maximum_input_rows=5,
          maximum_workspace_bytes=1048576L,
        )
        assert_eq(
          plan.geometry(),
          (
            1,
            spec.head_dimension(),
            spec.rope_head_dimension(),
            (5 + ratio - 1) / ratio,
            spec.max_position_embeddings(),
          ),
        )
        assert_true(plan.frequency() is Yarn(..))
        let owner = @deepseek_v4_query_kv.DeepSeekAttentionCompressor::new(
          spec,
          layer~,
          maximum_rows=5,
          maximum_state_bytes=16777216L,
          maximum_workspace_bytes=1048576L,
        )
        let source = @utf8.decode(owner.source_bytes())
        assert_true(source.contains("compressed_kv_rotary"))
        assert_false(source.contains("is_query"))
        try owner.output() catch {
          _ => ()
        } noraise {
          _ => fail("unprepared output")
        }
        owner.close()
        owner.close()
      }
    }
  }
}
