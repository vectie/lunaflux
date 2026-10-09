///|
test "learned index queries select actual full-head geometry only on indexed layers" {
  for
    model in [
      "deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-ai/DeepSeek-V4-Pro",
    ] {
    let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(model)
    for layer in 0..<spec.layer_count() {
      if spec.compression_ratio(layer) != Some(4) {
        try
          @deepseek_v4_query_kv.index_queries_plan(
            spec,
            layer~,
            maximum_rows=3,
            maximum_workspace_bytes=1048576L,
          )
        catch {
          @precision.PrecisionPlanError::NoImplementation => ()
          error => raise error
        } noraise {
          _ => fail("non-indexed layer accepted")
        }
        continue
      }
      let plan = @deepseek_v4_query_kv.index_queries_plan(
        spec,
        layer~,
        maximum_rows=3,
        maximum_workspace_bytes=1048576L,
      )
      assert_eq(
        plan.geometry(),
        (
          spec.hidden_size(),
          spec.q_lora_rank(),
          spec.index_head_count(),
          spec.index_head_dimension(),
          3,
        ),
      )
      assert_eq(plan.transform().block_width(), 32)
      let owner = @deepseek_v4_query_kv.DeepSeekIndexQueries::new(
        spec,
        layer~,
        maximum_rows=3,
        maximum_workspace_bytes=1048576L,
      )
      let text = @utf8.decode(owner.source_bytes())
      assert_true(text.contains("_index_queries_queries"))
      assert_true(text.contains("_index_queries_transform_hadamard"))
      owner.close()
      owner.close()
    }
  }
}
