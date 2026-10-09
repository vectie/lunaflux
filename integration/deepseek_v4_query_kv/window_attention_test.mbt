///|
test "window adapter selects real compression-zero layers only" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash",
  )
  for layer in [0, 1] {
    let plan = @deepseek_v4_query_kv.window_attention_plan(
      spec,
      layer~,
      maximum_rows=16,
      maximum_state_bytes=1179668L,
    )
    assert_eq(plan.geometry(), (64, 512, 128, 16, 1048576))
    let owner = @deepseek_v4_query_kv.DeepSeekWindowAttention::new(
      spec,
      layer~,
      maximum_rows=16,
      maximum_workspace_bytes=8388608L,
    )
    assert_true(owner.source_bytes().length() > 10000)
    owner.close()
    owner.close()
  }
  for layer in [2, 3, 42] {
    try
      @deepseek_v4_query_kv.window_attention_plan(
        spec,
        layer~,
        maximum_rows=1,
        maximum_state_bytes=1048576L,
      )
    catch {
      @precision.PrecisionPlanError::NoImplementation => ()
      error => raise error
    } noraise {
      _ => fail("compressed layer substituted with window-only attention")
    }
  }
}
