///|
test "all base layers bind the correct output frequency without claiming compressed execution" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let plain = @deepseek_v4_query_kv.attention_output_plan(
    spec,
    layer=0,
    maximum_rows=1,
    maximum_workspace_bytes=1000000L,
  )
  assert_eq(plain.geometry(), (8, 4096, 1024, 1))
  assert_eq(
    plain.rotary().frequency(),
    @precision.RotaryFrequency::Base(theta=10000.0),
  )
  let compressed = @deepseek_v4_query_kv.attention_output_plan(
    spec,
    layer=2,
    maximum_rows=1,
    maximum_workspace_bytes=1000000L,
  )
  assert_true(compressed.rotary().frequency() is Yarn(..))
  let owner = @deepseek_v4_query_kv.DeepSeekWindowSublayer::new(
    spec,
    layer=0,
    maximum_rows=1,
    maximum_workspace_bytes=1000000L,
  )
  assert_true(!owner.source_bytes().is_empty())
  owner.close()
  owner.close()
  try
    @deepseek_v4_query_kv.DeepSeekWindowSublayer::new(
      spec,
      layer=2,
      maximum_rows=1,
      maximum_workspace_bytes=1000000L,
    )
  catch {
    @precision.PrecisionPlanError::NoImplementation => ()
    error => raise error
  } noraise {
    _ => fail("compressed sublayer silently became window-only")
  }
}
