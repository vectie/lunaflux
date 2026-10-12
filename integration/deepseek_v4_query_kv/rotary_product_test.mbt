///|
test "indexer cache never silently discards an explicit attention rotary law" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  try
    @deepseek_v4_query_kv.DeepSeekCompressedKvCache::new(
      spec,
      layer=2,
      for_indexer=true,
      maximum_rows=3,
      maximum_context=256,
      maximum_state_bytes=16777216L,
      maximum_workspace_bytes=16777216L,
      rotary_product=ComplexFloat32Fma,
    )
  catch {
    @precision.PrecisionPlanError::NoImplementation => ()
    _ => fail("unexpected unsupported product error")
  } noraise {
    value => {
      value.close()
      fail("indexer ignored the explicit attention product law")
    }
  }
}
