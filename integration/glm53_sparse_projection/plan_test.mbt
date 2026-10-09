///|
test "Flash DSA maps into general rope-free projection geometry" {
  let spec = @spec.Glm53FlashSpec::official()
  let plan = @glm53_sparse_projection.flash_sparse_projection_plan(
    spec,
    layer_index=3,
    maximum_queries=1,
    maximum_history=4096,
    index_layer_norm_epsilon=0.00001,
    maximum_attention_workspace_bytes=10000000L,
    maximum_projection_workspace_bytes=181056L,
  )
  assert_eq(plan.geometry(), (4096, 1536, 512))
  assert_eq(plan.workspace_bytes(), 181056L)
  assert_eq(
    plan.weight_bytes().fold(init=0L, (sum, size) => sum + size),
    115610112L,
  )
  assert_eq(plan.attention().attention_geometry(), (64, 64, 256, 256))
  try
    @glm53_sparse_projection.flash_sparse_projection_plan(
      spec,
      layer_index=0,
      maximum_queries=1,
      maximum_history=4096,
      index_layer_norm_epsilon=0.00001,
      maximum_attention_workspace_bytes=10000000L,
      maximum_projection_workspace_bytes=181056L,
    )
  catch {
    @precision.PrecisionPlanError::IllegalPlacement => ()
    _ => fail("wrong layer error")
  } noraise {
    _ => fail("KDA replaced with DSA")
  }
}
