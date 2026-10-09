///|
test "GLM KDA layers map to general recurrent precision while DSA remains separate" {
  let spec = @glm53_spec.Glm53FlashSpec::official()
  let plan = @glm53_packed_expert_upload.recurrent_compute_plan(
    spec,
    layer_index=0,
    maximum_rows=32,
    maximum_sequences=16,
    cache_slots=32,
    maximum_state_bytes=134217728L,
  )
  assert_eq(plan.geometry(), (64, 128, 128))
  assert_eq(plan.log_decay_lower_bound(), -5.0)
  try
    @glm53_packed_expert_upload.recurrent_compute_plan(
      spec,
      layer_index=3,
      maximum_rows=32,
      maximum_sequences=16,
      cache_slots=32,
      maximum_state_bytes=134217728L,
    )
  catch {
    @precision_ir.PrecisionPlanError::IllegalPlacement => ()
    e => raise e
  } noraise {
    _ => fail("DSA silently replaced with KDA")
  }
}
