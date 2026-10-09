///|
/// Projects the authenticated full GLM-5.3 decoder into the family-neutral
/// workload envelope used by lifecycle code. The model-owned execution plan
/// remains the sole source of topology and identity.
pub fn build_full_workload_execution_plan(
  metadata : @glm53_spec.Glm53FullMetadata,
) -> @workload_execution_plan.WorkloadExecutionPlan {
  let decoder = build_full_execution_plan(metadata)
  try! @workload_execution_plan.WorkloadExecutionPlan::decoder_only(
    identity=decoder.identity(),
    schema_version=decoder.schema_version(),
    decoder_layer_count=decoder.decoder_layer_count(),
    maximum_token_context=decoder.context_length(),
  )
}

///|
/// Projects authenticated GLM-5.3-Flash into the family-neutral multimodal
/// workload envelope without leaking GLM-specific KDA, DSA, mHC, or vision
/// concrete types.
pub fn build_flash_workload_execution_plan(
  metadata : @glm53_spec.Glm53FlashMetadata,
) -> @workload_execution_plan.WorkloadExecutionPlan {
  let decoder = build_flash_execution_plan(metadata)
  try! @workload_execution_plan.WorkloadExecutionPlan::multimodal_conditional(
    identity=decoder.identity(),
    schema_version=decoder.schema_version(),
    decoder_layer_count=decoder.decoder_layer_count(),
    maximum_token_context=decoder.context_length(),
  )
}
