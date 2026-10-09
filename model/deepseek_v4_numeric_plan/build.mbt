///|
fn validate_plan_profile(
  semantic : @deepseek_v4.DeepSeekV4SemanticPlan,
) -> Unit raise DeepSeekV4NumericPlanError {
  let execution = semantic.execution_plan()
  let spec = semantic.metadata().spec()
  if execution.schema_version() != 1 ||
    execution.profile_identity() != spec.execution_profile_identity() ||
    execution.content_digest() != semantic.content_digest() ||
    execution.shape().decoder_layer_count() != spec.layer_count() ||
    execution.layers().length() != spec.layer_count() {
    raise ProfilePlanMismatch
  }
}

///|
/// Builds the exact bounded logical numeric schema for one of the five closed
/// DeepSeek-V4 profiles. It does not parse or authorize physical bytes.
pub fn build(
  semantic : @deepseek_v4.DeepSeekV4SemanticPlan,
) -> DeepSeekV4NumericPlan raise DeepSeekV4NumericPlanError {
  validate_plan_profile(semantic)
  let execution = semantic.execution_plan()
  let spec = semantic.metadata().spec()
  let identity = try! @spec.ModelIdentity::new(
    execution.content_digest(),
    execution.plan_digest(),
  )
  let (tensor_runs, expected_tensor_count) = build_tensor_runs(spec)
  let operations = build_operations(execution, spec)
  let fp4_codebook = if spec.expert_storage() is Fp4E2M1 {
    Some(CanonicalE2M1)
  } else {
    None
  }
  let logical_manifest_digest = logical_manifest_digest(
    spec.profile(),
    identity,
    expected_tensor_count,
    tensor_runs,
    fp4_codebook,
  )
  let numeric_schema_digest = numeric_schema_digest(
    identity, logical_manifest_digest, operations,
  )
  let binding_digest = numeric_binding_digest(
    identity, logical_manifest_digest, numeric_schema_digest,
  )
  let physical_readiness = match
    @deepseek_v4_weights.materialization_readiness(spec) {
    UnsupportedByteLayouts(encodings) =>
      UnsupportedPhysicalMaterialization(encodings)
  }
  {
    profile: spec.profile(),
    identity,
    logical_manifest_digest,
    numeric_schema_digest,
    binding_digest,
    expected_tensor_count,
    tensor_runs: ReadOnlyArray::from_array(tensor_runs),
    operations: ReadOnlyArray::from_array(operations),
    fp4_codebook,
    physical_readiness,
  }
}
