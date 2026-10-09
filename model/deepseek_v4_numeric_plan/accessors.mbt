///|
pub fn DeepSeekV4NumericPlan::profile(
  self : Self,
) -> @deepseek_v4_spec.DeepSeekV4Profile {
  self.profile
}

///|
pub fn DeepSeekV4NumericPlan::identity(self : Self) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn DeepSeekV4NumericPlan::logical_manifest_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.logical_manifest_digest
}

///|
pub fn DeepSeekV4NumericPlan::numeric_schema_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.numeric_schema_digest
}

///|
pub fn DeepSeekV4NumericPlan::binding_digest(self : Self) -> @spec.PlanDigest {
  self.binding_digest
}

///|
pub fn DeepSeekV4NumericPlan::source_plan_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.identity.plan()
}

///|
pub fn DeepSeekV4NumericPlan::expected_tensor_count(self : Self) -> Int {
  self.expected_tensor_count
}

///|
pub fn DeepSeekV4NumericPlan::tensor_runs(
  self : Self,
) -> ReadOnlyArray[DeepSeekV4NumericTensorRun] {
  self.tensor_runs
}

///|
pub fn DeepSeekV4NumericPlan::operations(
  self : Self,
) -> ReadOnlyArray[DeepSeekV4NumericOperationBinding] {
  self.operations
}

///|
pub fn DeepSeekV4NumericPlan::fp4_codebook(
  self : Self,
) -> DeepSeekV4Fp4CodebookBinding? {
  self.fp4_codebook
}

///|
pub fn DeepSeekV4NumericPlan::physical_readiness(
  self : Self,
) -> DeepSeekV4PhysicalReadiness {
  self.physical_readiness
}

///|
pub fn DeepSeekV4NumericPlan::verify_manifest_tensor_count(
  self : Self,
  actual : Int,
) -> Unit raise DeepSeekV4NumericPlanError {
  if actual != self.expected_tensor_count {
    raise ManifestCardinalityMismatch(
      expected=self.expected_tensor_count,
      actual~,
    )
  }
}

///|
/// Joins an already exact safetensors semantic binding to this logical schema.
/// Header validation remains owned by `deepseek_v4_weights`.
pub fn DeepSeekV4NumericPlan::validate_weight_manifest(
  self : Self,
  binding : @deepseek_v4_weights.DeepSeekV4WeightBinding,
) -> Unit raise DeepSeekV4NumericPlanError {
  if binding.profile() != self.profile || binding.identity() != self.identity {
    raise WeightManifestIdentityMismatch
  }
  self.verify_manifest_tensor_count(binding.tensor_count())
}

///|
/// Typed fail-closed physical boundary. Logical numeric planning never grants
/// a dequantizer or materializer.
pub fn DeepSeekV4NumericPlan::require_physical_materialization(
  self : Self,
) -> Unit raise DeepSeekV4NumericPlanError {
  match self.physical_readiness {
    UnsupportedPhysicalMaterialization(encodings) =>
      raise PhysicalMaterializationUnsupported(encodings)
  }
}
