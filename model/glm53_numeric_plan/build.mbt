///|
fn validate_profile_plan(
  profile : @glm53_spec.Glm53ProfileId,
  plan : @glm_hybrid_execution_plan.GlmHybridExecutionPlan,
) -> Unit raise Glm53NumericPlanError {
  let valid = match profile {
    FullText =>
      plan.profile_identity() == "zai-org/GLM-5.3:glm_moe_dsa" &&
      plan.kda() is None &&
      plan.hyper_connection() is None &&
      plan.vision_conditioning() is None
    FlashMultimodal =>
      plan.profile_identity() == "zai-org/GLM-5.3-Flash-BF16:glm5_next" &&
      plan.kda() is Some(_) &&
      plan.hyper_connection() is Some(_) &&
      plan.vision_conditioning() is Some(_)
  }
  if !valid {
    raise ProfilePlanMismatch
  }
}

///|
/// Binds one exact GLM tensor manifest to its exact hybrid execution plan.
/// Only logical numeric semantics are admitted.
pub fn bind_numeric_plan(
  manifest : @glm53_weights.Glm53TensorManifest,
  execution_plan : @glm_hybrid_execution_plan.GlmHybridExecutionPlan,
) -> Glm53NumericPlan raise Glm53NumericPlanError {
  if manifest.encoding() == Fp8E4M3Block128Dynamic {
    raise UnsupportedFp8BlockScaleLayout(block_rows=128, block_columns=128)
  }
  validate_profile_plan(manifest.profile(), execution_plan)
  if manifest.execution_identity() != execution_plan.identity() {
    raise ManifestIdentityMismatch
  }
  let (tensor_schemas, tensor_bindings, bf16_count, f32_count) = build_tensor_schemas(
    manifest,
  )
  if bf16_count == 0 || f32_count == 0 {
    raise NumericSchemaConstructionFailed
  }
  let (operation_schemas, operation_bindings) = build_operation_schemas(
    execution_plan,
  )
  let numeric_schema = @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors=tensor_schemas,
    operations=operation_schemas,
  ) catch {
    _ => raise NumericSchemaConstructionFailed
  }
  let binding_digest = numeric_binding_digest(
    manifest.profile(),
    manifest.encoding(),
    execution_plan.identity(),
    manifest.layout_digest(),
    numeric_schema,
    tensor_bindings,
    operation_bindings,
  )
  {
    profile: manifest.profile(),
    encoding: manifest.encoding(),
    execution_identity: execution_plan.identity(),
    manifest_layout_digest: manifest.layout_digest(),
    numeric_schema,
    tensor_bindings: ReadOnlyArray::from_array(tensor_bindings),
    operation_bindings: ReadOnlyArray::from_array(operation_bindings),
    binding_digest,
    bfloat16_tensor_count: bf16_count,
    float32_tensor_count: f32_count,
  }
}

///|
/// Builds the converted-BF16 full GLM-5.3 logical numeric plan. The official
/// block-FP8 artifact is a typed failure until its exact block-scale metadata
/// can be represented without pretending it is a per-tensor scale.
pub fn build_full_numeric_plan(
  metadata : @glm53_spec.Glm53FullMetadata,
  encoding : @glm53_weights.Glm53WeightEncoding,
) -> Glm53NumericPlan raise Glm53NumericPlanError {
  if encoding == Fp8E4M3Block128Dynamic {
    raise UnsupportedFp8BlockScaleLayout(block_rows=128, block_columns=128)
  }
  let manifest = @glm53_weights.build_full_tensor_manifest(metadata, encoding) catch {
    _ => raise NumericSchemaConstructionFailed
  }
  bind_numeric_plan(manifest, @glm53.build_full_execution_plan(metadata))
}

///|
/// Builds the exact official Flash-BF16 logical numeric plan.
pub fn build_flash_numeric_plan(
  metadata : @glm53_spec.Glm53FlashMetadata,
  encoding : @glm53_weights.Glm53WeightEncoding,
) -> Glm53NumericPlan raise Glm53NumericPlanError {
  if encoding == Fp8E4M3Block128Dynamic {
    raise UnsupportedFp8BlockScaleLayout(block_rows=128, block_columns=128)
  }
  let manifest = @glm53_weights.build_flash_tensor_manifest(metadata, encoding) catch {
    _ => raise NumericSchemaConstructionFailed
  }
  bind_numeric_plan(manifest, @glm53.build_flash_execution_plan(metadata))
}
