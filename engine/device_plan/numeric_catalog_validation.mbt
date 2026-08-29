///|
fn catalog_is_numeric_v4(version : @catalog.CatalogVersion) -> Bool {
  version == @catalog.CatalogVersion::v4()
}

///|
fn validate_plan_identities(
  model_plan : @plan.ModelPlan,
  weight_layout : @device_materialize.DeviceWeightLayout,
  resolved_catalog : @catalog.ResolvedKernelCatalog,
) -> Unit raise DevicePlanError {
  if weight_layout.identity().content() != model_plan.identity().content() {
    raise IdentityMismatch(WeightContent)
  }
  let version = resolved_catalog.catalog_version()
  if catalog_is_numeric_v4(version) &&
    weight_layout.identity() != model_plan.identity() {
    raise IdentityMismatch(WeightPlan)
  }
  if catalog_is_numeric_v4(version) &&
    weight_layout.numeric_schema_digest() !=
    Some(model_plan.numeric_binding().digest()) {
    raise IdentityMismatch(WeightNumericSchema)
  }
  if catalog_is_numeric_v4(version) {
    if resolved_catalog.model_identity() != model_plan.identity() {
      raise IdentityMismatch(KernelCatalog)
    }
  } else if resolved_catalog.model_identity().content() !=
    model_plan.identity().content() {
    raise IdentityMismatch(KernelCatalog)
  }
}

///|
fn validate_kernel_numeric_binding(
  operation_id : @plan.OperationId,
  execution_digest : @numeric_contract.OperationExecutionDigest,
  catalog_version : @catalog.CatalogVersion,
  binding : @catalog.KernelBinding,
) -> Unit raise DevicePlanError {
  if binding.semantic_version().as_int() != catalog_version.as_int() {
    raise InvalidKernelBinding(operation_id, SemanticVersion)
  }
  if catalog_is_numeric_v4(catalog_version) {
    if binding.operation_execution_digest() != Some(execution_digest) {
      raise InvalidKernelBinding(operation_id, NumericDigest)
    }
    if binding.aot_entry_point() is None {
      raise InvalidKernelBinding(operation_id, AotEntryPoint)
    }
  } else if binding.operation_execution_digest() is Some(_) {
    raise InvalidKernelBinding(operation_id, LegacyNumericDigest)
  }
}
