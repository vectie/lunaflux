///|
/// Admits one exact catalog-v4 symmetric-I8 weight-only capability record.
///
/// All evidence is derived from the already-admitted model, observed device
/// facts, and inert catalog resolution. There are no caller claims and no
/// artifact, module, symbol, launch, device-plan, or execution operation.
pub fn admit_catalog_only_symmetric_i8_weight_only_v1(
  plan : @plan.ModelPlan,
  device_numeric : @device.DeviceNumericCapability,
  resolved_v4 : @catalog.ResolvedKernelCatalog,
  limits : I8InertKernelCapabilityAdmissionLimits,
) -> I8InertKernelCapabilityAdmission raise I8InertKernelCapabilityAdmissionError {
  if plan.kv_execution() != PagedKeyValue {
    raise I8InertKernelCapabilityRejected(ExecutionScope)
  }
  if device_numeric.version() != @device.DeviceNumericCapabilityVersion::v1() ||
    device_numeric.feature() !=
    SymmetricI8WeightOnlyPerOutputChannelF32ScaleF32AccumulateBf16OutputV1 {
    raise I8InertKernelCapabilityRejected(DeviceCapability)
  }
  if resolved_v4.model_identity() != plan.identity() {
    raise I8InertKernelCapabilityRejected(ModelIdentity)
  }
  if !resolved_v4.device_target().matches_numeric_capability(device_numeric) {
    raise I8InertKernelCapabilityRejected(DeviceTarget)
  }
  if resolved_v4.catalog_version() != @catalog.CatalogVersion::v4() {
    raise I8InertKernelCapabilityRejected(CatalogVersion)
  }
  if plan.numeric_binding().operations().length() != plan.operations().length() {
    raise I8InertKernelCapabilityRejected(NumericSchema)
  }
  if resolved_v4.bindings().length() != plan.operations().length() {
    raise I8InertKernelCapabilityRejected(BindingCount)
  }
  let preflight = preflight_i8_admission(plan, resolved_v4, limits)
  validate_i8_scale_uniqueness_for_plan(plan)
  let operations : Array[I8InertOperationEvidence] = []
  let mut weight_count = 0
  for index, operation in plan.operations() {
    let numeric = plan.numeric_binding().operations()[index]
    let execution = numeric.execution()
    let binding = resolved_v4.bindings()[index]
    if exact_i8_weight_only_execution(execution) {
      guard binding.aot_entry_point() is Some(entry_point) else {
        raise I8InertKernelCapabilityRejected(AotBinding)
      }
      let weights = collect_i8_weight_evidence(plan, operation)
      weight_count += weights.length()
      operations.push({
        operation: operation.id(),
        operation_kind: operation.kind(),
        capability: operation.required_capability(),
        execution: execution.digest(),
        semantic_version: binding.semantic_version(),
        entry_point,
        workspace_bytes: binding.workspace_bytes(),
        workspace_alignment: binding.workspace_alignment(),
        weights,
      })
    }
  }
  if operations.length() != preflight.operation_count ||
    weight_count != preflight.weight_count {
    raise I8InertKernelCapabilityRejected(CanonicalBytes)
  }
  let owned_operations = ReadOnlyArray::from_array(operations)
  let canonical = encode_i8_inert_kernel_capability_admission(
    plan.identity(),
    plan.numeric_binding().digest(),
    device_numeric.digest(),
    resolved_v4.device_target(),
    resolved_v4.catalog_version(),
    owned_operations,
    limits.max_canonical_bytes,
    preflight.canonical_bytes,
  )
  {
    version: I8InertKernelCapabilityAdmissionVersion::v1(),
    model_identity: plan.identity(),
    numeric_schema: plan.numeric_binding().digest(),
    device_numeric: device_numeric.digest(),
    target: resolved_v4.device_target(),
    catalog_version: resolved_v4.catalog_version(),
    operations: owned_operations,
    canonical,
    digest: { value: @crypto.bytes_to_hex_string(@crypto.sha256(canonical)) },
  }
}
