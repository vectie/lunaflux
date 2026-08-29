///|
fn target_matches_device_numeric(
  target : @catalog.DeviceTarget,
  device_numeric : @device.DeviceNumericCapability,
) -> Bool {
  target.compute_major() == device_numeric.compute_major() &&
  target.compute_minor() == device_numeric.compute_minor() &&
  target.supports_bf16() == device_numeric.supports_bf16() &&
  target.supports_cublas_lt() == device_numeric.supports_cublas_lt()
}

///|
/// Cross-authenticates all Phase-8 FP8 evidence before startup publication.
///
/// The result remains a pure execution plan. It performs no load or launch and
/// cannot by itself make an instance ready.
pub fn admit(
  plan : @plan.ModelPlan,
  device_numeric : @device.DeviceNumericCapability,
  kernels : @numeric_capability_manifest.AdmittedFp8KernelArtifacts,
  limits : Fp8StartupAdmissionLimits,
) -> Fp8StartupAdmission raise Fp8StartupAdmissionError {
  if device_numeric.version() != @device.DeviceNumericCapabilityVersion::v1() ||
    device_numeric.feature() != FiniteFp8E4M3W8A8F32AccumulateBf16OutputV1 {
    raise Fp8StartupRejected(DeviceEvidence)
  }
  let manifest = kernels.manifest()
  if manifest.version() !=
    @numeric_capability_manifest.Fp8KernelCapabilityManifestVersion::v1() ||
    manifest.model_identity() != plan.identity() ||
    manifest.numeric_schema_digest() != plan.numeric_binding().digest() {
    raise Fp8StartupRejected(ModelPlan)
  }
  if manifest.device_numeric_digest() != device_numeric.digest() ||
    !target_matches_device_numeric(manifest.target(), device_numeric) {
    raise Fp8StartupRejected(DeviceEvidence)
  }
  let artifacts = kernels.artifacts()
  if artifacts.model_identity() != plan.identity() ||
    artifacts.device_target() != manifest.target() ||
    artifacts.catalog_version() != manifest.catalog_version() ||
    manifest.entries().is_empty() {
    raise Fp8StartupRejected(KernelEvidence)
  }
  for entry in manifest.entries() {
    let operation_index = entry.operation().as_int()
    guard plan.operations().get(operation_index) is Some(operation) &&
      plan.numeric_binding().operations().get(operation_index) is Some(numeric) else {
      raise Fp8StartupRejected(EntryBinding)
    }
    if operation.id() != entry.operation() ||
      numeric.operation() != entry.operation() ||
      numeric.execution().digest() != entry.execution_digest() {
      raise Fp8StartupRejected(EntryBinding)
    }
    for weight in entry.weights() {
      let tensor_index = weight.tensor().as_int()
      guard plan.numeric_binding().tensors().get(tensor_index) is Some(tensor) &&
        tensor.reference() == weight.tensor() &&
        tensor.role() == ParameterTensor &&
        tensor.storage().digest() == weight.storage_digest() &&
        tensor.scale() == Some(weight.scale()) else {
        raise Fp8StartupRejected(EntryBinding)
      }
      let scale_index = weight.scale().tensor().as_int()
      guard plan.numeric_binding().tensors().get(scale_index) is Some(scale) &&
        scale.role() == ScaleMetadataTensor &&
        scale.storage().digest() == weight.scale_storage_digest() else {
        raise Fp8StartupRejected(EntryBinding)
      }
    }
    guard artifacts.modules().get(entry.artifact_module_index())
      is Some(artifact_module) &&
      artifact_module.digest() == entry.module_digest() else {
      raise Fp8StartupRejected(KernelEvidence)
    }
  }
  let canonical = encode_fp8_startup_admission(
    plan,
    device_numeric,
    kernels,
    limits.max_canonical_bytes,
  )
  {
    version: Fp8StartupAdmissionVersion::v1(),
    plan,
    device_numeric,
    kernels,
    canonical,
    digest: Fp8StartupAdmissionDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}
