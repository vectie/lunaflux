///|
pub fn Fp8KernelManifestClaim::profile(
  self : Fp8KernelManifestClaim,
) -> @launch_contract.KernelProfileId {
  self.profile
}

///|
pub fn Fp8KernelManifestClaim::operation(
  self : Fp8KernelManifestClaim,
) -> @plan.OperationId {
  self.operation
}

///|
pub fn Fp8KernelManifestClaim::execution_digest(
  self : Fp8KernelManifestClaim,
) -> @numeric_contract.OperationExecutionDigest {
  self.execution
}

///|
pub fn Fp8KernelManifestClaim::entry_point(
  self : Fp8KernelManifestClaim,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn Fp8KernelWeightBinding::tensor(
  self : Fp8KernelWeightBinding,
) -> @plan.TensorRef {
  self.tensor
}

///|
pub fn Fp8KernelWeightBinding::storage_digest(
  self : Fp8KernelWeightBinding,
) -> @numeric_contract.TensorStorageDigest {
  self.storage
}

///|
pub fn Fp8KernelWeightBinding::scale(
  self : Fp8KernelWeightBinding,
) -> @plan.ScaleTensorRef {
  self.scale
}

///|
pub fn Fp8KernelWeightBinding::scale_storage_digest(
  self : Fp8KernelWeightBinding,
) -> @numeric_contract.TensorStorageDigest {
  self.scale_storage
}

///|
pub fn Fp8KernelManifestEntry::profile(
  self : Fp8KernelManifestEntry,
) -> @launch_contract.KernelProfileId {
  self.profile
}

///|
pub fn Fp8KernelManifestEntry::operation(
  self : Fp8KernelManifestEntry,
) -> @plan.OperationId {
  self.operation
}

///|
pub fn Fp8KernelManifestEntry::execution_digest(
  self : Fp8KernelManifestEntry,
) -> @numeric_contract.OperationExecutionDigest {
  self.execution
}

///|
pub fn Fp8KernelManifestEntry::entry_point(
  self : Fp8KernelManifestEntry,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn Fp8KernelManifestEntry::module_digest(
  self : Fp8KernelManifestEntry,
) -> @catalog.AotArtifactDigest {
  self.module_digest
}

///|
pub fn Fp8KernelManifestEntry::artifact_module_index(
  self : Fp8KernelManifestEntry,
) -> Int {
  self.artifact_module_index
}

///|
pub fn Fp8KernelManifestEntry::weights(
  self : Fp8KernelManifestEntry,
) -> ReadOnlyArray[Fp8KernelWeightBinding] {
  self.weights
}

///|
pub fn Fp8KernelCapabilityManifest::version(
  self : Fp8KernelCapabilityManifest,
) -> Fp8KernelCapabilityManifestVersion {
  self.version
}

///|
pub fn Fp8KernelCapabilityManifest::scope(
  self : Fp8KernelCapabilityManifest,
) -> Fp8KernelExecutionScope {
  self.scope
}

///|
pub fn Fp8KernelCapabilityManifest::model_identity(
  self : Fp8KernelCapabilityManifest,
) -> @spec.ModelIdentity {
  self.model_identity
}

///|
pub fn Fp8KernelCapabilityManifest::numeric_schema_digest(
  self : Fp8KernelCapabilityManifest,
) -> @numeric_contract.ModelNumericSchemaDigest {
  self.numeric_schema
}

///|
pub fn Fp8KernelCapabilityManifest::device_numeric_digest(
  self : Fp8KernelCapabilityManifest,
) -> @device.DeviceNumericCapabilityDigest {
  self.device_numeric
}

///|
pub fn Fp8KernelCapabilityManifest::target(
  self : Fp8KernelCapabilityManifest,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn Fp8KernelCapabilityManifest::catalog_version(
  self : Fp8KernelCapabilityManifest,
) -> @catalog.CatalogVersion {
  self.catalog_version
}

///|
pub fn Fp8KernelCapabilityManifest::entries(
  self : Fp8KernelCapabilityManifest,
) -> ReadOnlyArray[Fp8KernelManifestEntry] {
  self.entries
}

///|
pub fn Fp8KernelCapabilityManifest::canonical_bytes(
  self : Fp8KernelCapabilityManifest,
) -> Bytes {
  self.canonical
}

///|
pub fn Fp8KernelCapabilityManifest::digest(
  self : Fp8KernelCapabilityManifest,
) -> Fp8KernelCapabilityManifestDigest {
  self.digest
}

///|
pub fn AdmittedFp8KernelArtifacts::manifest(
  self : AdmittedFp8KernelArtifacts,
) -> Fp8KernelCapabilityManifest {
  self.manifest
}

///|
pub fn AdmittedFp8KernelArtifacts::artifacts(
  self : AdmittedFp8KernelArtifacts,
) -> @artifact.KernelArtifactBundle {
  self.artifacts
}
