///|
pub fn I8InertWeightEvidence::tensor(
  self : I8InertWeightEvidence,
) -> @plan.TensorRef {
  self.tensor
}

///|
pub fn I8InertWeightEvidence::storage_digest(
  self : I8InertWeightEvidence,
) -> @numeric_contract.TensorStorageDigest {
  self.storage
}

///|
pub fn I8InertWeightEvidence::shape(
  self : I8InertWeightEvidence,
) -> @numeric_contract.NumericShape {
  self.shape
}

///|
pub fn I8InertWeightEvidence::scale(
  self : I8InertWeightEvidence,
) -> @plan.ScaleTensorRef {
  self.scale
}

///|
pub fn I8InertWeightEvidence::scale_storage_digest(
  self : I8InertWeightEvidence,
) -> @numeric_contract.TensorStorageDigest {
  self.scale_storage
}

///|
pub fn I8InertWeightEvidence::scale_shape(
  self : I8InertWeightEvidence,
) -> @numeric_contract.NumericShape {
  self.scale_shape
}

///|
pub fn I8InertWeightEvidence::output_channel_count(
  self : I8InertWeightEvidence,
) -> Int64 {
  self.output_channel_count
}

///|
pub fn I8InertOperationEvidence::operation(
  self : I8InertOperationEvidence,
) -> @plan.OperationId {
  self.operation
}

///|
pub fn I8InertOperationEvidence::operation_kind(
  self : I8InertOperationEvidence,
) -> @plan.OperationKind {
  self.operation_kind
}

///|
pub fn I8InertOperationEvidence::capability(
  self : I8InertOperationEvidence,
) -> @plan.KernelCapabilityId {
  self.capability
}

///|
pub fn I8InertOperationEvidence::execution_digest(
  self : I8InertOperationEvidence,
) -> @numeric_contract.OperationExecutionDigest {
  self.execution
}

///|
pub fn I8InertOperationEvidence::semantic_version(
  self : I8InertOperationEvidence,
) -> @catalog.KernelSemanticVersion {
  self.semantic_version
}

///|
pub fn I8InertOperationEvidence::entry_point(
  self : I8InertOperationEvidence,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn I8InertOperationEvidence::workspace_bytes(
  self : I8InertOperationEvidence,
) -> Int64 {
  self.workspace_bytes
}

///|
pub fn I8InertOperationEvidence::workspace_alignment(
  self : I8InertOperationEvidence,
) -> Int64 {
  self.workspace_alignment
}

///|
pub fn I8InertOperationEvidence::weights(
  self : I8InertOperationEvidence,
) -> ReadOnlyArray[I8InertWeightEvidence] {
  self.weights
}

///|
pub fn I8InertKernelCapabilityAdmission::version(
  self : I8InertKernelCapabilityAdmission,
) -> I8InertKernelCapabilityAdmissionVersion {
  self.version
}

///|
pub fn I8InertKernelCapabilityAdmission::model_identity(
  self : I8InertKernelCapabilityAdmission,
) -> @spec.ModelIdentity {
  self.model_identity
}

///|
pub fn I8InertKernelCapabilityAdmission::numeric_schema_digest(
  self : I8InertKernelCapabilityAdmission,
) -> @numeric_contract.ModelNumericSchemaDigest {
  self.numeric_schema
}

///|
pub fn I8InertKernelCapabilityAdmission::device_numeric_digest(
  self : I8InertKernelCapabilityAdmission,
) -> @device.DeviceNumericCapabilityDigest {
  self.device_numeric
}

///|
pub fn I8InertKernelCapabilityAdmission::target(
  self : I8InertKernelCapabilityAdmission,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn I8InertKernelCapabilityAdmission::catalog_version(
  self : I8InertKernelCapabilityAdmission,
) -> @catalog.CatalogVersion {
  self.catalog_version
}

///|
pub fn I8InertKernelCapabilityAdmission::operations(
  self : I8InertKernelCapabilityAdmission,
) -> ReadOnlyArray[I8InertOperationEvidence] {
  self.operations
}

///|
pub fn I8InertKernelCapabilityAdmission::canonical_bytes(
  self : I8InertKernelCapabilityAdmission,
) -> Bytes {
  self.canonical
}

///|
pub fn I8InertKernelCapabilityAdmission::digest(
  self : I8InertKernelCapabilityAdmission,
) -> I8InertKernelCapabilityAdmissionDigest {
  self.digest
}
