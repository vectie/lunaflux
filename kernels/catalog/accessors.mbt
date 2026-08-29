///|
pub fn ResolvedKernelCatalog::model_identity(
  self : ResolvedKernelCatalog,
) -> @spec.ModelIdentity {
  self.model_identity
}

///|
pub fn ResolvedKernelCatalog::device_target(
  self : ResolvedKernelCatalog,
) -> DeviceTarget {
  self.device_target
}

///|
pub fn KernelBinding::operation_id(self : KernelBinding) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn KernelBinding::capability(
  self : KernelBinding,
) -> @plan.KernelCapabilityId {
  self.capability
}

///|
pub fn KernelBinding::semantic_version(
  self : KernelBinding,
) -> KernelSemanticVersion {
  self.semantic_version
}

///|
pub fn KernelBinding::execution_semantics(
  self : KernelBinding,
) -> KernelExecutionSemantics? {
  self.execution_semantics
}

///|
/// Exact numeric operation ABI for catalog v4; legacy bindings return `None`.
pub fn KernelBinding::operation_execution_digest(
  self : KernelBinding,
) -> @numeric_contract.OperationExecutionDigest? {
  self.operation_execution_digest
}

///|
/// Exact content-addressed callable identity for catalog v4.
pub fn KernelBinding::aot_entry_point(
  self : KernelBinding,
) -> AotKernelEntryPoint? {
  self.aot_entry_point
}

///|
pub fn KernelBinding::implementation(
  self : KernelBinding,
) -> KernelImplementationId {
  self.implementation
}

///|
pub fn KernelBinding::workspace_bytes(self : KernelBinding) -> Int64 {
  self.workspace_bytes
}

///|
pub fn KernelBinding::workspace_alignment(self : KernelBinding) -> Int64 {
  self.workspace_alignment
}

///|
pub fn ResolvedKernelCatalog::catalog_version(
  self : ResolvedKernelCatalog,
) -> CatalogVersion {
  self.catalog_version
}

///|
pub fn ResolvedKernelCatalog::bindings(
  self : ResolvedKernelCatalog,
) -> ReadOnlyArray[KernelBinding] {
  self.bindings
}

///|
pub fn ResolvedKernelCatalog::max_workspace_bytes(
  self : ResolvedKernelCatalog,
) -> Int64 {
  self.max_workspace_bytes
}

///|
pub fn ResolvedKernelCatalog::max_workspace_alignment(
  self : ResolvedKernelCatalog,
) -> Int64 {
  self.max_workspace_alignment
}
