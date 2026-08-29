///|
pub(all) enum LunaAotAdmissionRule {
  Manifest
  Specialization
  ModelIdentity
  DeviceTarget
  CatalogVersion
  ContractScope
  Operation
  Capability
  SemanticVersion
  Profile
  KvLayout
  Alignment
  EntryPoint
  Module
  Workspace
  Artifact
  GraphFallback
} derive(Debug, Eq)

///|
pub(all) suberror LunaAotAdmissionError {
  Mismatch(LunaAotAdmissionRule)
  Missing(LunaAotAdmissionRule)
  Unsupported(LunaAotAdmissionRule)
  Ambiguous(LunaAotAdmissionRule)
} derive(Debug, Eq)

///|
pub struct LunaAotKernelAdmission {
  priv manifest_digest : @luna_capability_manifest.LunaCapabilityManifestDigest
  priv specialization_digest : @luna_specialization.LunaSpecializationDigest
  priv identity : @spec.ModelIdentity
  priv target : @catalog.DeviceTarget
  priv catalog_version : @catalog.CatalogVersion
  priv operation_id : @plan.OperationId
  priv operation_kind : @plan.OperationKind
  priv operation_shape : @plan.OperationShape
  priv capability : @plan.KernelCapabilityId
  priv entry_point : @catalog.AotKernelEntryPoint
  priv module_digest : @catalog.AotArtifactDigest
  priv artifact_module_index : Int
  priv workspace_bytes : Int64
  priv workspace_alignment : Int64
  priv minimum_alignment_bytes : Int64
  priv approved_source : @luna_capability_manifest.LunaApprovedImmutableSourceIdentity
  priv approved_key : @luna_capability_manifest.LunaSignatureKeyIdentityDigest
  priv detached_signature : @luna_capability_manifest.LunaDetachedSignatureDigest
  priv graph : @luna_capability_manifest.LunaGraphCapabilityMetadata
  priv execution : @luna_capability_manifest.LunaGraphExecutionDisposition
} derive(Debug, Eq)

///|
pub fn LunaAotKernelAdmission::manifest_digest(
  self : LunaAotKernelAdmission,
) -> @luna_capability_manifest.LunaCapabilityManifestDigest {
  self.manifest_digest
}

///|
pub fn LunaAotKernelAdmission::specialization_digest(
  self : LunaAotKernelAdmission,
) -> @luna_specialization.LunaSpecializationDigest {
  self.specialization_digest
}

///|
pub fn LunaAotKernelAdmission::identity(
  self : LunaAotKernelAdmission,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn LunaAotKernelAdmission::target(
  self : LunaAotKernelAdmission,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn LunaAotKernelAdmission::catalog_version(
  self : LunaAotKernelAdmission,
) -> @catalog.CatalogVersion {
  self.catalog_version
}

///|
pub fn LunaAotKernelAdmission::operation_id(
  self : LunaAotKernelAdmission,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn LunaAotKernelAdmission::operation_kind(
  self : LunaAotKernelAdmission,
) -> @plan.OperationKind {
  self.operation_kind
}

///|
pub fn LunaAotKernelAdmission::operation_shape(
  self : LunaAotKernelAdmission,
) -> @plan.OperationShape {
  self.operation_shape
}

///|
pub fn LunaAotKernelAdmission::capability(
  self : LunaAotKernelAdmission,
) -> @plan.KernelCapabilityId {
  self.capability
}

///|
pub fn LunaAotKernelAdmission::entry_point(
  self : LunaAotKernelAdmission,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn LunaAotKernelAdmission::module_digest(
  self : LunaAotKernelAdmission,
) -> @catalog.AotArtifactDigest {
  self.module_digest
}

///|
pub fn LunaAotKernelAdmission::artifact_module_index(
  self : LunaAotKernelAdmission,
) -> Int {
  self.artifact_module_index
}

///|
pub fn LunaAotKernelAdmission::workspace_bytes(
  self : LunaAotKernelAdmission,
) -> Int64 {
  self.workspace_bytes
}

///|
pub fn LunaAotKernelAdmission::workspace_alignment(
  self : LunaAotKernelAdmission,
) -> Int64 {
  self.workspace_alignment
}

///|
pub fn LunaAotKernelAdmission::minimum_alignment_bytes(
  self : LunaAotKernelAdmission,
) -> Int64 {
  self.minimum_alignment_bytes
}

///|
pub fn LunaAotKernelAdmission::approved_source_identity(
  self : LunaAotKernelAdmission,
) -> @luna_capability_manifest.LunaApprovedImmutableSourceIdentity {
  self.approved_source
}

///|
pub fn LunaAotKernelAdmission::approved_key_identity(
  self : LunaAotKernelAdmission,
) -> @luna_capability_manifest.LunaSignatureKeyIdentityDigest {
  self.approved_key
}

///|
pub fn LunaAotKernelAdmission::detached_signature_digest(
  self : LunaAotKernelAdmission,
) -> @luna_capability_manifest.LunaDetachedSignatureDigest {
  self.detached_signature
}

///|
pub fn LunaAotKernelAdmission::graph_metadata(
  self : LunaAotKernelAdmission,
) -> @luna_capability_manifest.LunaGraphCapabilityMetadata {
  self.graph
}

///|
pub fn LunaAotKernelAdmission::execution_disposition(
  self : LunaAotKernelAdmission,
) -> @luna_capability_manifest.LunaGraphExecutionDisposition {
  self.execution
}
