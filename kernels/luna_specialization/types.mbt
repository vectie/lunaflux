///|
pub struct LunaSpecializedEntryPointId(Int) derive(Debug, Eq, Hash)

///|
pub fn LunaSpecializedEntryPointId::from_stable_int(
  value : Int,
) -> LunaSpecializedEntryPointId raise LunaSpecializationError {
  if value <= 0 {
    raise Invalid(EntryPoint)
  }
  LunaSpecializedEntryPointId(value)
}

///|
pub fn LunaSpecializedEntryPointId::as_int(
  self : LunaSpecializedEntryPointId,
) -> Int {
  self.0
}

///|
pub(all) enum LunaSpecializedRegionSpace {
  LunaWeightArena
  LunaTokenInput
  LunaActivationWorkspaceArena
} derive(Debug, Eq)

///|
pub struct LunaSpecializedRegion {
  priv space : LunaSpecializedRegionSpace
  priv semantic_id : Int
  priv offset_bytes : Int64
  priv byte_length : Int64
  priv alignment_bytes : Int64
} derive(Debug, Eq)

///|
pub fn LunaSpecializedRegion::space(
  self : LunaSpecializedRegion,
) -> LunaSpecializedRegionSpace {
  self.space
}

///|
pub fn LunaSpecializedRegion::semantic_id(self : LunaSpecializedRegion) -> Int {
  self.semantic_id
}

///|
pub fn LunaSpecializedRegion::offset_bytes(
  self : LunaSpecializedRegion,
) -> Int64 {
  self.offset_bytes
}

///|
pub fn LunaSpecializedRegion::byte_length(
  self : LunaSpecializedRegion,
) -> Int64 {
  self.byte_length
}

///|
pub fn LunaSpecializedRegion::alignment_bytes(
  self : LunaSpecializedRegion,
) -> Int64 {
  self.alignment_bytes
}

///|
pub struct LunaSpecializationDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaSpecializationDigest::from_sha256(
  value : String,
) -> LunaSpecializationDigest raise LunaSpecializationError {
  if !luna_specialization_is_sha256(value) {
    raise Invalid(Digest)
  }
  LunaSpecializationDigest(value)
}

///|
pub fn LunaSpecializationDigest::as_hex(
  self : LunaSpecializationDigest,
) -> String {
  self.0
}

///|
pub struct LunaSpecializationRecord {
  priv identity : @spec.ModelIdentity
  priv target : @catalog.DeviceTarget
  priv catalog_version : @catalog.CatalogVersion
  priv operation_id : @plan.OperationId
  priv operation_kind : @plan.OperationKind
  priv operation_shape : @plan.OperationShape
  priv capability : @plan.KernelCapabilityId
  priv semantic_version : @catalog.KernelSemanticVersion
  priv module_digest : @catalog.AotArtifactDigest
  priv entry_point_id : LunaSpecializedEntryPointId
  priv compiler : LunaCompilerPolicy
  priv numerical : LunaNumericalPolicy
  priv quantization : LunaQuantizationPolicy
  priv profile_shape : @device_profile.ExactExecutionShape
  priv regions : ReadOnlyArray[LunaSpecializedRegion]
  priv workspace_bytes : Int64
  priv workspace_alignment : Int64
  priv rope_theta_bits : UInt64
  priv rms_epsilon_bits : UInt64
  priv program_digest : @luna_tile_ir.LunaTileProgramDigest
  priv cuda_aot_digest : @luna_tile_ir.LunaTileCudaAotDigest
  priv evidence : LunaSpecializationEvidence
  priv canonical : Bytes
  priv digest : LunaSpecializationDigest
} derive(Debug, Eq)

///|
pub fn LunaSpecializationRecord::identity(
  self : LunaSpecializationRecord,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn LunaSpecializationRecord::target(
  self : LunaSpecializationRecord,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn LunaSpecializationRecord::catalog_version(
  self : LunaSpecializationRecord,
) -> @catalog.CatalogVersion {
  self.catalog_version
}

///|
pub fn LunaSpecializationRecord::operation_id(
  self : LunaSpecializationRecord,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn LunaSpecializationRecord::operation_kind(
  self : LunaSpecializationRecord,
) -> @plan.OperationKind {
  self.operation_kind
}

///|
pub fn LunaSpecializationRecord::operation_shape(
  self : LunaSpecializationRecord,
) -> @plan.OperationShape {
  self.operation_shape
}

///|
pub fn LunaSpecializationRecord::capability(
  self : LunaSpecializationRecord,
) -> @plan.KernelCapabilityId {
  self.capability
}

///|
pub fn LunaSpecializationRecord::semantic_version(
  self : LunaSpecializationRecord,
) -> @catalog.KernelSemanticVersion {
  self.semantic_version
}

///|
pub fn LunaSpecializationRecord::module_digest(
  self : LunaSpecializationRecord,
) -> @catalog.AotArtifactDigest {
  self.module_digest
}

///|
pub fn LunaSpecializationRecord::entry_point_id(
  self : LunaSpecializationRecord,
) -> LunaSpecializedEntryPointId {
  self.entry_point_id
}

///|
pub fn LunaSpecializationRecord::workspace_bytes(
  self : LunaSpecializationRecord,
) -> Int64 {
  self.workspace_bytes
}

///|
pub fn LunaSpecializationRecord::workspace_alignment(
  self : LunaSpecializationRecord,
) -> Int64 {
  self.workspace_alignment
}

///|
pub fn LunaSpecializationRecord::program_digest(
  self : LunaSpecializationRecord,
) -> @luna_tile_ir.LunaTileProgramDigest {
  self.program_digest
}

///|
pub fn LunaSpecializationRecord::cuda_aot_digest(
  self : LunaSpecializationRecord,
) -> @luna_tile_ir.LunaTileCudaAotDigest {
  self.cuda_aot_digest
}

///|
pub fn LunaSpecializationRecord::digest(
  self : LunaSpecializationRecord,
) -> LunaSpecializationDigest {
  self.digest
}

///|
pub fn LunaSpecializationRecord::canonical_bytes(
  self : LunaSpecializationRecord,
) -> Bytes {
  self.canonical
}

///|
pub fn LunaSpecializationRecord::regions(
  self : LunaSpecializationRecord,
) -> ReadOnlyArray[LunaSpecializedRegion] {
  self.regions
}

///|
pub fn LunaSpecializationRecord::compiler_policy(
  self : LunaSpecializationRecord,
) -> LunaCompilerPolicy {
  self.compiler
}

///|
pub fn LunaSpecializationRecord::numerical_policy(
  self : LunaSpecializationRecord,
) -> LunaNumericalPolicy {
  self.numerical
}

///|
pub fn LunaSpecializationRecord::quantization_policy(
  self : LunaSpecializationRecord,
) -> LunaQuantizationPolicy {
  self.quantization
}

///|
pub fn LunaSpecializationRecord::profile_shape(
  self : LunaSpecializationRecord,
) -> @device_profile.ExactExecutionShape {
  self.profile_shape
}

///|
pub fn LunaSpecializationRecord::evidence(
  self : LunaSpecializationRecord,
) -> LunaSpecializationEvidence {
  self.evidence
}
