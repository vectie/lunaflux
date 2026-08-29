///|
fn luna_capability_is_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}

///|
pub struct LunaCapabilityManifestVersion(Int) derive(Debug, Eq)

///|
pub fn LunaCapabilityManifestVersion::v1() -> LunaCapabilityManifestVersion {
  LunaCapabilityManifestVersion(1)
}

///|
pub fn LunaCapabilityManifestVersion::as_int(
  self : LunaCapabilityManifestVersion,
) -> Int {
  self.0
}

///|
pub struct LunaGpuArchitectureRange {
  priv minimum_major : Int
  priv minimum_minor : Int
  priv maximum_major : Int
  priv maximum_minor : Int
} derive(Debug, Eq)

///|
pub fn LunaGpuArchitectureRange::new(
  minimum_major~ : Int,
  minimum_minor~ : Int,
  maximum_major~ : Int,
  maximum_minor~ : Int,
) -> LunaGpuArchitectureRange raise LunaCapabilityManifestError {
  if minimum_major <= 0 ||
    minimum_major > 99 ||
    minimum_minor < 0 ||
    maximum_major <= 0 ||
    maximum_major > 99 ||
    maximum_minor < 0 ||
    minimum_minor > 99 ||
    maximum_minor > 99 ||
    minimum_major * 100 + minimum_minor > maximum_major * 100 + maximum_minor {
    raise Invalid(Architecture)
  }
  { minimum_major, minimum_minor, maximum_major, maximum_minor }
}

///|
pub fn LunaGpuArchitectureRange::contains(
  self : LunaGpuArchitectureRange,
  target : @catalog.DeviceTarget,
) -> Bool {
  let actual = target.compute_major() * 100 + target.compute_minor()
  actual >= self.minimum_major * 100 + self.minimum_minor &&
  actual <= self.maximum_major * 100 + self.maximum_minor
}

///|
pub(all) enum LunaCapabilityDType {
  LunaBFloat16
  LunaFloat16
  LunaFloat32
} derive(Debug, Eq)

///|
pub(all) enum LunaAccumulatorDType {
  LunaAccumulatorFloat32
} derive(Debug, Eq)

///|
pub(all) enum LunaTensorLayoutContract {
  LunaRowMajorStridedV1
} derive(Debug, Eq)

///|
pub struct LunaKvLayoutContract {
  priv stable_version : Int
  priv tokens_per_page : Int
} derive(Debug, Eq)

///|
pub fn LunaKvLayoutContract::new(
  stable_version~ : Int,
  tokens_per_page~ : Int,
) -> LunaKvLayoutContract raise LunaCapabilityManifestError {
  if stable_version <= 0 || tokens_per_page <= 0 {
    raise Invalid(Layout)
  }
  { stable_version, tokens_per_page }
}

///|
pub fn LunaKvLayoutContract::stable_version(self : LunaKvLayoutContract) -> Int {
  self.stable_version
}

///|
pub fn LunaKvLayoutContract::tokens_per_page(
  self : LunaKvLayoutContract,
) -> Int {
  self.tokens_per_page
}

///|
pub struct LunaCapabilityManifestDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaCapabilityManifestDigest::from_sha256(
  value : String,
) -> LunaCapabilityManifestDigest raise LunaCapabilityManifestError {
  if !luna_capability_is_sha256(value) {
    raise Invalid(Digest)
  }
  LunaCapabilityManifestDigest(value)
}

///|
pub fn LunaCapabilityManifestDigest::as_hex(
  self : LunaCapabilityManifestDigest,
) -> String {
  self.0
}

///|
pub struct LunaCapabilityManifestEntry {
  priv version : LunaCapabilityManifestVersion
  priv specialization : @luna_specialization.LunaSpecializationRecord
  priv architecture : LunaGpuArchitectureRange
  priv input_dtype : LunaCapabilityDType
  priv accumulator_dtype : LunaAccumulatorDType
  priv tensor_layout : LunaTensorLayoutContract
  priv kv_layout : LunaKvLayoutContract
  priv minimum_alignment_bytes : Int64
  priv graph : LunaGraphCapabilityMetadata
  priv canonical : Bytes
  priv digest : LunaCapabilityManifestDigest
} derive(Debug, Eq)

///|
pub fn LunaCapabilityManifestEntry::version(
  self : LunaCapabilityManifestEntry,
) -> LunaCapabilityManifestVersion {
  self.version
}

///|
pub fn LunaCapabilityManifestEntry::signing_bytes(
  self : LunaCapabilityManifestEntry,
) -> Bytes {
  self.canonical
}

///|
pub fn LunaCapabilityManifestEntry::digest(
  self : LunaCapabilityManifestEntry,
) -> LunaCapabilityManifestDigest {
  self.digest
}

///|
pub fn LunaCapabilityManifestEntry::specialization_digest(
  self : LunaCapabilityManifestEntry,
) -> @luna_specialization.LunaSpecializationDigest {
  self.specialization.digest()
}

///|
pub struct LunaAdmittedCapabilityManifestEntry {
  priv entry : LunaCapabilityManifestEntry
  priv approved_source : LunaApprovedImmutableSourceIdentity
  priv key_identity : LunaSignatureKeyIdentityDigest
  priv signature_digest : LunaDetachedSignatureDigest
} derive(Debug, Eq)

///|
pub fn LunaAdmittedCapabilityManifestEntry::manifest_digest(
  self : LunaAdmittedCapabilityManifestEntry,
) -> LunaCapabilityManifestDigest {
  self.entry.digest
}

///|
pub fn LunaAdmittedCapabilityManifestEntry::specialization_digest(
  self : LunaAdmittedCapabilityManifestEntry,
) -> @luna_specialization.LunaSpecializationDigest {
  self.entry.specialization.digest()
}

///|
pub fn LunaAdmittedCapabilityManifestEntry::model_identity(
  self : LunaAdmittedCapabilityManifestEntry,
) -> @spec.ModelIdentity {
  self.entry.specialization.identity()
}

///|
pub fn LunaAdmittedCapabilityManifestEntry::target(
  self : LunaAdmittedCapabilityManifestEntry,
) -> @catalog.DeviceTarget {
  self.entry.specialization.target()
}

///|
pub fn LunaAdmittedCapabilityManifestEntry::operation_id(
  self : LunaAdmittedCapabilityManifestEntry,
) -> @plan.OperationId {
  self.entry.specialization.operation_id()
}

///|
pub fn LunaAdmittedCapabilityManifestEntry::module_digest(
  self : LunaAdmittedCapabilityManifestEntry,
) -> @catalog.AotArtifactDigest {
  self.entry.specialization.module_digest()
}

///|
pub fn LunaAdmittedCapabilityManifestEntry::graph_metadata(
  self : LunaAdmittedCapabilityManifestEntry,
) -> LunaGraphCapabilityMetadata {
  self.entry.graph
}

///|
pub fn LunaAdmittedCapabilityManifestEntry::catalog_version(
  self : LunaAdmittedCapabilityManifestEntry,
) -> @catalog.CatalogVersion {
  self.entry.specialization.catalog_version()
}

///|
pub fn LunaAdmittedCapabilityManifestEntry::capability(
  self : LunaAdmittedCapabilityManifestEntry,
) -> @plan.KernelCapabilityId {
  self.entry.specialization.capability()
}

///|
pub fn LunaAdmittedCapabilityManifestEntry::semantic_version(
  self : LunaAdmittedCapabilityManifestEntry,
) -> @catalog.KernelSemanticVersion {
  self.entry.specialization.semantic_version()
}

///|
pub fn LunaAdmittedCapabilityManifestEntry::entry_point_id(
  self : LunaAdmittedCapabilityManifestEntry,
) -> @luna_specialization.LunaSpecializedEntryPointId {
  self.entry.specialization.entry_point_id()
}

///|
pub fn LunaAdmittedCapabilityManifestEntry::workspace_bytes(
  self : LunaAdmittedCapabilityManifestEntry,
) -> Int64 {
  self.entry.specialization.workspace_bytes()
}

///|
pub fn LunaAdmittedCapabilityManifestEntry::workspace_alignment(
  self : LunaAdmittedCapabilityManifestEntry,
) -> Int64 {
  self.entry.specialization.workspace_alignment()
}

///|
pub fn LunaAdmittedCapabilityManifestEntry::minimum_alignment_bytes(
  self : LunaAdmittedCapabilityManifestEntry,
) -> Int64 {
  self.entry.minimum_alignment_bytes
}

///|
pub fn LunaAdmittedCapabilityManifestEntry::kv_layout(
  self : LunaAdmittedCapabilityManifestEntry,
) -> LunaKvLayoutContract {
  self.entry.kv_layout
}

///|
pub fn LunaAdmittedCapabilityManifestEntry::approved_source_identity(
  self : LunaAdmittedCapabilityManifestEntry,
) -> LunaApprovedImmutableSourceIdentity {
  self.approved_source
}

///|
pub fn LunaAdmittedCapabilityManifestEntry::approved_key_identity(
  self : LunaAdmittedCapabilityManifestEntry,
) -> LunaSignatureKeyIdentityDigest {
  self.key_identity
}

///|
pub fn LunaAdmittedCapabilityManifestEntry::detached_signature_digest(
  self : LunaAdmittedCapabilityManifestEntry,
) -> LunaDetachedSignatureDigest {
  self.signature_digest
}

///|
pub fn LunaAdmittedCapabilityManifestEntry::execution_disposition(
  _self : LunaAdmittedCapabilityManifestEntry,
) -> LunaGraphExecutionDisposition {
  LunaValidatedEagerFallback
}
