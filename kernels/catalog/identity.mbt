///|
/// Version of the serialized catalog contract.
pub struct CatalogVersion(Int) derive(Debug, Eq)

///|
/// Semantic version of one operation implementation contract.
pub struct KernelSemanticVersion(Int) derive(Debug, Eq)

///|
/// SHA-256 identity of one AOT CUDA module. No artifact path is stored.
pub struct AotArtifactDigest(String) derive(Debug, Eq, Hash)

///|
/// Stable identity of one callable kernel family inside an AOT module.
///
/// The integer is scoped by the module digest. It identifies an ABI family,
/// not a concrete launch shape or CUDA symbol.
pub struct AotKernelFamilyId(Int) derive(Debug, Eq, Hash)

///|
/// Content-addressed module plus one stable ABI family within that module.
pub struct AotKernelFamily {
  module_digest : AotArtifactDigest
  family_id : AotKernelFamilyId
} derive(Debug, Eq, Hash)

///|
/// Stable identity of one callable entry point within an AOT kernel family.
pub struct AotKernelEntryPointId(Int) derive(Debug, Eq, Hash)

///|
/// Fully qualified entry-point identity. CUDA symbol spelling is admitted
/// separately with the module bytes and is never used as this identity.
pub struct AotKernelEntryPoint {
  family : AotKernelFamily
  entry_point_id : AotKernelEntryPointId
} derive(Debug, Eq, Hash)

///|
/// Concrete production implementation selected at startup.
pub(all) enum KernelImplementationId {
  VendorBf16GemmImplicitHeuristicV1
  ContentAddressedAot(AotKernelFamily)
} derive(Debug, Eq)

///|
pub fn CatalogVersion::from_stable_int(
  value : Int,
) -> CatalogVersion raise CatalogError {
  if value <= 0 {
    raise InvalidCatalog(NonPositiveVersion)
  }
  CatalogVersion(value)
}

///|
pub fn CatalogVersion::v1() -> CatalogVersion {
  CatalogVersion(1)
}

///|
/// Catalog contract whose entries name the complete paged-KV semantic ABI.
pub fn CatalogVersion::v2() -> CatalogVersion {
  CatalogVersion(2)
}

///|
pub fn CatalogVersion::as_int(self : CatalogVersion) -> Int {
  self.0
}

///|
pub fn KernelSemanticVersion::from_stable_int(
  value : Int,
) -> KernelSemanticVersion raise CatalogError {
  if value <= 0 {
    raise InvalidCatalog(NonPositiveVersion)
  }
  KernelSemanticVersion(value)
}

///|
pub fn KernelSemanticVersion::v1() -> KernelSemanticVersion {
  KernelSemanticVersion(1)
}

///|
/// Paged-KV operation ABI with ordered live inputs and persistent state.
pub fn KernelSemanticVersion::v2() -> KernelSemanticVersion {
  KernelSemanticVersion(2)
}

///|
pub fn KernelSemanticVersion::as_int(self : KernelSemanticVersion) -> Int {
  self.0
}

///|
fn is_lower_sha256(value : String) -> Bool {
  if value.length() != 64 {
    return false
  }
  for character in value {
    if !(character is ('0'..='9') || character is ('a'..='f')) {
      return false
    }
  }
  true
}

///|
pub fn AotArtifactDigest::from_sha256(
  value : String,
) -> AotArtifactDigest raise CatalogError {
  if !is_lower_sha256(value) {
    raise InvalidCatalog(InvalidArtifactDigest)
  }
  AotArtifactDigest(value)
}

///|
pub fn AotArtifactDigest::sha256(self : AotArtifactDigest) -> String {
  self.0
}

///|
pub fn AotKernelFamilyId::from_stable_int(
  value : Int,
) -> AotKernelFamilyId raise CatalogError {
  if value <= 0 {
    raise InvalidCatalog(InvalidAotFamilyId)
  }
  AotKernelFamilyId(value)
}

///|
pub fn AotKernelFamilyId::as_int(self : AotKernelFamilyId) -> Int {
  self.0
}

///|
pub fn AotKernelFamily::new(
  module_digest~ : AotArtifactDigest,
  family_id~ : AotKernelFamilyId,
) -> AotKernelFamily {
  { module_digest, family_id }
}

///|
pub fn AotKernelFamily::module_digest(
  self : AotKernelFamily,
) -> AotArtifactDigest {
  self.module_digest
}

///|
pub fn AotKernelFamily::family_id(self : AotKernelFamily) -> AotKernelFamilyId {
  self.family_id
}

///|
pub fn AotKernelEntryPointId::from_stable_int(
  value : Int,
) -> AotKernelEntryPointId raise CatalogError {
  if value <= 0 {
    raise InvalidCatalog(InvalidAotEntryPointId)
  }
  AotKernelEntryPointId(value)
}

///|
pub fn AotKernelEntryPointId::as_int(self : AotKernelEntryPointId) -> Int {
  self.0
}

///|
pub fn AotKernelEntryPoint::new(
  family~ : AotKernelFamily,
  entry_point_id~ : AotKernelEntryPointId,
) -> AotKernelEntryPoint {
  { family, entry_point_id }
}

///|
pub fn AotKernelEntryPoint::family(
  self : AotKernelEntryPoint,
) -> AotKernelFamily {
  self.family
}

///|
pub fn AotKernelEntryPoint::entry_point_id(
  self : AotKernelEntryPoint,
) -> AotKernelEntryPointId {
  self.entry_point_id
}
