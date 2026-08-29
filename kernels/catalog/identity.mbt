///|
/// Version of the serialized catalog contract.
pub struct CatalogVersion {
  priv value : Int
} derive(Debug, Eq)

///|
/// Semantic version of one operation implementation contract.
pub struct KernelSemanticVersion {
  priv value : Int
} derive(Debug, Eq)

///|
/// SHA-256 identity of one AOT CUDA module. No artifact path is stored.
pub struct AotArtifactDigest {
  priv value : String
} derive(Debug, Eq, Hash)

///|
/// Stable identity of one callable kernel family inside an AOT module.
///
/// The integer is scoped by the module digest. It identifies an ABI family,
/// not a concrete launch shape or CUDA symbol.
pub struct AotKernelFamilyId {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
/// Content-addressed module plus one stable ABI family within that module.
pub struct AotKernelFamily {
  priv module_digest : AotArtifactDigest
  priv family_id : AotKernelFamilyId
} derive(Debug, Eq, Hash)

///|
/// Stable identity of one callable entry point within an AOT kernel family.
pub struct AotKernelEntryPointId {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
/// Fully qualified entry-point identity. CUDA symbol spelling is admitted
/// separately with the module bytes and is never used as this identity.
pub struct AotKernelEntryPoint {
  priv family : AotKernelFamily
  priv entry_point_id : AotKernelEntryPointId
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
  { value, }
}

///|
pub fn CatalogVersion::v1() -> CatalogVersion {
  { value: 1 }
}

///|
/// Catalog contract whose entries name the complete paged-KV semantic ABI.
pub fn CatalogVersion::v3() -> CatalogVersion {
  { value: 3 }
}

///|
/// Catalog contract whose entries bind the exact numeric operation ABI.
pub fn CatalogVersion::v4() -> CatalogVersion {
  { value: 4 }
}

///|
pub fn CatalogVersion::as_int(self : CatalogVersion) -> Int {
  self.value
}

///|
pub fn KernelSemanticVersion::from_stable_int(
  value : Int,
) -> KernelSemanticVersion raise CatalogError {
  if value <= 0 {
    raise InvalidCatalog(NonPositiveVersion)
  }
  { value, }
}

///|
pub fn KernelSemanticVersion::v1() -> KernelSemanticVersion {
  { value: 1 }
}

///|
/// Paged-KV operation ABI with ordered live inputs and persistent state.
pub fn KernelSemanticVersion::v3() -> KernelSemanticVersion {
  { value: 3 }
}

///|
/// Paged-KV operation ABI bound to one exact numeric execution digest.
pub fn KernelSemanticVersion::v4() -> KernelSemanticVersion {
  { value: 4 }
}

///|
pub fn KernelSemanticVersion::as_int(self : KernelSemanticVersion) -> Int {
  self.value
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
  { value, }
}

///|
pub fn AotArtifactDigest::sha256(self : AotArtifactDigest) -> String {
  self.value
}

///|
pub fn AotKernelFamilyId::from_stable_int(
  value : Int,
) -> AotKernelFamilyId raise CatalogError {
  if value <= 0 {
    raise InvalidCatalog(InvalidAotFamilyId)
  }
  { value, }
}

///|
pub fn AotKernelFamilyId::as_int(self : AotKernelFamilyId) -> Int {
  self.value
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
  { value, }
}

///|
pub fn AotKernelEntryPointId::as_int(self : AotKernelEntryPointId) -> Int {
  self.value
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
