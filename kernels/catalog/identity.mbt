///|
/// Version of the serialized catalog contract.
pub struct CatalogVersion(Int) derive(Debug, Eq)

///|
/// Semantic version of one operation implementation contract.
pub struct KernelSemanticVersion(Int) derive(Debug, Eq)

///|
/// SHA-256 identity of one AOT kernel artifact. No artifact path is stored.
pub struct AotArtifactDigest(String) derive(Debug, Eq)

///|
/// Concrete production implementation selected at startup.
pub(all) enum KernelImplementationId {
  VendorBf16GemmImplicitHeuristicV1
  ContentAddressedAot(AotArtifactDigest)
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
