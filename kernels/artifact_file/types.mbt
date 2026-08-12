///|
/// Explicit SHA-256 identity supplied independently of the manifest file.
pub struct KernelManifestDigest(String) derive(Debug, Eq)

///|
/// Parsed exact v1 AOT manifest, already pinned to one launch-contract set.
pub struct KernelArtifactManifest {
  module_digests : ReadOnlyArray[@catalog.AotArtifactDigest]
  module_path_identifiers : ReadOnlyArray[String]
  entry_points : ReadOnlyArray[@artifact.KernelEntryPointInput]
}

///|
fn is_lower_sha256(value : StringView) -> Bool {
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
pub fn KernelManifestDigest::from_sha256(
  value : String,
) -> KernelManifestDigest raise KernelArtifactFileError {
  if !is_lower_sha256(value) {
    raise InvalidExpectedManifestDigest
  }
  KernelManifestDigest(value)
}

///|
pub fn KernelManifestDigest::as_hex(self : KernelManifestDigest) -> String {
  self.0
}

///|
pub fn KernelArtifactManifest::module_count(
  self : KernelArtifactManifest,
) -> Int {
  self.module_digests.length()
}

///|
pub fn KernelArtifactManifest::entry_point_count(
  self : KernelArtifactManifest,
) -> Int {
  self.entry_points.length()
}
