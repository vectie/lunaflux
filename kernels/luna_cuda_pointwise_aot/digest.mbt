///|
fn pointwise_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
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
pub struct PointwiseSourceDigest(String) derive(Debug, Eq, Hash)

///|
pub fn PointwiseSourceDigest::from_sha256(
  value : String,
) -> PointwiseSourceDigest raise PointwiseAotError {
  if !is_lower_sha256(value) {
    raise Invalid(Digest)
  }
  PointwiseSourceDigest(value)
}

///|
pub fn PointwiseSourceDigest::as_hex(self : PointwiseSourceDigest) -> String {
  self.0
}

///|
pub struct PointwiseRecipeDigest(String) derive(Debug, Eq, Hash)

///|
pub fn PointwiseRecipeDigest::from_sha256(
  value : String,
) -> PointwiseRecipeDigest raise PointwiseAotError {
  if !is_lower_sha256(value) {
    raise Invalid(Digest)
  }
  PointwiseRecipeDigest(value)
}

///|
pub fn PointwiseRecipeDigest::as_hex(self : PointwiseRecipeDigest) -> String {
  self.0
}

///|
pub struct PointwiseManifestBindingDigest(String) derive(Debug, Eq, Hash)

///|
pub fn PointwiseManifestBindingDigest::as_hex(
  self : PointwiseManifestBindingDigest,
) -> String {
  self.0
}

///|
pub fn content_addressed_cubin_path(
  artifact_sha256 : String,
) -> String raise PointwiseAotError {
  if !is_lower_sha256(artifact_sha256) {
    raise Invalid(Digest)
  }
  "sha256/" + artifact_sha256 + ".cubin"
}
