///|
fn projection_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn projection_is_sha256(value : StringView) -> Bool {
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
pub struct ProjectionCudaSourceDigest(String) derive(Debug, Eq, Hash)

///|
pub fn ProjectionCudaSourceDigest::from_sha256(
  value : String,
) -> ProjectionCudaSourceDigest raise ProjectionAotError {
  if !projection_is_sha256(value) {
    raise Invalid(Digest)
  }
  ProjectionCudaSourceDigest(value)
}

///|
pub fn ProjectionCudaSourceDigest::as_hex(
  self : ProjectionCudaSourceDigest,
) -> String {
  self.0
}

///|
pub struct ProjectionCudaRecipeDigest(String) derive(Debug, Eq, Hash)

///|
pub struct ProjectionCudaManifestBindingDigest(String) derive(Debug, Eq, Hash)

///|
pub fn ProjectionCudaRecipeDigest::from_sha256(
  value : String,
) -> ProjectionCudaRecipeDigest raise ProjectionAotError {
  if !projection_is_sha256(value) {
    raise Invalid(Digest)
  }
  ProjectionCudaRecipeDigest(value)
}

///|
pub fn ProjectionCudaRecipeDigest::as_hex(
  self : ProjectionCudaRecipeDigest,
) -> String {
  self.0
}

///|
pub fn ProjectionCudaManifestBindingDigest::as_hex(
  self : ProjectionCudaManifestBindingDigest,
) -> String {
  self.0
}

///|
pub fn projection_content_addressed_cubin_path(
  artifact_sha256 : String,
) -> String raise ProjectionAotError {
  if !projection_is_sha256(artifact_sha256) {
    raise Invalid(Digest)
  }
  "sha256/" + artifact_sha256 + ".cubin"
}
