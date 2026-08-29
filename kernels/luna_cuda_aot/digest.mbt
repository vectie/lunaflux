///|
fn luna_cuda_aot_is_sha256(value : StringView) -> Bool {
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
fn luna_cuda_aot_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
pub struct LunaCudaSourceDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaCudaSourceDigest::from_sha256(
  value : String,
) -> LunaCudaSourceDigest raise LunaCudaAotError {
  if !luna_cuda_aot_is_sha256(value) {
    raise Invalid(Source)
  }
  LunaCudaSourceDigest(value)
}

///|
pub fn LunaCudaSourceDigest::as_hex(self : LunaCudaSourceDigest) -> String {
  self.0
}

///|
pub struct LunaCudaRecipeDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaCudaRecipeDigest::from_sha256(
  value : String,
) -> LunaCudaRecipeDigest raise LunaCudaAotError {
  if !luna_cuda_aot_is_sha256(value) {
    raise Invalid(Recipe)
  }
  LunaCudaRecipeDigest(value)
}

///|
pub fn LunaCudaRecipeDigest::as_hex(self : LunaCudaRecipeDigest) -> String {
  self.0
}

///|
pub struct LunaCudaArtifactDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaCudaArtifactDigest::from_sha256(
  value : String,
) -> LunaCudaArtifactDigest raise LunaCudaAotError {
  if !luna_cuda_aot_is_sha256(value) {
    raise Invalid(Artifact)
  }
  LunaCudaArtifactDigest(value)
}

///|
pub fn LunaCudaArtifactDigest::as_hex(self : LunaCudaArtifactDigest) -> String {
  self.0
}

///|
pub struct LunaCudaEvidenceDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaCudaEvidenceDigest::from_sha256(
  value : String,
) -> LunaCudaEvidenceDigest raise LunaCudaAotError {
  if !luna_cuda_aot_is_sha256(value) {
    raise Invalid(Evidence)
  }
  LunaCudaEvidenceDigest(value)
}

///|
pub fn LunaCudaEvidenceDigest::as_hex(self : LunaCudaEvidenceDigest) -> String {
  self.0
}
