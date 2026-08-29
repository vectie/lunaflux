///|
fn paged_attention_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn paged_attention_is_sha256(value : StringView) -> Bool {
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
pub struct PagedAttentionCudaSourceDigest(String) derive(Debug, Eq, Hash)

///|
pub fn PagedAttentionCudaSourceDigest::from_sha256(
  value : String,
) -> PagedAttentionCudaSourceDigest raise PagedAttentionAotError {
  if !paged_attention_is_sha256(value) {
    raise Invalid(Source)
  }
  PagedAttentionCudaSourceDigest(value)
}

///|
pub fn PagedAttentionCudaSourceDigest::as_hex(
  self : PagedAttentionCudaSourceDigest,
) -> String {
  self.0
}

///|
pub struct PagedAttentionCudaRecipeDigest(String) derive(Debug, Eq, Hash)

///|
pub struct PagedAttentionManifestBindingDigest(String) derive(Debug, Eq, Hash)

///|
pub fn PagedAttentionCudaRecipeDigest::from_sha256(
  value : String,
) -> PagedAttentionCudaRecipeDigest raise PagedAttentionAotError {
  if !paged_attention_is_sha256(value) {
    raise Invalid(Recipe)
  }
  PagedAttentionCudaRecipeDigest(value)
}

///|
pub fn PagedAttentionCudaRecipeDigest::as_hex(
  self : PagedAttentionCudaRecipeDigest,
) -> String {
  self.0
}

///|
pub fn PagedAttentionManifestBindingDigest::as_hex(
  self : PagedAttentionManifestBindingDigest,
) -> String {
  self.0
}
