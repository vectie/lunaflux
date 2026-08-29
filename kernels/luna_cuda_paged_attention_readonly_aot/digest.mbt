///|
pub struct PagedAttentionReadOnlyDigest(String) derive(Debug, Eq, Hash)

///|
fn readonly_is_sha256(value : String) -> Bool {
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
pub fn PagedAttentionReadOnlyDigest::from_sha256(
  value : String,
) -> PagedAttentionReadOnlyDigest raise PagedAttentionReadOnlyAotError {
  if !readonly_is_sha256(value) {
    raise PagedAttentionReadOnlyInvalid(Digest)
  }
  PagedAttentionReadOnlyDigest(value)
}

///|
pub fn PagedAttentionReadOnlyDigest::as_hex(
  self : PagedAttentionReadOnlyDigest,
) -> String {
  self.0
}

///|
fn readonly_digest(bytes : Bytes) -> PagedAttentionReadOnlyDigest {
  PagedAttentionReadOnlyDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
}
