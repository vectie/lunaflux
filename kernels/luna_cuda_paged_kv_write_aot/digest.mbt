///|
fn paged_kv_write_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn paged_kv_write_is_sha256(value : StringView) -> Bool {
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
pub struct PagedKvWriteDigest(String) derive(Debug, Eq, Hash)

///|
pub fn PagedKvWriteDigest::from_sha256(
  value : String,
) -> PagedKvWriteDigest raise PagedKvWriteAotError {
  if !paged_kv_write_is_sha256(value) {
    raise Invalid(Digest)
  }
  PagedKvWriteDigest(value)
}

///|
pub fn PagedKvWriteDigest::as_hex(self : PagedKvWriteDigest) -> String {
  self.0
}
