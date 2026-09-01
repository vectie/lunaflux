///|
pub struct QkNormRopePagedKvWriteDigest(String) derive(Debug, Eq, Hash)

///|
pub fn QkNormRopePagedKvWriteDigest::from_sha256(
  value : String,
) -> QkNormRopePagedKvWriteDigest raise QkNormRopePagedKvWriteAotError {
  if !postprocess_is_sha256(value) {
    raise Invalid(Binding)
  }
  QkNormRopePagedKvWriteDigest(value)
}

///|
pub fn QkNormRopePagedKvWriteDigest::as_hex(
  self : QkNormRopePagedKvWriteDigest,
) -> String {
  self.0
}

///|
fn postprocess_sha256(bytes : BytesView) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn postprocess_is_sha256(value : String) -> Bool {
  if value.length() != 64 {
    return false
  }
  for code in value {
    if !(code >= '0' && code <= '9') && !(code >= 'a' && code <= 'f') {
      return false
    }
  }
  true
}
