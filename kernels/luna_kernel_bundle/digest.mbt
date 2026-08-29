///|
fn kernel_bundle_is_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for ch in value {
    if !(ch >= '0' && ch <= '9') && !(ch >= 'a' && ch <= 'f') {
      return false
    }
  }
  true
}

///|
fn kernel_bundle_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}
