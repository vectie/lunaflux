///|
pub struct FusedParallelDigest(String) derive(Debug, Eq, Hash)

///|
fn fused_parallel_is_sha256(value : String) -> Bool {
  if value.length() != 64 {
    return false
  }
  for character in value {
    if !(character is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}

///|
pub fn FusedParallelDigest::from_sha256(
  value : String,
) -> FusedParallelDigest raise FusedParallelAotError {
  if !fused_parallel_is_sha256(value) || value == "0".repeat(64) {
    raise FusedParallelInvalid(Digest)
  }
  FusedParallelDigest(value)
}

///|
pub fn FusedParallelDigest::as_hex(self : FusedParallelDigest) -> String {
  self.0
}

///|
fn fused_parallel_digest(bytes : Bytes) -> FusedParallelDigest {
  FusedParallelDigest(@crypto.bytes_to_hex_string(@crypto.sha256(bytes)))
}
