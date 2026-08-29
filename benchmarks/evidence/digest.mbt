///|
/// A validated lowercase SHA-256. Digests bind external artifacts and raw
/// evidence without granting filesystem or process authority.
pub struct BenchmarkDigest {
  priv hex : String
} derive(Debug, Eq, Hash)

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
pub fn BenchmarkDigest::from_sha256(
  value : String,
) -> BenchmarkDigest raise BenchmarkEvidenceError {
  if !is_lower_sha256(value) {
    raise InvalidDigest
  }
  { hex: value }
}

///|
pub fn BenchmarkDigest::as_hex(self : BenchmarkDigest) -> String {
  self.hex
}
