///|
pub struct ReadOnlyProductionDigest(String) derive(Debug, Eq, Hash)

///|
pub fn ReadOnlyProductionDigest::from_sha256(
  value : String,
) -> ReadOnlyProductionDigest raise ReadOnlyProductionError {
  if value.length() != 64 {
    raise InvalidReadOnlyProduction(Candidate)
  }
  for character in value {
    if !(character >= '0' && character <= '9') &&
      !(character >= 'a' && character <= 'f') {
      raise InvalidReadOnlyProduction(Candidate)
    }
  }
  ReadOnlyProductionDigest(value)
}

///|
pub fn ReadOnlyProductionDigest::as_hex(
  self : ReadOnlyProductionDigest,
) -> String {
  self.0
}

///|
fn readonly_production_digest(bytes : Bytes) -> ReadOnlyProductionDigest {
  ReadOnlyProductionDigest(@crypto.bytes_to_hex_string(@crypto.sha256(bytes)))
}
