///|
fn tensor_core_evidence_valid_sha256(value : String) -> Bool {
  if value.length() != 64 {
    return false
  }
  for character in value {
    if !(character >= '0' && character <= '9') &&
      !(character >= 'a' && character <= 'f') {
      return false
    }
  }
  true
}

///|
pub struct LunaTensorCorePhysicalEvidenceDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaTensorCorePhysicalEvidenceDigest::from_sha256(
  value : String,
) -> LunaTensorCorePhysicalEvidenceDigest raise LunaTensorCorePhysicalEvidenceError {
  if !tensor_core_evidence_valid_sha256(value) || value == "0".repeat(64) {
    raise LunaTensorCorePhysicalEvidenceInvalid(Digest)
  }
  LunaTensorCorePhysicalEvidenceDigest(value)
}

///|
pub fn LunaTensorCorePhysicalEvidenceDigest::as_hex(
  self : LunaTensorCorePhysicalEvidenceDigest,
) -> String {
  self.0
}

///|
fn tensor_core_evidence_sha256(
  bytes : Bytes,
) -> LunaTensorCorePhysicalEvidenceDigest {
  let state = @canonical_sha256.CanonicalSha256::new()
  if !state.update_bytes(bytes) {
    abort("fresh tensor-core evidence SHA-256 rejected bytes")
  }
  LunaTensorCorePhysicalEvidenceDigest::from_sha256(state.finalize_hex()) catch {
    _ => abort("tensor-core evidence SHA-256 produced invalid bytes")
  }
}

///|
fn tensor_core_evidence_read_digest(
  value : String,
) -> LunaTensorCorePhysicalEvidenceDigest raise LunaTensorCorePhysicalEvidenceError {
  LunaTensorCorePhysicalEvidenceDigest::from_sha256(value) catch {
    _ => raise LunaTensorCorePhysicalEvidenceInvalid(Digest)
  }
}
