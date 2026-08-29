///|
fn readonly_physical_valid_sha256(value : String) -> Bool {
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
pub struct PagedAttentionReadOnlyPhysicalEvidenceDigest(String) derive (
  Debug,
  Eq,
  Hash,
)

///|
pub fn PagedAttentionReadOnlyPhysicalEvidenceDigest::from_sha256(
  value : String,
) -> PagedAttentionReadOnlyPhysicalEvidenceDigest raise PagedAttentionReadOnlyPhysicalEvidenceError {
  if !readonly_physical_valid_sha256(value) || value == "0".repeat(64) {
    raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Digest)
  }
  PagedAttentionReadOnlyPhysicalEvidenceDigest(value)
}

///|
pub fn PagedAttentionReadOnlyPhysicalEvidenceDigest::as_hex(
  self : PagedAttentionReadOnlyPhysicalEvidenceDigest,
) -> String {
  self.0
}

///|
fn readonly_physical_digest(
  bytes : Bytes,
) -> PagedAttentionReadOnlyPhysicalEvidenceDigest {
  let state = @canonical_sha256.CanonicalSha256::new()
  if !state.update_bytes(bytes) {
    abort("fresh read-only evidence SHA-256 rejected bytes")
  }
  PagedAttentionReadOnlyPhysicalEvidenceDigest::from_sha256(
    state.finalize_hex(),
  ) catch {
    _ => abort("read-only evidence SHA-256 returned malformed output")
  }
}

///|
fn readonly_physical_read_digest(
  value : String,
) -> PagedAttentionReadOnlyPhysicalEvidenceDigest raise PagedAttentionReadOnlyPhysicalEvidenceError {
  PagedAttentionReadOnlyPhysicalEvidenceDigest::from_sha256(value) catch {
    _ => raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Digest)
  }
}
