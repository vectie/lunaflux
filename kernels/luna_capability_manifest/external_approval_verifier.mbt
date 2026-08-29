///|
fn luna_external_approval_decode_tag(value : String) -> FixedArray[Byte]? {
  if value.length() != 64 {
    return None
  }
  let output = FixedArray::make(32, b'\x00')
  for index in 0..<32 {
    let high = luna_external_approval_nibble(value[index * 2])
    let low = luna_external_approval_nibble(value[index * 2 + 1])
    guard high is Some(h) && low is Some(l) else {
      output.fill(b'\x00')
      return None
    }
    output[index] = h.shl(4).lor(l)
  }
  Some(output)
}

///|
fn luna_external_approval_nibble(value : UInt16) -> Byte? {
  if value >= '0' && value <= '9' {
    return Some((value - '0').to_byte())
  }
  if value >= 'a' && value <= 'f' {
    return Some((value - 'a' + 10).to_byte())
  }
  None
}

///|
fn luna_external_approval_constant_time_equal(
  left : FixedArray[Byte],
  right : FixedArray[Byte],
) -> Bool {
  if left.length() != 32 || right.length() != 32 {
    return false
  }
  let mut difference : Byte = b'\x00'
  for index in 0..<32 {
    difference = difference.lor(left[index].lxor(right[index]))
  }
  difference == b'\x00'
}

///|
fn luna_external_approval_key_is_nonzero(key : FixedArray[Byte]) -> Bool {
  let mut combined : Byte = b'\x00'
  for byte in key {
    combined = combined.lor(byte)
  }
  combined != b'\x00'
}

///|
fn luna_external_approval_hmac(
  key : FixedArray[Byte],
  material : Bytes,
) -> FixedArray[Byte] {
  let inner_pad = FixedArray::make(64, b'\x36')
  let outer_pad = FixedArray::make(64, b'\x5c')
  for index in 0..<key.length() {
    inner_pad[index] = inner_pad[index].lxor(key[index])
    outer_pad[index] = outer_pad[index].lxor(key[index])
  }
  let inner = @crypto.SHA256::new()
  inner.update(inner_pad)
  inner.update(material)
  let inner_digest = inner.finalize()
  let outer = @crypto.SHA256::new()
  outer.update(outer_pad)
  outer.update(inner_digest)
  let digest = outer.finalize()
  inner.reset()
  outer.reset()
  inner_pad.fill(b'\x00')
  outer_pad.fill(b'\x00')
  inner_digest.fill(b'\x00')
  digest
}

///|
fn LunaExternalSignedApproval::signing_bytes(
  self : LunaExternalSignedApproval,
) -> Bytes {
  @utf8.encode(
    LUNA_EXTERNAL_APPROVAL_HMAC_DOMAIN +
    "|" +
    self.schema +
    "|" +
    self.rollout_digest.as_hex() +
    "|" +
    self.verifier_key_id +
    "|" +
    self.authority_id +
    "|" +
    self.manifest_digest.as_hex() +
    "|luna-external-signature-v1" +
    "|" +
    self.key_identity.as_hex() +
    "|" +
    self.signature_digest.as_hex() +
    "|" +
    self.approved_source.as_hex(),
  )
}

///|
pub fn LunaExternalApprovalVerifier::close(
  self : LunaExternalApprovalVerifier,
) -> Unit {
  if !self.closed {
    self.key.fill(b'\x00')
    self.closed = true
  }
}

///|
/// Authenticates one exact canonical record and all envelope/source subjects.
/// The verifier cannot be created by untrusted runtime input; closed instances
/// reject permanently and every temporary key-derived buffer is wiped.
pub fn LunaExternalApprovalVerifier::verify(
  self : LunaExternalApprovalVerifier,
  canonical : Bytes,
  envelope : LunaDetachedSignatureEnvelope,
  signed : LunaExternalSignedApproval,
  approved_source : LunaApprovedImmutableSourceIdentity,
) -> LunaAuthenticatedExternalApproval raise LunaCapabilityManifestError {
  if self.closed ||
    self.key.length() < 32 ||
    self.key.length() > 64 ||
    !luna_external_approval_key_is_nonzero(self.key) ||
    canonical.is_empty() ||
    canonical.length() > 16777216 {
    raise Invalid(ExternalApproval)
  }
  let manifest_digest = LunaCapabilityManifestDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  if signed.schema != LUNA_EXTERNAL_SIGNED_APPROVAL_SCHEMA ||
    signed.rollout_digest != self.policy.rollout_digest ||
    signed.verifier_key_id != self.policy.verifier_key_id ||
    signed.authority_id != self.policy.authority_id ||
    signed.manifest_digest != manifest_digest ||
    signed.algorithm != envelope.algorithm ||
    signed.key_identity != envelope.key_identity ||
    signed.signature_digest != envelope.signature_digest ||
    signed.approved_source != approved_source {
    raise Mismatch(ExternalApproval)
  }
  guard luna_external_approval_decode_tag(signed.authentication_tag.as_hex())
    is Some(observed) else {
    raise Invalid(ExternalApproval)
  }
  let material = signed.signing_bytes()
  let expected = luna_external_approval_hmac(self.key, material)
  let authenticated = luna_external_approval_constant_time_equal(
    observed, expected,
  )
  observed.fill(b'\x00')
  expected.fill(b'\x00')
  if !authenticated {
    raise Mismatch(ExternalApproval)
  }
  {
    manifest_digest,
    algorithm: signed.algorithm,
    key_identity: signed.key_identity,
    signature_digest: signed.signature_digest,
    approved_source,
  }
}
