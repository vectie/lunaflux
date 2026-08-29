///|
/// Private-channel decode seam. Static authority gates restrict this public
/// construction surface to the worker attestation codec; raw request and
/// ordinary manifest input must never call it.
pub fn LunaParentStartupApprovalWitness::from_private_parent_channel_v1(
  manifest_digest_sha256~ : String,
  key_identity_sha256~ : String,
  signature_digest_sha256~ : String,
  approved_source_sha256~ : String,
  bootstrap_digest_sha256~ : String,
  source_digest_sha256~ : String,
  model_generation~ : UInt64,
  device_ordinal~ : Int,
) -> LunaParentStartupApprovalWitness raise LunaCapabilityManifestError {
  if model_generation == 0UL || device_ordinal < 0 {
    raise Invalid(ExternalApproval)
  }
  {
    manifest_digest: LunaCapabilityManifestDigest::from_sha256(
      manifest_digest_sha256,
    ),
    key_identity: LunaSignatureKeyIdentityDigest::from_sha256(
      key_identity_sha256,
    ),
    signature_digest: LunaDetachedSignatureDigest::from_sha256(
      signature_digest_sha256,
    ),
    approved_source: LunaApprovedImmutableSourceIdentity::from_sha256(
      approved_source_sha256,
    ),
    bootstrap_digest: LunaCapabilityManifestDigest::from_sha256(
      bootstrap_digest_sha256,
    ),
    source_digest: LunaCapabilityManifestDigest::from_sha256(
      source_digest_sha256,
    ),
    model_generation,
    device_ordinal,
    consumed: false,
  }
}

///|
pub fn LunaParentStartupApprovalWitness::require_launch_identity(
  self : LunaParentStartupApprovalWitness,
  bootstrap_digest_sha256~ : String,
  source_digest_sha256~ : String,
  model_generation~ : UInt64,
  device_ordinal~ : Int,
) -> Unit raise LunaCapabilityManifestError {
  if self.consumed ||
    self.bootstrap_digest.as_hex() != bootstrap_digest_sha256 ||
    self.source_digest.as_hex() != source_digest_sha256 ||
    self.model_generation != model_generation ||
    self.device_ordinal != device_ordinal {
    raise Mismatch(ExternalApproval)
  }
}

///|
/// Reconstitutes approval authority only after the child has matched the exact
/// private-channel launch identity. The witness is consumed on every outcome.
pub fn LunaParentStartupApprovalWitness::authenticate_manifest_claim(
  self : LunaParentStartupApprovalWitness,
  canonical : Bytes,
  manifest_digest_sha256~ : String,
  key_identity_sha256~ : String,
  signature_digest_sha256~ : String,
  approved_source_sha256~ : String,
) -> LunaAuthenticatedExternalApproval raise LunaCapabilityManifestError {
  if self.consumed {
    raise Invalid(ExternalApproval)
  }
  self.consumed = true
  let actual = LunaCapabilityManifestDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  if actual != self.manifest_digest ||
    self.manifest_digest.as_hex() != manifest_digest_sha256 ||
    self.key_identity.as_hex() != key_identity_sha256 ||
    self.signature_digest.as_hex() != signature_digest_sha256 ||
    self.approved_source.as_hex() != approved_source_sha256 {
    raise Mismatch(ExternalApproval)
  }
  {
    manifest_digest: self.manifest_digest,
    algorithm: LunaExternalEd25519V1,
    key_identity: self.key_identity,
    signature_digest: self.signature_digest,
    approved_source: self.approved_source,
  }
}
