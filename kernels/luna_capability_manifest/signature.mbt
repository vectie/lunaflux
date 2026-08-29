///|
pub(all) enum LunaDetachedSignatureAlgorithm {
  LunaExternalEd25519V1
} derive(Debug, Eq)

///|
pub struct LunaSignatureKeyIdentityDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaSignatureKeyIdentityDigest::from_sha256(
  value : String,
) -> LunaSignatureKeyIdentityDigest raise LunaCapabilityManifestError {
  if !luna_capability_is_sha256(value) {
    raise Invalid(SignatureEnvelope)
  }
  LunaSignatureKeyIdentityDigest(value)
}

///|
pub fn LunaSignatureKeyIdentityDigest::as_hex(
  self : LunaSignatureKeyIdentityDigest,
) -> String {
  self.0
}

///|
pub struct LunaDetachedSignatureDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaDetachedSignatureDigest::from_sha256(
  value : String,
) -> LunaDetachedSignatureDigest raise LunaCapabilityManifestError {
  if !luna_capability_is_sha256(value) {
    raise Invalid(SignatureEnvelope)
  }
  LunaDetachedSignatureDigest(value)
}

///|
pub fn LunaDetachedSignatureDigest::as_hex(
  self : LunaDetachedSignatureDigest,
) -> String {
  self.0
}

///|
pub struct LunaApprovedImmutableSourceIdentity(String) derive(Debug, Eq, Hash)

///|
pub fn LunaApprovedImmutableSourceIdentity::from_sha256(
  value : String,
) -> LunaApprovedImmutableSourceIdentity raise LunaCapabilityManifestError {
  if !luna_capability_is_sha256(value) {
    raise Invalid(ApprovedSource)
  }
  LunaApprovedImmutableSourceIdentity(value)
}

///|
pub fn LunaApprovedImmutableSourceIdentity::as_hex(
  self : LunaApprovedImmutableSourceIdentity,
) -> String {
  self.0
}

///|
pub struct LunaDetachedSignatureEnvelope {
  priv algorithm : LunaDetachedSignatureAlgorithm
  priv key_identity : LunaSignatureKeyIdentityDigest
  priv signature_digest : LunaDetachedSignatureDigest
} derive(Debug, Eq)

///|
/// Records detached-signature metadata without granting verification authority.
///
/// LunaFlux compares this inert envelope with an externally verified approval
/// receipt. It never interprets the digest as proof by itself.
pub fn LunaDetachedSignatureEnvelope::new(
  algorithm~ : LunaDetachedSignatureAlgorithm,
  key_identity~ : LunaSignatureKeyIdentityDigest,
  signature_digest~ : LunaDetachedSignatureDigest,
) -> LunaDetachedSignatureEnvelope {
  { algorithm, key_identity, signature_digest }
}

///|
pub struct LunaAuthenticatedExternalApproval {
  priv manifest_digest : LunaCapabilityManifestDigest
  priv algorithm : LunaDetachedSignatureAlgorithm
  priv key_identity : LunaSignatureKeyIdentityDigest
  priv signature_digest : LunaDetachedSignatureDigest
  priv approved_source : LunaApprovedImmutableSourceIdentity
} derive(Debug, Eq)

///|
/// Opaque acknowledgement that the deployment trust boundary approved one
/// exact immutable record. The record's owning package still decides what the
/// approved bytes mean; this type only carries the external signature join.
pub struct LunaAdmittedExternalRecord {
  priv record_digest : LunaCapabilityManifestDigest
  priv approved_source : LunaApprovedImmutableSourceIdentity
  priv key_identity : LunaSignatureKeyIdentityDigest
  priv signature_digest : LunaDetachedSignatureDigest
} derive(Debug, Eq)

///|
/// Replays a deployment-owned detached-signature approval against the exact
/// immutable bytes supplied by an owning evidence package. This function does
/// not verify signatures and cannot manufacture an approval receipt.
pub fn admit_luna_external_approved_record(
  canonical : Bytes,
  envelope : LunaDetachedSignatureEnvelope,
  approval : LunaAuthenticatedExternalApproval,
  maximum_canonical_bytes~ : Int,
) -> LunaAdmittedExternalRecord raise LunaCapabilityManifestError {
  if maximum_canonical_bytes <= 0 ||
    maximum_canonical_bytes > 16777216 ||
    canonical.is_empty() ||
    canonical.length() > maximum_canonical_bytes {
    raise Invalid(CanonicalBytes)
  }
  let actual = LunaCapabilityManifestDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  if approval.manifest_digest != actual {
    raise Mismatch(Digest)
  }
  if envelope.algorithm != approval.algorithm ||
    envelope.key_identity != approval.key_identity ||
    envelope.signature_digest != approval.signature_digest {
    raise Mismatch(ExternalApproval)
  }
  {
    record_digest: actual,
    approved_source: approval.approved_source,
    key_identity: approval.key_identity,
    signature_digest: approval.signature_digest,
  }
}

///|
pub fn LunaAdmittedExternalRecord::record_digest(
  self : LunaAdmittedExternalRecord,
) -> LunaCapabilityManifestDigest {
  self.record_digest
}

///|
pub fn LunaAdmittedExternalRecord::approved_source_identity(
  self : LunaAdmittedExternalRecord,
) -> LunaApprovedImmutableSourceIdentity {
  self.approved_source
}

///|
pub fn LunaAdmittedExternalRecord::approved_key_identity(
  self : LunaAdmittedExternalRecord,
) -> LunaSignatureKeyIdentityDigest {
  self.key_identity
}

///|
pub fn LunaAdmittedExternalRecord::detached_signature_digest(
  self : LunaAdmittedExternalRecord,
) -> LunaDetachedSignatureDigest {
  self.signature_digest
}
