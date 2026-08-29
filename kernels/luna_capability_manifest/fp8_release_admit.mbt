///|
/// Joins the exact FP8 release-manifest bytes to the existing externally
/// verified deployment approval. This function verifies no signature itself.
pub fn admit_luna_external_approved_fp8_release_v2(
  manifest : LunaFp8ReleaseManifestV2,
  envelope : LunaDetachedSignatureEnvelope,
  approval : LunaAuthenticatedExternalApproval,
) -> LunaAdmittedFp8ReleaseManifestV2 raise LunaCapabilityManifestError {
  let digest = LunaCapabilityManifestDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(manifest.canonical)),
  )
  if digest != manifest.digest || digest != approval.manifest_digest {
    raise Mismatch(Digest)
  }
  if envelope.algorithm != approval.algorithm ||
    envelope.key_identity != approval.key_identity ||
    envelope.signature_digest != approval.signature_digest {
    raise Mismatch(ExternalApproval)
  }
  {
    manifest,
    approved_source: approval.approved_source,
    key_identity: approval.key_identity,
    signature_digest: approval.signature_digest,
  }
}
