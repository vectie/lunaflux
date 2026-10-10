///|
/// Joins the immutable FP8 manifest's existing label to deployment metadata.
/// No canonical replay, payload checksum or signature verification occurs.
pub fn admit_luna_external_approved_fp8_release_v2(
  manifest : LunaFp8ReleaseManifestV2,
  envelope : LunaDetachedSignatureEnvelope,
  approval : LunaAuthenticatedExternalApproval,
) -> LunaAdmittedFp8ReleaseManifestV2 raise LunaCapabilityManifestError {
  if manifest.digest != approval.manifest_digest {
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
