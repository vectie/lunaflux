///|
/// Joins v3 reusable-paged signing bytes to a deployment-owned detached
/// signature approval. LunaFlux does not verify the signature itself.
pub fn admit_luna_external_approved_fp8_reusable_paged_release_v3(
  manifest : LunaFp8ReusablePagedReleaseManifestV3,
  envelope : LunaDetachedSignatureEnvelope,
  approval : LunaAuthenticatedExternalApproval,
) -> LunaAdmittedFp8ReusablePagedReleaseManifestV3 raise LunaCapabilityManifestError {
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
