///|
/// Admits exact canonical bundle evidence under an independent digest.
pub fn admit_deployment_bundle_evidence(
  bytes : Bytes,
  expected_digest~ : DeploymentBundleDigest,
  limits~ : DeploymentBundleLimits,
) -> DeploymentBundleEvidence raise DeploymentBundleError {
  let actual = @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
  if actual != expected_digest.as_hex() {
    raise DigestMismatch
  }
  parse_canonical_evidence(bytes, expected_digest, limits)
}

///|
/// Projects admitted bundle evidence into the generic Phase-9 artifact type.
///
/// This does not mark any campaign role passed. It only creates the exact
/// locator/length/digest value a separately verified campaign may observe.
pub fn DeploymentBundleEvidence::as_release_artifact(
  self : DeploymentBundleEvidence,
  locator : StringView,
  limits : @release_evidence.ReleaseCampaignLimits,
) -> @release_evidence.ReleaseEvidenceArtifact raise DeploymentBundleError {
  let digest = @release_evidence.ReleaseArtifactDigest::from_sha256(
    self.digest.as_hex(),
  ) catch {
    _ => raise ReleaseEvidenceRejected
  }
  @release_evidence.ReleaseEvidenceArtifact::new(
    locator,
    byte_length=Int64::from_int(self.canonical.length()),
    digest~,
    limits~,
  ) catch {
    _ => raise ReleaseEvidenceRejected
  }
}
