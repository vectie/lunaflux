///|
/// Payload-free failures from release-campaign construction, admission, and
/// host verification. Artifact locators and externally supplied identities
/// never become error payloads.
pub(all) suberror ReleaseEvidenceError {
  InvalidLimits
  InvalidDigest
  InvalidCampaignIdentity
  InvalidArtifact
  InvalidEvidenceEntry
  InvalidReviewerEntry
  InvalidCanonicalRecord
  CampaignDigestMismatch
  VerificationPolicyMismatch
  InvalidObservation
  MissingObservation
} derive(Debug, Eq)
