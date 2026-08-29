///|
/// Startup-only bounds for one canonical campaign record and every referenced
/// evidence artifact.
pub struct ReleaseCampaignLimits {
  priv max_canonical_bytes : Int
  priv max_locator_bytes : Int
  priv max_artifact_bytes : Int64
}

///|
pub fn ReleaseCampaignLimits::new(
  max_canonical_bytes~ : Int,
  max_locator_bytes~ : Int,
  max_artifact_bytes~ : Int64,
) -> ReleaseCampaignLimits raise ReleaseEvidenceError {
  if max_canonical_bytes < 1024 ||
    max_canonical_bytes > 1024 * 1024 ||
    max_locator_bytes <= 0 ||
    max_locator_bytes > 1024 ||
    max_artifact_bytes <= 0 {
    raise InvalidLimits
  }
  { max_canonical_bytes, max_locator_bytes, max_artifact_bytes }
}

///|
pub fn ReleaseCampaignLimits::max_canonical_bytes(
  self : ReleaseCampaignLimits,
) -> Int {
  self.max_canonical_bytes
}

///|
pub fn ReleaseCampaignLimits::max_locator_bytes(
  self : ReleaseCampaignLimits,
) -> Int {
  self.max_locator_bytes
}

///|
pub fn ReleaseCampaignLimits::max_artifact_bytes(
  self : ReleaseCampaignLimits,
) -> Int64 {
  self.max_artifact_bytes
}
