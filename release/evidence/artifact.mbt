///|
fn locator_component_is_forbidden(
  value : StringView,
  start : Int,
  end : Int,
) -> Bool {
  let length = end - start
  length == 0 ||
  (length == 1 && value[start] == '.') ||
  (length == 2 && value[start] == '.' && value[start + 1] == '.')
}

///|
fn validate_release_artifact_locator(
  value : StringView,
  limits : ReleaseCampaignLimits,
) -> Unit raise ReleaseEvidenceError {
  if value.length() == 0 ||
    value.length() > limits.max_locator_bytes() ||
    value[0] == '/' ||
    value[value.length() - 1] == '/' {
    raise InvalidArtifact
  }
  let mut component_start = 0
  for index, unit in value {
    let allowed = unit
      is ('a'..='z' | 'A'..='Z' | '0'..='9' | '.' | '_' | '-' | '/')
    if !allowed {
      raise InvalidArtifact
    }
    if unit == '/' {
      if locator_component_is_forbidden(value, component_start, index) {
        raise InvalidArtifact
      }
      component_start = index + 1
    }
  }
  if locator_component_is_forbidden(value, component_start, value.length()) {
    raise InvalidArtifact
  }
}

///|
/// One immutable evidence object in a host-provided release bundle.
///
/// The locator is strict-relative inventory, not ambient filesystem authority.
pub struct ReleaseEvidenceArtifact {
  priv locator : String
  priv byte_length : Int64
  priv digest : ReleaseArtifactDigest
} derive(Debug, Eq)

///|
pub fn ReleaseEvidenceArtifact::new(
  locator : StringView,
  byte_length~ : Int64,
  digest~ : ReleaseArtifactDigest,
  limits~ : ReleaseCampaignLimits,
) -> ReleaseEvidenceArtifact raise ReleaseEvidenceError {
  validate_release_artifact_locator(locator, limits)
  if byte_length <= 0 || byte_length > limits.max_artifact_bytes() {
    raise InvalidArtifact
  }
  { locator: locator.to_owned(), byte_length, digest }
}

///|
pub fn ReleaseEvidenceArtifact::locator(
  self : ReleaseEvidenceArtifact,
) -> String {
  self.locator
}

///|
pub fn ReleaseEvidenceArtifact::byte_length(
  self : ReleaseEvidenceArtifact,
) -> Int64 {
  self.byte_length
}

///|
pub fn ReleaseEvidenceArtifact::digest(
  self : ReleaseEvidenceArtifact,
) -> ReleaseArtifactDigest {
  self.digest
}

///|
fn ReleaseEvidenceArtifact::validate_against(
  self : ReleaseEvidenceArtifact,
  limits : ReleaseCampaignLimits,
) -> Unit raise ReleaseEvidenceError {
  validate_release_artifact_locator(self.locator, limits)
  if self.byte_length <= 0 || self.byte_length > limits.max_artifact_bytes() {
    raise InvalidArtifact
  }
}
