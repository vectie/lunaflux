///|
fn release_evidence_is_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}

///|
fn release_evidence_is_nonzero_sha256(value : StringView) -> Bool {
  if !release_evidence_is_sha256(value) {
    return false
  }
  for unit in value {
    if unit != '0' {
      return true
    }
  }
  false
}

///|
pub struct ReleaseCampaignDigest(String) derive(Debug, Eq, Hash)

///|
pub fn ReleaseCampaignDigest::from_sha256(
  value : StringView,
) -> ReleaseCampaignDigest raise ReleaseEvidenceError {
  if !release_evidence_is_nonzero_sha256(value) {
    raise InvalidDigest
  }
  ReleaseCampaignDigest(value.to_owned())
}

///|
pub fn ReleaseCampaignDigest::as_hex(self : ReleaseCampaignDigest) -> String {
  self.0
}

///|
pub struct ReleaseSubjectDigest(String) derive(Debug, Eq, Hash)

///|
pub fn ReleaseSubjectDigest::from_sha256(
  value : StringView,
) -> ReleaseSubjectDigest raise ReleaseEvidenceError {
  if !release_evidence_is_nonzero_sha256(value) {
    raise InvalidDigest
  }
  ReleaseSubjectDigest(value.to_owned())
}

///|
pub fn ReleaseSubjectDigest::as_hex(self : ReleaseSubjectDigest) -> String {
  self.0
}

///|
pub struct ReleaseImageDigest(String) derive(Debug, Eq, Hash)

///|
pub fn ReleaseImageDigest::from_sha256(
  value : StringView,
) -> ReleaseImageDigest raise ReleaseEvidenceError {
  if !release_evidence_is_nonzero_sha256(value) {
    raise InvalidDigest
  }
  ReleaseImageDigest(value.to_owned())
}

///|
pub fn ReleaseImageDigest::as_hex(self : ReleaseImageDigest) -> String {
  self.0
}

///|
pub struct ReleaseTargetDigest(String) derive(Debug, Eq, Hash)

///|
pub fn ReleaseTargetDigest::from_sha256(
  value : StringView,
) -> ReleaseTargetDigest raise ReleaseEvidenceError {
  if !release_evidence_is_nonzero_sha256(value) {
    raise InvalidDigest
  }
  ReleaseTargetDigest(value.to_owned())
}

///|
pub fn ReleaseTargetDigest::as_hex(self : ReleaseTargetDigest) -> String {
  self.0
}

///|
pub struct ReleaseArtifactDigest(String) derive(Debug, Eq, Hash)

///|
pub fn ReleaseArtifactDigest::from_sha256(
  value : StringView,
) -> ReleaseArtifactDigest raise ReleaseEvidenceError {
  if !release_evidence_is_nonzero_sha256(value) {
    raise InvalidDigest
  }
  ReleaseArtifactDigest(value.to_owned())
}

///|
pub fn ReleaseArtifactDigest::as_hex(self : ReleaseArtifactDigest) -> String {
  self.0
}

///|
pub struct ReleaseReviewerIdentityDigest(String) derive(Debug, Eq, Hash)

///|
pub fn ReleaseReviewerIdentityDigest::from_sha256(
  value : StringView,
) -> ReleaseReviewerIdentityDigest raise ReleaseEvidenceError {
  if !release_evidence_is_nonzero_sha256(value) {
    raise InvalidDigest
  }
  ReleaseReviewerIdentityDigest(value.to_owned())
}

///|
pub fn ReleaseReviewerIdentityDigest::as_hex(
  self : ReleaseReviewerIdentityDigest,
) -> String {
  self.0
}
