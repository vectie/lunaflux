///|
/// Independently approved identities used to verify one admitted campaign.
/// Reviewer digests identify allowlisted people or review principals; this
/// package neither stores nor verifies their credentials.
pub struct ReleaseVerificationPolicy {
  priv subject : ReleaseSubjectDigest
  priv image : ReleaseImageDigest
  priv physical_target : ReleaseTargetDigest
  priv security_reviewer : ReleaseReviewerIdentityDigest
  priv operations_reviewer : ReleaseReviewerIdentityDigest
  priv performance_reviewer : ReleaseReviewerIdentityDigest
}

///|
pub fn ReleaseVerificationPolicy::new(
  subject~ : ReleaseSubjectDigest,
  image~ : ReleaseImageDigest,
  physical_target~ : ReleaseTargetDigest,
  security_reviewer~ : ReleaseReviewerIdentityDigest,
  operations_reviewer~ : ReleaseReviewerIdentityDigest,
  performance_reviewer~ : ReleaseReviewerIdentityDigest,
) -> ReleaseVerificationPolicy {
  {
    subject,
    image,
    physical_target,
    security_reviewer,
    operations_reviewer,
    performance_reviewer,
  }
}

///|
fn ReleaseVerificationPolicy::reviewer_identity(
  self : ReleaseVerificationPolicy,
  role : ReleaseReviewerRole,
) -> ReleaseReviewerIdentityDigest {
  match role {
    SecurityReviewer => self.security_reviewer
    OperationsReviewer => self.operations_reviewer
    PerformanceReviewer => self.performance_reviewer
  }
}

///|
/// Host-observed identity of one evidence artifact after reading or streaming
/// that artifact outside this authority-free package.
pub struct ReleaseEvidenceObservation {
  priv role : ReleaseEvidenceRole
  priv locator : String
  priv byte_length : Int64
  priv digest : ReleaseArtifactDigest
} derive(Debug, Eq)

///|
pub fn ReleaseEvidenceObservation::new(
  role~ : ReleaseEvidenceRole,
  locator : StringView,
  byte_length~ : Int64,
  digest~ : ReleaseArtifactDigest,
  limits~ : ReleaseCampaignLimits,
) -> ReleaseEvidenceObservation raise ReleaseEvidenceError {
  validate_release_artifact_locator(locator, limits)
  if byte_length <= 0 || byte_length > limits.max_artifact_bytes() {
    raise InvalidObservation
  }
  { role, locator: locator.to_owned(), byte_length, digest }
}

///|
/// Compute a host observation from already bounded bytes supplied by the
/// caller. File acquisition and trust remain outside this package.
pub fn ReleaseEvidenceObservation::from_payload(
  role~ : ReleaseEvidenceRole,
  locator : StringView,
  payload : Bytes,
  limits~ : ReleaseCampaignLimits,
) -> ReleaseEvidenceObservation raise ReleaseEvidenceError {
  validate_release_artifact_locator(locator, limits)
  if payload.length() <= 0 ||
    Int64::from_int(payload.length()) > limits.max_artifact_bytes() {
    raise InvalidObservation
  }
  ReleaseEvidenceObservation::new(
    role~,
    locator,
    byte_length=Int64::from_int(payload.length()),
    digest=ReleaseArtifactDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(payload)),
    ),
    limits~,
  )
}

///|
pub fn ReleaseEvidenceObservation::role(
  self : ReleaseEvidenceObservation,
) -> ReleaseEvidenceRole {
  self.role
}

///|
pub fn ReleaseEvidenceObservation::locator(
  self : ReleaseEvidenceObservation,
) -> String {
  self.locator
}

///|
/// Host-observed identity of one reviewer's bounded attestation artifact.
pub struct ReleaseReviewerObservation {
  priv role : ReleaseReviewerRole
  priv locator : String
  priv byte_length : Int64
  priv digest : ReleaseArtifactDigest
} derive(Debug, Eq)

///|
pub fn ReleaseReviewerObservation::new(
  role~ : ReleaseReviewerRole,
  locator : StringView,
  byte_length~ : Int64,
  digest~ : ReleaseArtifactDigest,
  limits~ : ReleaseCampaignLimits,
) -> ReleaseReviewerObservation raise ReleaseEvidenceError {
  validate_release_artifact_locator(locator, limits)
  if byte_length <= 0 || byte_length > limits.max_artifact_bytes() {
    raise InvalidObservation
  }
  { role, locator: locator.to_owned(), byte_length, digest }
}

///|
pub fn ReleaseReviewerObservation::from_payload(
  role~ : ReleaseReviewerRole,
  locator : StringView,
  payload : Bytes,
  limits~ : ReleaseCampaignLimits,
) -> ReleaseReviewerObservation raise ReleaseEvidenceError {
  validate_release_artifact_locator(locator, limits)
  if payload.length() <= 0 ||
    Int64::from_int(payload.length()) > limits.max_artifact_bytes() {
    raise InvalidObservation
  }
  ReleaseReviewerObservation::new(
    role~,
    locator,
    byte_length=Int64::from_int(payload.length()),
    digest=ReleaseArtifactDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(payload)),
    ),
    limits~,
  )
}

///|
pub fn ReleaseReviewerObservation::role(
  self : ReleaseReviewerObservation,
) -> ReleaseReviewerRole {
  self.role
}

///|
pub fn ReleaseReviewerObservation::locator(
  self : ReleaseReviewerObservation,
) -> String {
  self.locator
}

///|
fn verify_observed_artifact(
  artifact : ReleaseEvidenceArtifact,
  locator : String,
  byte_length : Int64,
  digest : ReleaseArtifactDigest,
) -> Unit raise ReleaseEvidenceError {
  if artifact.locator() != locator ||
    artifact.byte_length() != byte_length ||
    artifact.digest() != digest {
    raise InvalidObservation
  }
}

///|
fn verify_evidence_observations(
  record : ReleaseCampaignRecord,
  observations : ArrayView[ReleaseEvidenceObservation],
) -> Unit raise ReleaseEvidenceError {
  let mut terminal_count = 0
  for entry in record.evidence {
    if is_terminal_status(entry.status()) {
      terminal_count += 1
    }
  }
  if observations.length() < terminal_count {
    raise MissingObservation
  }
  if observations.length() > terminal_count {
    raise InvalidObservation
  }
  for entry in record.evidence {
    let mut match_count = 0
    let mut matched_locator : String? = None
    let mut matched_length = 0L
    let mut matched_digest : ReleaseArtifactDigest? = None
    for observation in observations {
      if observation.role == entry.role() {
        match_count += 1
        matched_locator = Some(observation.locator)
        matched_length = observation.byte_length
        matched_digest = Some(observation.digest)
      }
    }
    if is_terminal_status(entry.status()) {
      guard entry.artifact() is Some(artifact) else {
        raise InvalidEvidenceEntry
      }
      guard match_count == 1 &&
        matched_locator is Some(locator) &&
        matched_digest is Some(digest) else {
        raise MissingObservation
      }
      verify_observed_artifact(artifact, locator, matched_length, digest)
    } else if match_count != 0 {
      raise InvalidObservation
    }
  }
}

///|
fn verify_reviewer_observations(
  record : ReleaseCampaignRecord,
  policy : ReleaseVerificationPolicy,
  observations : ArrayView[ReleaseReviewerObservation],
) -> Unit raise ReleaseEvidenceError {
  let mut terminal_count = 0
  for reviewer in record.reviewers {
    if is_terminal_status(reviewer.status()) {
      terminal_count += 1
    }
  }
  if observations.length() < terminal_count {
    raise MissingObservation
  }
  if observations.length() > terminal_count {
    raise InvalidObservation
  }
  for reviewer in record.reviewers {
    let mut match_count = 0
    let mut matched_locator : String? = None
    let mut matched_length = 0L
    let mut matched_digest : ReleaseArtifactDigest? = None
    for observation in observations {
      if observation.role == reviewer.role() {
        match_count += 1
        matched_locator = Some(observation.locator)
        matched_length = observation.byte_length
        matched_digest = Some(observation.digest)
      }
    }
    match reviewer.status() {
      Passed | Failed => {
        guard reviewer.identity() is Some(identity) &&
          identity == policy.reviewer_identity(reviewer.role()) else {
          raise VerificationPolicyMismatch
        }
        guard reviewer.attestation() is Some(attestation) else {
          raise InvalidReviewerEntry
        }
        guard match_count == 1 &&
          matched_locator is Some(locator) &&
          matched_digest is Some(digest) else {
          raise MissingObservation
        }
        verify_observed_artifact(attestation, locator, matched_length, digest)
      }
      Pending => {
        if reviewer.identity() is Some(identity) &&
          identity != policy.reviewer_identity(reviewer.role()) {
          raise VerificationPolicyMismatch
        }
        if match_count != 0 {
          raise InvalidObservation
        }
      }
      NotRun => if match_count != 0 { raise InvalidObservation }
    }
  }
}

///|
/// Opaque result of complete host verification. It has no public constructor;
/// a declared state alone cannot fabricate this capability.
pub struct VerifiedReleaseCampaign {
  priv campaign_digest : ReleaseCampaignDigest
  priv status : ReleaseEvidenceStatus
} derive(Debug, Eq)

///|
pub fn VerifiedReleaseCampaign::campaign_digest(
  self : VerifiedReleaseCampaign,
) -> ReleaseCampaignDigest {
  self.campaign_digest
}

///|
pub fn VerifiedReleaseCampaign::status(
  self : VerifiedReleaseCampaign,
) -> ReleaseEvidenceStatus {
  self.status
}

///|
pub fn VerifiedReleaseCampaign::is_accepted(
  self : VerifiedReleaseCampaign,
) -> Bool {
  self.status is Passed
}

///|
/// Verify every terminal artifact observation and every terminal reviewer's
/// independently allowlisted identity, then return the four-state campaign
/// result. Missing observations raise; they can never produce an accepted
/// verification capability.
pub fn verify_release_campaign(
  record : ReleaseCampaignRecord,
  policy~ : ReleaseVerificationPolicy,
  evidence_observations~ : ArrayView[ReleaseEvidenceObservation],
  reviewer_observations~ : ArrayView[ReleaseReviewerObservation],
) -> VerifiedReleaseCampaign raise ReleaseEvidenceError {
  if record.subject() != policy.subject ||
    record.image() != policy.image ||
    record.physical_target() != policy.physical_target {
    raise VerificationPolicyMismatch
  }
  verify_evidence_observations(record, evidence_observations)
  verify_reviewer_observations(record, policy, reviewer_observations)
  { campaign_digest: record.digest(), status: record.declared_status() }
}
