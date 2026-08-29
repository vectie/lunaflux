///|
priv struct ReleaseCampaignEncoder {
  bytes : Array[Byte]
  maximum : Int
}

///|
fn ReleaseCampaignEncoder::append_byte(
  self : ReleaseCampaignEncoder,
  value : Byte,
) -> Unit raise ReleaseEvidenceError {
  if self.bytes.length() >= self.maximum {
    raise InvalidCanonicalRecord
  }
  self.bytes.push(value)
}

///|
fn ReleaseCampaignEncoder::append_u64(
  self : ReleaseCampaignEncoder,
  value : UInt64,
) -> Unit raise ReleaseEvidenceError {
  for byte in value.to_le_bytes() {
    self.append_byte(byte)
  }
}

///|
fn ReleaseCampaignEncoder::append_int(
  self : ReleaseCampaignEncoder,
  value : Int,
) -> Unit raise ReleaseEvidenceError {
  self.append_u64(Int64::from_int(value).reinterpret_as_uint64())
}

///|
fn ReleaseCampaignEncoder::append_i64(
  self : ReleaseCampaignEncoder,
  value : Int64,
) -> Unit raise ReleaseEvidenceError {
  self.append_u64(value.reinterpret_as_uint64())
}

///|
fn ReleaseCampaignEncoder::append_string(
  self : ReleaseCampaignEncoder,
  value : StringView,
) -> Unit raise ReleaseEvidenceError {
  if value.length() > self.maximum {
    raise InvalidCanonicalRecord
  }
  let encoded = @utf8.encode(value)
  self.append_int(encoded.length())
  for byte in encoded {
    self.append_byte(byte)
  }
}

///|
fn ReleaseCampaignEncoder::append_artifact(
  self : ReleaseCampaignEncoder,
  artifact : ReleaseEvidenceArtifact?,
) -> Unit raise ReleaseEvidenceError {
  match artifact {
    None => self.append_int(0)
    Some(value) => {
      self.append_int(1)
      self.append_string(value.locator())
      self.append_i64(value.byte_length())
      self.append_string(value.digest().as_hex())
    }
  }
}

///|
fn ReleaseCampaignEncoder::append_evidence(
  self : ReleaseCampaignEncoder,
  entry : ReleaseEvidenceEntry,
) -> Unit raise ReleaseEvidenceError {
  self.append_int(evidence_role_tag(entry.role()))
  self.append_int(evidence_status_tag(entry.status()))
  self.append_artifact(entry.artifact())
  self.append_int(entry.workload_count())
  self.append_int(entry.losing_workload_count())
}

///|
fn ReleaseCampaignEncoder::append_reviewer(
  self : ReleaseCampaignEncoder,
  reviewer : ReleaseReviewerEntry,
) -> Unit raise ReleaseEvidenceError {
  self.append_int(reviewer_role_tag(reviewer.role()))
  self.append_int(evidence_status_tag(reviewer.status()))
  match reviewer.identity() {
    None => self.append_int(0)
    Some(identity) => {
      self.append_int(1)
      self.append_string(identity.as_hex())
    }
  }
  self.append_artifact(reviewer.attestation())
}

///|
fn validate_release_campaign_entries(
  evidence : ArrayView[ReleaseEvidenceEntry],
  reviewers : ArrayView[ReleaseReviewerEntry],
) -> Unit raise ReleaseEvidenceError {
  let evidence_roles = all_evidence_roles()
  if evidence.length() != evidence_roles.length() {
    raise InvalidEvidenceEntry
  }
  for index in 0..<evidence_roles.length() {
    if evidence[index].role() != evidence_roles[index] {
      raise InvalidEvidenceEntry
    }
  }
  let reviewer_roles = all_reviewer_roles()
  if reviewers.length() != reviewer_roles.length() {
    raise InvalidReviewerEntry
  }
  for index in 0..<reviewer_roles.length() {
    if reviewers[index].role() != reviewer_roles[index] {
      raise InvalidReviewerEntry
    }
  }
}

///|
fn validate_release_campaign_artifacts(
  evidence : ArrayView[ReleaseEvidenceEntry],
  reviewers : ArrayView[ReleaseReviewerEntry],
  limits : ReleaseCampaignLimits,
) -> Unit raise ReleaseEvidenceError {
  let locators : Array[String] = []
  let digests : Array[ReleaseArtifactDigest] = []
  for entry in evidence {
    if entry.artifact() is Some(artifact) {
      artifact.validate_against(limits)
      if locators.contains(artifact.locator()) ||
        digests.contains(artifact.digest()) {
        raise InvalidArtifact
      }
      locators.push(artifact.locator())
      digests.push(artifact.digest())
    }
  }
  for reviewer in reviewers {
    if reviewer.attestation() is Some(attestation) {
      attestation.validate_against(limits)
      if locators.contains(attestation.locator()) ||
        digests.contains(attestation.digest()) {
        raise InvalidArtifact
      }
      locators.push(attestation.locator())
      digests.push(attestation.digest())
    }
  }
}

///|
fn encode_release_campaign(
  campaign_id : UInt64,
  subject : ReleaseSubjectDigest,
  image : ReleaseImageDigest,
  physical_target : ReleaseTargetDigest,
  evidence : ArrayView[ReleaseEvidenceEntry],
  reviewers : ArrayView[ReleaseReviewerEntry],
  limits : ReleaseCampaignLimits,
) -> Bytes raise ReleaseEvidenceError {
  validate_release_campaign_entries(evidence, reviewers)
  validate_release_campaign_artifacts(evidence, reviewers, limits)
  let encoder : ReleaseCampaignEncoder = {
    bytes: [],
    maximum: limits.max_canonical_bytes(),
  }
  encoder.append_string(release_campaign_schema())
  encoder.append_int(1)
  encoder.append_u64(campaign_id)
  encoder.append_string(subject.as_hex())
  encoder.append_string(image.as_hex())
  encoder.append_string(physical_target.as_hex())
  encoder.append_int(evidence.length())
  for entry in evidence {
    encoder.append_evidence(entry)
  }
  encoder.append_int(reviewers.length())
  for reviewer in reviewers {
    encoder.append_reviewer(reviewer)
  }
  Bytes::from_array(encoder.bytes)
}
