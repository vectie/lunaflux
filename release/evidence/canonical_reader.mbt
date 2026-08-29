///|
priv struct ReleaseCampaignDecoder {
  bytes : Bytes
  limits : ReleaseCampaignLimits
  mut cursor : Int
}

///|
fn ReleaseCampaignDecoder::read_u64(
  self : ReleaseCampaignDecoder,
) -> UInt64 raise ReleaseEvidenceError {
  if self.cursor < 0 || self.cursor > self.bytes.length() - 8 {
    raise InvalidCanonicalRecord
  }
  let mut value = 0UL
  for index in 0..<8 {
    value = value | (self.bytes[self.cursor + index].to_uint64() << (index * 8))
  }
  self.cursor += 8
  value
}

///|
fn ReleaseCampaignDecoder::read_i64(
  self : ReleaseCampaignDecoder,
) -> Int64 raise ReleaseEvidenceError {
  self.read_u64().reinterpret_as_int64()
}

///|
fn ReleaseCampaignDecoder::read_int(
  self : ReleaseCampaignDecoder,
) -> Int raise ReleaseEvidenceError {
  let value = self.read_i64()
  let decoded = value.to_int()
  if Int64::from_int(decoded) != value {
    raise InvalidCanonicalRecord
  }
  decoded
}

///|
fn ReleaseCampaignDecoder::read_string(
  self : ReleaseCampaignDecoder,
) -> String raise ReleaseEvidenceError {
  let length = self.read_int()
  if length < 0 ||
    length > self.limits.max_canonical_bytes() ||
    self.cursor < 0 ||
    self.cursor > self.bytes.length() - length {
    raise InvalidCanonicalRecord
  }
  let value = @utf8.decode(
    Bytes::makei(length, index => self.bytes[self.cursor + index]),
  ) catch {
    _ => raise InvalidCanonicalRecord
  }
  self.cursor += length
  value
}

///|
fn ReleaseCampaignDecoder::read_artifact(
  self : ReleaseCampaignDecoder,
) -> ReleaseEvidenceArtifact? raise ReleaseEvidenceError {
  match self.read_int() {
    0 => None
    1 => {
      let locator = self.read_string()
      let byte_length = self.read_i64()
      let digest = ReleaseArtifactDigest::from_sha256(self.read_string())
      Some(
        ReleaseEvidenceArtifact::new(
          locator,
          byte_length~,
          digest~,
          limits=self.limits,
        ),
      )
    }
    _ => raise InvalidCanonicalRecord
  }
}

///|
fn ReleaseCampaignDecoder::read_evidence(
  self : ReleaseCampaignDecoder,
  expected : ReleaseEvidenceRole,
) -> ReleaseEvidenceEntry raise ReleaseEvidenceError {
  let role = evidence_role_from_tag(self.read_int())
  if role != expected {
    raise InvalidCanonicalRecord
  }
  ReleaseEvidenceEntry::new(
    role~,
    status=evidence_status_from_tag(self.read_int()),
    artifact=self.read_artifact(),
    workload_count=self.read_int(),
    losing_workload_count=self.read_int(),
  )
}

///|
fn ReleaseCampaignDecoder::read_reviewer(
  self : ReleaseCampaignDecoder,
  expected : ReleaseReviewerRole,
) -> ReleaseReviewerEntry raise ReleaseEvidenceError {
  let role = reviewer_role_from_tag(self.read_int())
  if role != expected {
    raise InvalidCanonicalRecord
  }
  let status = evidence_status_from_tag(self.read_int())
  let identity = match self.read_int() {
    0 => None
    1 => Some(ReleaseReviewerIdentityDigest::from_sha256(self.read_string()))
    _ => raise InvalidCanonicalRecord
  }
  ReleaseReviewerEntry::new(
    role~,
    status~,
    identity~,
    attestation=self.read_artifact(),
  )
}

///|
fn decode_release_campaign(
  canonical : Bytes,
  limits : ReleaseCampaignLimits,
) -> ReleaseCampaignRecord raise ReleaseEvidenceError {
  if canonical.length() == 0 ||
    canonical.length() > limits.max_canonical_bytes() {
    raise InvalidCanonicalRecord
  }
  let decoder : ReleaseCampaignDecoder = { bytes: canonical, limits, cursor: 0 }
  if decoder.read_string() != release_campaign_schema() ||
    decoder.read_int() != 1 {
    raise InvalidCanonicalRecord
  }
  let campaign_id = decoder.read_u64()
  let subject = ReleaseSubjectDigest::from_sha256(decoder.read_string())
  let image = ReleaseImageDigest::from_sha256(decoder.read_string())
  let physical_target = ReleaseTargetDigest::from_sha256(decoder.read_string())
  let evidence_roles = all_evidence_roles()
  if decoder.read_int() != evidence_roles.length() {
    raise InvalidCanonicalRecord
  }
  let evidence = [ for role in evidence_roles => decoder.read_evidence(role) ]
  let reviewer_roles = all_reviewer_roles()
  if decoder.read_int() != reviewer_roles.length() {
    raise InvalidCanonicalRecord
  }
  let reviewers = [ for role in reviewer_roles => decoder.read_reviewer(role) ]
  if decoder.cursor != canonical.length() {
    raise InvalidCanonicalRecord
  }
  let record = ReleaseCampaignRecord::new(
    campaign_id~,
    subject~,
    image~,
    physical_target~,
    evidence~,
    reviewers~,
    limits~,
  )
  if record.canonical_bytes() != canonical {
    raise InvalidCanonicalRecord
  }
  record
}

///|
/// Admit exact canonical campaign bytes under an independently supplied digest.
/// This opens no filesystem, device, network, or reviewer credential authority.
pub fn admit_release_campaign(
  canonical : Bytes,
  expected_digest~ : ReleaseCampaignDigest,
  limits~ : ReleaseCampaignLimits,
) -> ReleaseCampaignRecord raise ReleaseEvidenceError {
  if canonical.length() == 0 ||
    canonical.length() > limits.max_canonical_bytes() {
    raise InvalidCanonicalRecord
  }
  if release_campaign_digest_of(canonical) != expected_digest {
    raise CampaignDigestMismatch
  }
  decode_release_campaign(canonical, limits)
}
