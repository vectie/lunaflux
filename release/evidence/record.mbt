///|
fn release_campaign_digest_of(
  canonical : Bytes,
) -> ReleaseCampaignDigest raise ReleaseEvidenceError {
  ReleaseCampaignDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
}

///|
/// Construct a canonical v1 campaign declaration. Evidence and reviewer arrays
/// must contain every fixed role exactly once in stable role order.
pub fn ReleaseCampaignRecord::new(
  campaign_id~ : UInt64,
  subject~ : ReleaseSubjectDigest,
  image~ : ReleaseImageDigest,
  physical_target~ : ReleaseTargetDigest,
  evidence~ : ArrayView[ReleaseEvidenceEntry],
  reviewers~ : ArrayView[ReleaseReviewerEntry],
  limits~ : ReleaseCampaignLimits,
) -> ReleaseCampaignRecord raise ReleaseEvidenceError {
  if campaign_id == 0UL {
    raise InvalidCampaignIdentity
  }
  validate_release_campaign_entries(evidence, reviewers)
  let evidence_copy = [ for entry in evidence => entry ]
  let reviewer_copy = [ for reviewer in reviewers => reviewer ]
  let canonical = encode_release_campaign(
    campaign_id, subject, image, physical_target, evidence_copy, reviewer_copy, limits,
  )
  let digest = release_campaign_digest_of(canonical)
  {
    campaign_id,
    subject,
    image,
    physical_target,
    evidence: ReadOnlyArray::from_array(evidence_copy),
    reviewers: ReadOnlyArray::from_array(reviewer_copy),
    canonical,
    digest,
  }
}
