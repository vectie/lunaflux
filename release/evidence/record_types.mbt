///|
fn release_campaign_schema() -> String {
  "lunaflux.release-campaign.v1"
}

///|
/// Digest-bound, inert campaign declaration. Admission proves canonical
/// structure and exact identities; it does not interpret evidence payloads or
/// verify reviewer credentials.
pub struct ReleaseCampaignRecord {
  priv campaign_id : UInt64
  priv subject : ReleaseSubjectDigest
  priv image : ReleaseImageDigest
  priv physical_target : ReleaseTargetDigest
  priv evidence : ReadOnlyArray[ReleaseEvidenceEntry]
  priv reviewers : ReadOnlyArray[ReleaseReviewerEntry]
  priv canonical : Bytes
  priv digest : ReleaseCampaignDigest
} derive(Debug, Eq)

///|
pub fn ReleaseCampaignRecord::campaign_id(
  self : ReleaseCampaignRecord,
) -> UInt64 {
  self.campaign_id
}

///|
pub fn ReleaseCampaignRecord::subject(
  self : ReleaseCampaignRecord,
) -> ReleaseSubjectDigest {
  self.subject
}

///|
pub fn ReleaseCampaignRecord::image(
  self : ReleaseCampaignRecord,
) -> ReleaseImageDigest {
  self.image
}

///|
pub fn ReleaseCampaignRecord::physical_target(
  self : ReleaseCampaignRecord,
) -> ReleaseTargetDigest {
  self.physical_target
}

///|
pub fn ReleaseCampaignRecord::evidence_count(
  self : ReleaseCampaignRecord,
) -> Int {
  self.evidence.length()
}

///|
pub fn ReleaseCampaignRecord::evidence_at(
  self : ReleaseCampaignRecord,
  index : Int,
) -> ReleaseEvidenceEntry? {
  self.evidence.get(index)
}

///|
pub fn ReleaseCampaignRecord::reviewer_count(
  self : ReleaseCampaignRecord,
) -> Int {
  self.reviewers.length()
}

///|
pub fn ReleaseCampaignRecord::reviewer_at(
  self : ReleaseCampaignRecord,
  index : Int,
) -> ReleaseReviewerEntry? {
  self.reviewers.get(index)
}

///|
pub fn ReleaseCampaignRecord::canonical_bytes(
  self : ReleaseCampaignRecord,
) -> Bytes {
  self.canonical
}

///|
pub fn ReleaseCampaignRecord::digest(
  self : ReleaseCampaignRecord,
) -> ReleaseCampaignDigest {
  self.digest
}

///|
/// Aggregate of declared states only. This is deliberately not a release
/// verdict; use `verify_release_campaign` for host-observed evidence.
pub fn ReleaseCampaignRecord::declared_status(
  self : ReleaseCampaignRecord,
) -> ReleaseEvidenceStatus {
  let mut saw_pending = false
  let mut saw_not_run = false
  for entry in self.evidence {
    match entry.status() {
      Failed => return Failed
      Pending => saw_pending = true
      NotRun => saw_not_run = true
      Passed => ()
    }
  }
  for reviewer in self.reviewers {
    match reviewer.status() {
      Failed => return Failed
      Pending => saw_pending = true
      NotRun => saw_not_run = true
      Passed => ()
    }
  }
  if saw_pending {
    Pending
  } else if saw_not_run {
    NotRun
  } else {
    Passed
  }
}
