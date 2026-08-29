///|
/// One exact release obligation. Only the performance role may carry workload
/// counts; its losing count is preserved even when the role passes.
pub struct ReleaseEvidenceEntry {
  priv role : ReleaseEvidenceRole
  priv status : ReleaseEvidenceStatus
  priv artifact : ReleaseEvidenceArtifact?
  priv workload_count : Int
  priv losing_workload_count : Int
} derive(Debug, Eq)

///|
pub fn ReleaseEvidenceEntry::new(
  role~ : ReleaseEvidenceRole,
  status~ : ReleaseEvidenceStatus,
  artifact~ : ReleaseEvidenceArtifact?,
  workload_count~ : Int,
  losing_workload_count~ : Int,
) -> ReleaseEvidenceEntry raise ReleaseEvidenceError {
  if is_terminal_status(status) != (artifact is Some(_)) {
    raise InvalidEvidenceEntry
  }
  match role {
    Performance =>
      if is_terminal_status(status) {
        if workload_count <= 0 ||
          losing_workload_count < 0 ||
          losing_workload_count > workload_count {
          raise InvalidEvidenceEntry
        }
      } else if workload_count != 0 || losing_workload_count != 0 {
        raise InvalidEvidenceEntry
      }
    _ =>
      if workload_count != 0 || losing_workload_count != 0 {
        raise InvalidEvidenceEntry
      }
  }
  { role, status, artifact, workload_count, losing_workload_count }
}

///|
pub fn ReleaseEvidenceEntry::role(
  self : ReleaseEvidenceEntry,
) -> ReleaseEvidenceRole {
  self.role
}

///|
pub fn ReleaseEvidenceEntry::status(
  self : ReleaseEvidenceEntry,
) -> ReleaseEvidenceStatus {
  self.status
}

///|
pub fn ReleaseEvidenceEntry::artifact(
  self : ReleaseEvidenceEntry,
) -> ReleaseEvidenceArtifact? {
  self.artifact
}

///|
pub fn ReleaseEvidenceEntry::workload_count(self : ReleaseEvidenceEntry) -> Int {
  self.workload_count
}

///|
pub fn ReleaseEvidenceEntry::losing_workload_count(
  self : ReleaseEvidenceEntry,
) -> Int {
  self.losing_workload_count
}

///|
/// One named review obligation. Terminal review states require both an
/// identity digest and a separately digest-bound attestation artifact.
pub struct ReleaseReviewerEntry {
  priv role : ReleaseReviewerRole
  priv status : ReleaseEvidenceStatus
  priv identity : ReleaseReviewerIdentityDigest?
  priv attestation : ReleaseEvidenceArtifact?
} derive(Debug, Eq)

///|
pub fn ReleaseReviewerEntry::new(
  role~ : ReleaseReviewerRole,
  status~ : ReleaseEvidenceStatus,
  identity~ : ReleaseReviewerIdentityDigest?,
  attestation~ : ReleaseEvidenceArtifact?,
) -> ReleaseReviewerEntry raise ReleaseEvidenceError {
  match status {
    Passed | Failed =>
      if !(identity is Some(_)) || !(attestation is Some(_)) {
        raise InvalidReviewerEntry
      }
    Pending => if attestation is Some(_) { raise InvalidReviewerEntry }
    NotRun =>
      if identity is Some(_) || attestation is Some(_) {
        raise InvalidReviewerEntry
      }
  }
  { role, status, identity, attestation }
}

///|
pub fn ReleaseReviewerEntry::role(
  self : ReleaseReviewerEntry,
) -> ReleaseReviewerRole {
  self.role
}

///|
pub fn ReleaseReviewerEntry::status(
  self : ReleaseReviewerEntry,
) -> ReleaseEvidenceStatus {
  self.status
}

///|
pub fn ReleaseReviewerEntry::identity(
  self : ReleaseReviewerEntry,
) -> ReleaseReviewerIdentityDigest? {
  self.identity
}

///|
pub fn ReleaseReviewerEntry::attestation(
  self : ReleaseReviewerEntry,
) -> ReleaseEvidenceArtifact? {
  self.attestation
}
