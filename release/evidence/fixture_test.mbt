///|
fn release_test_limits() -> @evidence.ReleaseCampaignLimits raise {
  @evidence.ReleaseCampaignLimits::new(
    max_canonical_bytes=65536,
    max_locator_bytes=128,
    max_artifact_bytes=1048576L,
  )
}

///|
fn release_test_evidence_payload(index : Int) -> Bytes {
  Bytes::from_array([b'E', (index + 1).to_byte()])
}

///|
fn release_test_reviewer_payload(index : Int) -> Bytes {
  Bytes::from_array([b'R', (index + 1).to_byte()])
}

///|
fn release_test_artifact(
  locator : String,
  payload : Bytes,
) -> @evidence.ReleaseEvidenceArtifact raise {
  @evidence.ReleaseEvidenceArtifact::new(
    locator,
    byte_length=Int64::from_int(payload.length()),
    digest=@evidence.ReleaseArtifactDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(payload)),
    ),
    limits=release_test_limits(),
  )
}

///|
fn release_test_subject() -> @evidence.ReleaseSubjectDigest raise {
  @evidence.ReleaseSubjectDigest::from_sha256(
    "1111111111111111111111111111111111111111111111111111111111111111",
  )
}

///|
fn release_test_image() -> @evidence.ReleaseImageDigest raise {
  @evidence.ReleaseImageDigest::from_sha256(
    "2222222222222222222222222222222222222222222222222222222222222222",
  )
}

///|
fn release_test_target() -> @evidence.ReleaseTargetDigest raise {
  @evidence.ReleaseTargetDigest::from_sha256(
    "3333333333333333333333333333333333333333333333333333333333333333",
  )
}

///|
fn release_test_roles() -> Array[@evidence.ReleaseEvidenceRole] {
  [
    PhysicalTarget,
    ReferenceCorrectness,
    NativeLeak,
    StructuralBoundary,
    Performance,
    SecurityScan,
    OciImage,
    Sbom,
    BuildProvenance,
    LicenseInventory,
    KernelManifest,
    RuntimeBuild,
    ExternalAdapterBuild,
    DirectPath,
    ExternalAdapterPath,
    OperationalLifecycle,
  ]
}

///|
fn release_test_reviewer_roles() -> Array[@evidence.ReleaseReviewerRole] {
  [SecurityReviewer, OperationsReviewer, PerformanceReviewer]
}

///|
fn release_test_reviewer_identity(
  role : @evidence.ReleaseReviewerRole,
) -> @evidence.ReleaseReviewerIdentityDigest raise {
  @evidence.ReleaseReviewerIdentityDigest::from_sha256(
    match role {
      SecurityReviewer =>
        "4444444444444444444444444444444444444444444444444444444444444444"
      OperationsReviewer =>
        "5555555555555555555555555555555555555555555555555555555555555555"
      PerformanceReviewer =>
        "6666666666666666666666666666666666666666666666666666666666666666"
    },
  )
}

///|
fn release_test_is_terminal(status : @evidence.ReleaseEvidenceStatus) -> Bool {
  status is (Passed | Failed)
}

///|
fn release_test_entries(
  status : @evidence.ReleaseEvidenceStatus,
) -> Array[@evidence.ReleaseEvidenceEntry] raise {
  let entries : Array[@evidence.ReleaseEvidenceEntry] = []
  for index, role in release_test_roles() {
    let artifact = if release_test_is_terminal(status) {
      Some(
        release_test_artifact(
          "evidence/\{index}.bin",
          release_test_evidence_payload(index),
        ),
      )
    } else {
      None
    }
    entries.push(
      @evidence.ReleaseEvidenceEntry::new(
        role~,
        status~,
        artifact~,
        workload_count=if role == Performance &&
          release_test_is_terminal(status) {
          9
        } else {
          0
        },
        losing_workload_count=if role == Performance &&
          release_test_is_terminal(status) {
          3
        } else {
          0
        },
      ),
    )
  }
  entries
}

///|
fn release_test_reviewers(
  status : @evidence.ReleaseEvidenceStatus,
) -> Array[@evidence.ReleaseReviewerEntry] raise {
  let reviewers : Array[@evidence.ReleaseReviewerEntry] = []
  for index, role in release_test_reviewer_roles() {
    let identity = match status {
      Passed | Failed | Pending => Some(release_test_reviewer_identity(role))
      NotRun => None
    }
    let attestation = if release_test_is_terminal(status) {
      Some(
        release_test_artifact(
          "review/\{index}.bin",
          release_test_reviewer_payload(index),
        ),
      )
    } else {
      None
    }
    reviewers.push(
      @evidence.ReleaseReviewerEntry::new(
        role~,
        status~,
        identity~,
        attestation~,
      ),
    )
  }
  reviewers
}

///|
fn release_test_record(
  status : @evidence.ReleaseEvidenceStatus,
) -> @evidence.ReleaseCampaignRecord raise {
  @evidence.ReleaseCampaignRecord::new(
    campaign_id=7UL,
    subject=release_test_subject(),
    image=release_test_image(),
    physical_target=release_test_target(),
    evidence=release_test_entries(status),
    reviewers=release_test_reviewers(status),
    limits=release_test_limits(),
  )
}

///|
fn release_test_policy() -> @evidence.ReleaseVerificationPolicy raise {
  @evidence.ReleaseVerificationPolicy::new(
    subject=release_test_subject(),
    image=release_test_image(),
    physical_target=release_test_target(),
    security_reviewer=release_test_reviewer_identity(SecurityReviewer),
    operations_reviewer=release_test_reviewer_identity(OperationsReviewer),
    performance_reviewer=release_test_reviewer_identity(PerformanceReviewer),
  )
}

///|
fn release_test_evidence_observations(
  status : @evidence.ReleaseEvidenceStatus,
) -> Array[@evidence.ReleaseEvidenceObservation] raise {
  if !release_test_is_terminal(status) {
    return []
  }
  let observations : Array[@evidence.ReleaseEvidenceObservation] = []
  for index, role in release_test_roles() {
    observations.push(
      @evidence.ReleaseEvidenceObservation::from_payload(
        role~,
        "evidence/\{index}.bin",
        release_test_evidence_payload(index),
        limits=release_test_limits(),
      ),
    )
  }
  observations
}

///|
fn release_test_reviewer_observations(
  status : @evidence.ReleaseEvidenceStatus,
) -> Array[@evidence.ReleaseReviewerObservation] raise {
  if !release_test_is_terminal(status) {
    return []
  }
  let observations : Array[@evidence.ReleaseReviewerObservation] = []
  for index, role in release_test_reviewer_roles() {
    observations.push(
      @evidence.ReleaseReviewerObservation::from_payload(
        role~,
        "review/\{index}.bin",
        release_test_reviewer_payload(index),
        limits=release_test_limits(),
      ),
    )
  }
  observations
}
