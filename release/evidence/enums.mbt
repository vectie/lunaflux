///|
/// Truthful state of one evidence or reviewer obligation.
pub(all) enum ReleaseEvidenceStatus {
  Pending
  Passed
  Failed
  NotRun
} derive(Debug, Eq)

///|
/// Exact release obligations owned by the LunaFlux campaign record.
///
/// External-adapter roles are intentionally product-neutral. A deployment
/// system may own an adapter, but this package has no adapter or credential
/// authority.
pub(all) enum ReleaseEvidenceRole {
  PhysicalTarget
  ReferenceCorrectness
  NativeLeak
  StructuralBoundary
  Performance
  SecurityScan
  OciImage
  Sbom
  BuildProvenance
  LicenseInventory
  KernelManifest
  RuntimeBuild
  ExternalAdapterBuild
  DirectPath
  ExternalAdapterPath
  OperationalLifecycle
} derive(Debug, Eq)

///|
/// Exact human review roles required by the Phase 9 release gate.
pub(all) enum ReleaseReviewerRole {
  SecurityReviewer
  OperationsReviewer
  PerformanceReviewer
} derive(Debug, Eq)

///|
fn evidence_status_tag(status : ReleaseEvidenceStatus) -> Int {
  match status {
    Pending => 1
    Passed => 2
    Failed => 3
    NotRun => 4
  }
}

///|
fn evidence_status_from_tag(
  tag : Int,
) -> ReleaseEvidenceStatus raise ReleaseEvidenceError {
  match tag {
    1 => Pending
    2 => Passed
    3 => Failed
    4 => NotRun
    _ => raise InvalidCanonicalRecord
  }
}

///|
fn evidence_role_tag(role : ReleaseEvidenceRole) -> Int {
  match role {
    PhysicalTarget => 1
    ReferenceCorrectness => 2
    NativeLeak => 3
    StructuralBoundary => 4
    Performance => 5
    SecurityScan => 6
    OciImage => 7
    Sbom => 8
    BuildProvenance => 9
    LicenseInventory => 10
    KernelManifest => 11
    RuntimeBuild => 12
    ExternalAdapterBuild => 13
    DirectPath => 14
    ExternalAdapterPath => 15
    OperationalLifecycle => 16
  }
}

///|
fn evidence_role_from_tag(
  tag : Int,
) -> ReleaseEvidenceRole raise ReleaseEvidenceError {
  match tag {
    1 => PhysicalTarget
    2 => ReferenceCorrectness
    3 => NativeLeak
    4 => StructuralBoundary
    5 => Performance
    6 => SecurityScan
    7 => OciImage
    8 => Sbom
    9 => BuildProvenance
    10 => LicenseInventory
    11 => KernelManifest
    12 => RuntimeBuild
    13 => ExternalAdapterBuild
    14 => DirectPath
    15 => ExternalAdapterPath
    16 => OperationalLifecycle
    _ => raise InvalidCanonicalRecord
  }
}

///|
fn all_evidence_roles() -> Array[ReleaseEvidenceRole] {
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
fn reviewer_role_tag(role : ReleaseReviewerRole) -> Int {
  match role {
    SecurityReviewer => 1
    OperationsReviewer => 2
    PerformanceReviewer => 3
  }
}

///|
fn reviewer_role_from_tag(
  tag : Int,
) -> ReleaseReviewerRole raise ReleaseEvidenceError {
  match tag {
    1 => SecurityReviewer
    2 => OperationsReviewer
    3 => PerformanceReviewer
    _ => raise InvalidCanonicalRecord
  }
}

///|
fn all_reviewer_roles() -> Array[ReleaseReviewerRole] {
  [SecurityReviewer, OperationsReviewer, PerformanceReviewer]
}

///|
fn is_terminal_status(status : ReleaseEvidenceStatus) -> Bool {
  status is (Passed | Failed)
}
