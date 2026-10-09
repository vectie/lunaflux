///|
pub fn AotCandidateCoverage::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn AotCandidateCoverage::requirements_digest_sha256(self : Self) -> String {
  self.requirements_digest_sha256
}

///|
pub fn AotCandidateCoverage::requirement_count(self : Self) -> Int {
  self.requirement_count
}

///|
pub fn AotCandidateCoverage::candidate_count(self : Self) -> Int {
  self.candidate_count
}

///|
pub fn AotCandidateCoverage::missing_ordinals(
  self : Self,
) -> ReadOnlyArray[Int] {
  self.missing_ordinals
}

///|
pub fn AotCandidateCoverage::non_bindable_ordinals(
  self : Self,
) -> ReadOnlyArray[Int] {
  self.non_bindable_ordinals
}

///|
pub fn AotCandidateCoverage::status(self : Self) -> AotCandidateCoverageStatus {
  self.status
}

///|
pub fn AotCandidateCoverage::digest(self : Self) -> AotCandidateCoverageDigest {
  self.digest
}

///|
pub fn AotCandidateCoverageDigest::as_hex(self : Self) -> String {
  self.hex
}

///|
/// Requires structural ordinal coverage only. This does not grant artifact or
/// execution authority.
pub fn AotCandidateCoverage::require_complete(
  self : Self,
) -> Unit raise AotCandidateCoverageError {
  if !self.missing_ordinals.is_empty() {
    raise IncompleteCoverage(self.missing_ordinals.length())
  }
}

///|
/// Requires every structurally complete candidate to claim manifest
/// bindability. The claims still require independent artifact admission and
/// qualification.
pub fn AotCandidateCoverage::require_all_claimed_bindable(
  self : Self,
) -> Unit raise AotCandidateCoverageError {
  self.require_complete()
  if !self.non_bindable_ordinals.is_empty() {
    raise NonBindableCoverage(self.non_bindable_ordinals.length())
  }
}
