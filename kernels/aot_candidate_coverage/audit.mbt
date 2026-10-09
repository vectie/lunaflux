///|
/// Structurally audits lowered candidate coverage for one exact requirement
/// set. This function grants no artifact, qualification, or execution
/// authority.
pub fn audit(
  source_identity~ : @spec.ModelIdentity,
  requirements_digest_sha256~ : String,
  requirement_count~ : Int,
  candidates~ : ArrayView[AotCandidateEvidenceInput],
  limits~ : AotCandidateCoverageLimits,
) -> AotCandidateCoverage raise AotCandidateCoverageError {
  if !valid_sha256(requirements_digest_sha256) {
    raise InvalidDigest(RequirementsDigest)
  }
  if requirement_count <= 0 || requirement_count > limits.maximum_requirements {
    raise InvalidRequirementCount(requirement_count)
  }
  if candidates.length() > limits.maximum_candidates ||
    candidates.length() > requirement_count {
    raise CandidateCountExceedsLimit(candidates.length())
  }
  let ordered : FixedArray[AotCandidateEvidenceInput?] = FixedArray::make(
    requirement_count,
    None,
  )
  let owners = FixedArray::make(requirement_count, -1)
  for candidate_index, candidate in candidates {
    if candidate.source_identity != source_identity {
      raise CandidateIdentityMismatch(candidate_index)
    }
    if candidate.requirements_digest_sha256 != requirements_digest_sha256 {
      raise CandidateRequirementsDigestMismatch(candidate_index)
    }
    let ordinal = candidate.requirement_ordinal
    if ordinal < 0 || ordinal >= requirement_count {
      raise InvalidRequirementOrdinal(candidate=candidate_index, ordinal~)
    }
    if owners[ordinal] >= 0 {
      raise DuplicateRequirementOrdinal(
        first_candidate=owners[ordinal],
        second_candidate=candidate_index,
        ordinal~,
      )
    }
    for previous in 0..<candidate_index {
      if candidates[previous].recipe_digest_sha256 ==
        candidate.recipe_digest_sha256 {
        raise DuplicateRecipeDigest(
          first_candidate=previous,
          second_candidate=candidate_index,
        )
      }
    }
    owners[ordinal] = candidate_index
    ordered[ordinal] = Some(candidate)
  }
  let missing : Array[Int] = []
  let non_bindable : Array[Int] = []
  for ordinal in 0..<requirement_count {
    match ordered[ordinal] {
      None => missing.push(ordinal)
      Some(candidate) if !candidate.manifest_bindable =>
        non_bindable.push(ordinal)
      Some(_) => ()
    }
  }
  let status = if !missing.is_empty() {
    Incomplete
  } else if !non_bindable.is_empty() {
    CompleteInert
  } else {
    CompleteWithBindableClaims
  }
  let digest = coverage_digest(
    source_identity,
    requirements_digest_sha256,
    requirement_count,
    ordered,
    limits.maximum_encoding_bytes,
  )
  {
    source_identity,
    requirements_digest_sha256,
    requirement_count,
    candidate_count: candidates.length(),
    missing_ordinals: ReadOnlyArray::from_array(missing),
    non_bindable_ordinals: ReadOnlyArray::from_array(non_bindable),
    status,
    digest,
  }
}
