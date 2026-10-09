///|
pub(all) enum AotCandidateDigestField {
  RequirementsDigest
  SourceDigest
  RecipeDigest
} derive(Debug, Eq)

///|
/// Startup-only coverage failures. These errors never retain artifact bytes or
/// request data.
pub(all) suberror AotCandidateCoverageError {
  InvalidLimit
  InvalidRequirementCount(Int)
  InvalidDigest(AotCandidateDigestField)
  CandidateCountExceedsLimit(Int)
  CandidateIdentityMismatch(Int)
  CandidateRequirementsDigestMismatch(Int)
  InvalidRequirementOrdinal(candidate~ : Int, ordinal~ : Int)
  DuplicateRequirementOrdinal(
    first_candidate~ : Int,
    second_candidate~ : Int,
    ordinal~ : Int
  )
  DuplicateRecipeDigest(first_candidate~ : Int, second_candidate~ : Int)
  CoverageEncodingTooLarge
  IncompleteCoverage(Int)
  NonBindableCoverage(Int)
} derive(Debug, Eq)

///|
pub struct AotCandidateCoverageLimits {
  priv maximum_requirements : Int
  priv maximum_candidates : Int
  priv maximum_encoding_bytes : Int
}

///|
/// One already-lowered candidate claim. Construction validates only digest
/// syntax; `audit` authenticates identity, requirement digest, ordinal, and
/// uniqueness against the complete requirement set.
pub struct AotCandidateEvidenceInput {
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest_sha256 : String
  priv requirement_ordinal : Int
  priv source_digest_sha256 : String
  priv recipe_digest_sha256 : String
  priv manifest_bindable : Bool
}

///|
/// Coverage status is deliberately weaker than artifact or execution
/// admission. Even `CompleteWithBindableClaims` proves only caller-supplied
/// candidate metadata passed this structural audit.
pub(all) enum AotCandidateCoverageStatus {
  Incomplete
  CompleteInert
  CompleteWithBindableClaims
} derive(Debug, Eq)

///|
pub struct AotCandidateCoverageDigest {
  priv hex : String
} derive(Debug, Eq, Hash)

///|
/// Canonical startup summary for one exact ordered requirement set.
pub struct AotCandidateCoverage {
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest_sha256 : String
  priv requirement_count : Int
  priv candidate_count : Int
  priv missing_ordinals : ReadOnlyArray[Int]
  priv non_bindable_ordinals : ReadOnlyArray[Int]
  priv status : AotCandidateCoverageStatus
  priv digest : AotCandidateCoverageDigest
}
