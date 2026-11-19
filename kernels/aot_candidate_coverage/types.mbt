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
  InvalidRequirementCount(Int32)
  InvalidDigest(AotCandidateDigestField)
  CandidateCountExceedsLimit(Int32)
  CandidateIdentityMismatch(Int32)
  CandidateRequirementsDigestMismatch(Int32)
  InvalidRequirementOrdinal(candidate~ : Int32, ordinal~ : Int32)
  DuplicateRequirementOrdinal(
    first_candidate~ : Int32,
    second_candidate~ : Int32,
    ordinal~ : Int32
  )
  DuplicateRecipeDigest(first_candidate~ : Int32, second_candidate~ : Int32)
  CoverageEncodingTooLarge
  IncompleteCoverage(Int32)
  NonBindableCoverage(Int32)
} derive(Debug, Eq)

///|
pub struct AotCandidateCoverageLimits {
  priv maximum_requirements : Int32
  priv maximum_candidates : Int32
  priv maximum_encoding_bytes : Int32
}

///|
/// One already-lowered candidate claim. Construction validates only digest
/// syntax; `audit` authenticates identity, requirement digest, ordinal, and
/// uniqueness against the complete requirement set.
pub struct AotCandidateEvidenceInput {
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest_sha256 : String
  priv requirement_ordinal : Int32
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
  priv requirement_count : Int32
  priv candidate_count : Int32
  priv missing_ordinals : ReadOnlyArray[Int32]
  priv non_bindable_ordinals : ReadOnlyArray[Int32]
  priv status : AotCandidateCoverageStatus
  priv digest : AotCandidateCoverageDigest
}

///|
pub extend AotCandidateDigestField with @debug.Debug::{to_repr}

///|
pub extend AotCandidateDigestField with Eq::{not_equal, equal}

///|
pub extend AotCandidateCoverageError with @debug.Debug::{to_repr}

///|
pub extend AotCandidateCoverageError with Eq::{not_equal, equal}

///|
pub extend AotCandidateCoverageStatus with @debug.Debug::{to_repr}

///|
pub extend AotCandidateCoverageStatus with Eq::{not_equal, equal}

///|
pub extend AotCandidateCoverageDigest with @debug.Debug::{to_repr}

///|
pub extend AotCandidateCoverageDigest with Eq::{not_equal, equal}

///|
pub extend AotCandidateCoverageDigest with Hash::{hash, hash_combine}
