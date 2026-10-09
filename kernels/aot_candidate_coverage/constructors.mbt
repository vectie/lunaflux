///|
fn valid_sha256(value : StringView) -> Bool {
  value.length() == 64 &&
  value.all(character => character is ('0'..='9' | 'a'..='f'))
}

///|
pub fn AotCandidateCoverageLimits::new(
  maximum_requirements~ : Int,
  maximum_candidates~ : Int,
  maximum_encoding_bytes~ : Int,
) -> Self raise AotCandidateCoverageError {
  if maximum_requirements <= 0 ||
    maximum_requirements > 4096 ||
    maximum_candidates <= 0 ||
    maximum_candidates > 4096 ||
    maximum_encoding_bytes < 256 ||
    maximum_encoding_bytes > 1048576 {
    raise InvalidLimit
  }
  { maximum_requirements, maximum_candidates, maximum_encoding_bytes, }
}

///|
pub fn AotCandidateEvidenceInput::new(
  source_identity~ : @spec.ModelIdentity,
  requirements_digest_sha256~ : String,
  requirement_ordinal~ : Int,
  source_digest_sha256~ : String,
  recipe_digest_sha256~ : String,
  manifest_bindable~ : Bool,
) -> Self raise AotCandidateCoverageError {
  if !valid_sha256(requirements_digest_sha256) {
    raise InvalidDigest(RequirementsDigest)
  }
  if !valid_sha256(source_digest_sha256) {
    raise InvalidDigest(SourceDigest)
  }
  if !valid_sha256(recipe_digest_sha256) {
    raise InvalidDigest(RecipeDigest)
  }
  {
    source_identity,
    requirements_digest_sha256,
    requirement_ordinal,
    source_digest_sha256,
    recipe_digest_sha256,
    manifest_bindable,
  }
}
