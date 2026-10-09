///|
fn coverage_identity(fill : Char) -> @spec.ModelIdentity {
  try! @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(String::make(64, fill)),
    @spec.PlanDigest::from_sha256(String::make(64, 'f')),
  )
}

///|
fn coverage_limits(
  maximum_requirements? : Int = 16,
  maximum_candidates? : Int = 16,
  maximum_encoding_bytes? : Int = 4096,
) -> AotCandidateCoverageLimits {
  try! AotCandidateCoverageLimits::new(
    maximum_requirements~,
    maximum_candidates~,
    maximum_encoding_bytes~,
  )
}

///|
fn coverage_candidate(
  identity : @spec.ModelIdentity,
  ordinal : Int,
  bindable : Bool,
  recipe_fill? : Char,
) -> AotCandidateEvidenceInput {
  let recipe_character = match recipe_fill {
    Some(value) => value
    None =>
      match "0123456789abcdef".get_char((ordinal + 8) % 16) {
        Some(value) => value
        None => abort("valid hexadecimal candidate digest fill")
      }
  }
  try! AotCandidateEvidenceInput::new(
    source_identity=identity,
    requirements_digest_sha256=String::make(64, 'a'),
    requirement_ordinal=ordinal,
    source_digest_sha256=String::make(64, 'b'),
    recipe_digest_sha256=String::make(64, recipe_character),
    manifest_bindable=bindable,
  )
}

///|
test "coverage canonicalizes input order and reports exact gaps" {
  let identity = coverage_identity('1')
  let first = audit(
    source_identity=identity,
    requirements_digest_sha256=String::make(64, 'a'),
    requirement_count=4,
    candidates=[
      coverage_candidate(identity, 2, true),
      coverage_candidate(identity, 0, false),
    ],
    limits=coverage_limits(),
  )
  let reordered = audit(
    source_identity=identity,
    requirements_digest_sha256=String::make(64, 'a'),
    requirement_count=4,
    candidates=[
      coverage_candidate(identity, 0, false),
      coverage_candidate(identity, 2, true),
    ],
    limits=coverage_limits(),
  )
  assert_eq(first.status(), Incomplete)
  assert_eq(first.source_identity(), identity)
  assert_eq(first.requirements_digest_sha256(), String::make(64, 'a'))
  assert_eq(first.requirement_count(), 4)
  assert_eq(first.missing_ordinals(), [1, 3])
  assert_eq(first.non_bindable_ordinals(), [0])
  assert_eq(first.digest(), reordered.digest())
  try first.require_complete() catch {
    IncompleteCoverage(2) => ()
    _ => fail("incomplete coverage returned the wrong error")
  } noraise {
    _ => fail("partial candidate coverage was accepted as complete")
  }
}

///|
test "complete inert and claimed-bindable states remain distinct" {
  let identity = coverage_identity('2')
  let inert = audit(
    source_identity=identity,
    requirements_digest_sha256=String::make(64, 'a'),
    requirement_count=2,
    candidates=[
      coverage_candidate(identity, 0, false),
      coverage_candidate(identity, 1, true),
    ],
    limits=coverage_limits(),
  )
  assert_eq(inert.status(), CompleteInert)
  inert.require_complete()
  try inert.require_all_claimed_bindable() catch {
    NonBindableCoverage(1) => ()
    _ => fail("inert coverage returned the wrong error")
  } noraise {
    _ => fail("non-bindable candidate claims were promoted")
  }
  let claimed = audit(
    source_identity=identity,
    requirements_digest_sha256=String::make(64, 'a'),
    requirement_count=2,
    candidates=[
      coverage_candidate(identity, 0, true),
      coverage_candidate(identity, 1, true),
    ],
    limits=coverage_limits(),
  )
  assert_eq(claimed.status(), CompleteWithBindableClaims)
  claimed.require_all_claimed_bindable()
}

///|
test "coverage rejects hostile identity digest ordinal and duplicate evidence" {
  let identity = coverage_identity('3')
  let other = coverage_identity('4')
  let cases : Array[
    (Array[AotCandidateEvidenceInput], AotCandidateCoverageError),
  ] = [
    ([coverage_candidate(other, 0, false)], CandidateIdentityMismatch(0)),
    (
      [
        try! AotCandidateEvidenceInput::new(
          source_identity=identity,
          requirements_digest_sha256=String::make(64, 'e'),
          requirement_ordinal=0,
          source_digest_sha256=String::make(64, 'b'),
          recipe_digest_sha256=String::make(64, 'd'),
          manifest_bindable=false,
        ),
      ],
      CandidateRequirementsDigestMismatch(0),
    ),
    (
      [coverage_candidate(identity, 2, false)],
      InvalidRequirementOrdinal(candidate=0, ordinal=2),
    ),
    (
      [
        coverage_candidate(identity, 0, false),
        coverage_candidate(identity, 0, true, recipe_fill='e'),
      ],
      DuplicateRequirementOrdinal(
        first_candidate=0,
        second_candidate=1,
        ordinal=0,
      ),
    ),
    (
      [
        coverage_candidate(identity, 0, false),
        coverage_candidate(identity, 1, true, recipe_fill='8'),
      ],
      DuplicateRecipeDigest(first_candidate=0, second_candidate=1),
    ),
  ]
  for entry in cases {
    let (candidates, expected) = entry
    try
      audit(
        source_identity=identity,
        requirements_digest_sha256=String::make(64, 'a'),
        requirement_count=2,
        candidates~,
        limits=coverage_limits(),
      )
    catch {
      error => assert_eq(error, expected)
    } noraise {
      _ => fail("hostile candidate evidence was accepted")
    }
  }
}

///|
test "coverage rejects malformed digests limits and encoding exhaustion" {
  let identity = coverage_identity('5')
  for hostile in [(0, 1, 256), (1, 0, 256), (1, 1, 255), (4097, 1, 256)] {
    let (requirements, candidates, encoding) = hostile
    try
      AotCandidateCoverageLimits::new(
        maximum_requirements=requirements,
        maximum_candidates=candidates,
        maximum_encoding_bytes=encoding,
      )
    catch {
      InvalidLimit => ()
      _ => fail("invalid coverage limit returned the wrong error")
    } noraise {
      _ => fail("invalid coverage limit was admitted")
    }
  }
  try
    AotCandidateEvidenceInput::new(
      source_identity=identity,
      requirements_digest_sha256=String::make(64, 'A'),
      requirement_ordinal=0,
      source_digest_sha256=String::make(64, 'b'),
      recipe_digest_sha256=String::make(64, 'c'),
      manifest_bindable=false,
    )
  catch {
    InvalidDigest(RequirementsDigest) => ()
    _ => fail("malformed digest returned the wrong error")
  } noraise {
    _ => fail("uppercase digest was admitted")
  }
  let malformed_fields : Array[AotCandidateDigestField] = [
    SourceDigest,
    RecipeDigest,
  ]
  for field in malformed_fields {
    try
      AotCandidateEvidenceInput::new(
        source_identity=identity,
        requirements_digest_sha256=String::make(64, 'a'),
        requirement_ordinal=0,
        source_digest_sha256=if field == SourceDigest {
          "short"
        } else {
          String::make(64, 'b')
        },
        recipe_digest_sha256=if field == RecipeDigest {
          "short"
        } else {
          String::make(64, 'c')
        },
        manifest_bindable=false,
      )
    catch {
      InvalidDigest(actual) => assert_eq(actual, field)
      _ => fail("malformed candidate digest returned the wrong error")
    } noraise {
      _ => fail("malformed candidate digest was admitted")
    }
  }
  try
    audit(
      source_identity=identity,
      requirements_digest_sha256=String::make(64, 'a'),
      requirement_count=0,
      candidates=[],
      limits=coverage_limits(),
    )
  catch {
    InvalidRequirementCount(0) => ()
    _ => fail("invalid requirement count returned the wrong error")
  } noraise {
    _ => fail("invalid requirement count was admitted")
  }
  try
    audit(
      source_identity=identity,
      requirements_digest_sha256=String::make(64, 'a'),
      requirement_count=2,
      candidates=[
        coverage_candidate(identity, 0, false),
        coverage_candidate(identity, 1, false),
      ],
      limits=coverage_limits(maximum_candidates=1),
    )
  catch {
    CandidateCountExceedsLimit(2) => ()
    _ => fail("candidate ceiling returned the wrong error")
  } noraise {
    _ => fail("candidate ceiling was bypassed")
  }
  try
    audit(
      source_identity=identity,
      requirements_digest_sha256=String::make(64, 'a'),
      requirement_count=2,
      candidates=[coverage_candidate(identity, 0, false)],
      limits=coverage_limits(maximum_encoding_bytes=256),
    )
  catch {
    CoverageEncodingTooLarge => ()
    _ => fail("encoding exhaustion returned the wrong error")
  } noraise {
    _ => fail("coverage exceeded its canonical encoding budget")
  }
}
