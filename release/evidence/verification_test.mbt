///|
test "missing physical target evidence never becomes success" {
  let record = release_test_record(Passed)
  let observations = release_test_evidence_observations(Passed)
  ignore(observations.remove(0))
  try
    @evidence.verify_release_campaign(
      record,
      policy=release_test_policy(),
      evidence_observations=observations,
      reviewer_observations=release_test_reviewer_observations(Passed),
    )
  catch {
    @evidence.MissingObservation => ()
    _ => fail("unexpected missing physical evidence failure")
  } noraise {
    _ => fail("missing physical evidence yielded success")
  }
}

///|
test "missing reviewer attestation never becomes success" {
  let observations = release_test_reviewer_observations(Passed)
  ignore(observations.remove(2))
  try
    @evidence.verify_release_campaign(
      release_test_record(Passed),
      policy=release_test_policy(),
      evidence_observations=release_test_evidence_observations(Passed),
      reviewer_observations=observations,
    )
  catch {
    @evidence.MissingObservation => ()
    _ => fail("unexpected missing reviewer evidence failure")
  } noraise {
    _ => fail("missing reviewer attestation yielded success")
  }
}

///|
test "reviewer identity is checked against independent policy" {
  let wrong_policy = @evidence.ReleaseVerificationPolicy::new(
    subject=release_test_subject(),
    image=release_test_image(),
    physical_target=release_test_target(),
    security_reviewer=@evidence.ReleaseReviewerIdentityDigest::from_sha256(
      "9999999999999999999999999999999999999999999999999999999999999999",
    ),
    operations_reviewer=release_test_reviewer_identity(OperationsReviewer),
    performance_reviewer=release_test_reviewer_identity(PerformanceReviewer),
  )
  try
    @evidence.verify_release_campaign(
      release_test_record(Passed),
      policy=wrong_policy,
      evidence_observations=release_test_evidence_observations(Passed),
      reviewer_observations=release_test_reviewer_observations(Passed),
    )
  catch {
    @evidence.VerificationPolicyMismatch => ()
    _ => fail("unexpected reviewer policy failure")
  } noraise {
    _ => fail("unapproved reviewer identity yielded success")
  }
}

///|
test "artifact digest mismatch and duplicate observation fail closed" {
  let observations = release_test_evidence_observations(Passed)
  observations[1] = @evidence.ReleaseEvidenceObservation::from_payload(
    role=ReferenceCorrectness,
    "evidence/1.bin",
    b"different",
    limits=release_test_limits(),
  )
  try
    @evidence.verify_release_campaign(
      release_test_record(Passed),
      policy=release_test_policy(),
      evidence_observations=observations,
      reviewer_observations=release_test_reviewer_observations(Passed),
    )
  catch {
    @evidence.InvalidObservation => ()
    _ => fail("unexpected artifact observation failure")
  } noraise {
    _ => fail("wrong artifact digest yielded success")
  }
  let duplicate = release_test_evidence_observations(Passed)
  duplicate.push(duplicate[0])
  try
    @evidence.verify_release_campaign(
      release_test_record(Passed),
      policy=release_test_policy(),
      evidence_observations=duplicate,
      reviewer_observations=release_test_reviewer_observations(Passed),
    )
  catch {
    @evidence.InvalidObservation => ()
    _ => fail("unexpected duplicate observation failure")
  } noraise {
    _ => fail("duplicate observation yielded success")
  }
}

///|
test "host observations are bounded before hashing" {
  let tiny = @evidence.ReleaseCampaignLimits::new(
    max_canonical_bytes=1024,
    max_locator_bytes=16,
    max_artifact_bytes=1L,
  )
  try
    @evidence.ReleaseEvidenceObservation::from_payload(
      role=PhysicalTarget,
      "e/p",
      b"too large",
      limits=tiny,
    )
  catch {
    InvalidObservation => ()
    _ => fail("unexpected observation bound failure")
  } noraise {
    _ => fail("oversized host observation was hashed")
  }
}

///|
test "an observation cannot substitute an alias for the declared locator" {
  let observations = release_test_evidence_observations(Passed)
  observations[1] = @evidence.ReleaseEvidenceObservation::from_payload(
    role=ReferenceCorrectness,
    "evidence/alias.bin",
    release_test_evidence_payload(1),
    limits=release_test_limits(),
  )
  try
    @evidence.verify_release_campaign(
      release_test_record(Passed),
      policy=release_test_policy(),
      evidence_observations=observations,
      reviewer_observations=release_test_reviewer_observations(Passed),
    )
  catch {
    @evidence.InvalidObservation => ()
    _ => fail("unexpected locator-alias failure")
  } noraise {
    _ => fail("artifact content at the wrong locator was accepted")
  }
}

///|
test "reviewer observation count is bounded before role scans" {
  let observations = release_test_reviewer_observations(Passed)
  observations.push(observations[0])
  try
    @evidence.verify_release_campaign(
      release_test_record(Passed),
      policy=release_test_policy(),
      evidence_observations=release_test_evidence_observations(Passed),
      reviewer_observations=observations,
    )
  catch {
    @evidence.InvalidObservation => ()
    _ => fail("unexpected reviewer observation bound failure")
  } noraise {
    _ => fail("oversized reviewer observation set was accepted")
  }
}
