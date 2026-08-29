///|
test "canonical release campaign round trips and preserves losing workloads" {
  let record = release_test_record(Passed)
  let admitted = @evidence.admit_release_campaign(
    record.canonical_bytes(),
    expected_digest=record.digest(),
    limits=release_test_limits(),
  )
  assert_eq(admitted, record)
  assert_eq(admitted.evidence_count(), 16)
  assert_eq(admitted.reviewer_count(), 3)
  guard admitted.evidence_at(4) is Some(performance) else {
    fail("missing performance role")
  }
  assert_eq(performance.role(), Performance)
  assert_eq(performance.workload_count(), 9)
  assert_eq(performance.losing_workload_count(), 3)
  assert_eq(admitted.declared_status(), Passed)
}

///|
test "four campaign states remain distinct after host verification" {
  for status in [@evidence.Pending, Passed, Failed, NotRun] {
    let record = release_test_record(status)
    assert_eq(record.declared_status(), status)
    let verified = @evidence.verify_release_campaign(
      record,
      policy=release_test_policy(),
      evidence_observations=release_test_evidence_observations(status),
      reviewer_observations=release_test_reviewer_observations(status),
    )
    assert_eq(verified.status(), status)
    assert_eq(verified.campaign_digest(), record.digest())
    assert_eq(verified.is_accepted(), status == Passed)
  }
}

///|
test "runtime and external adapter builds and paths are independent roles" {
  let record = release_test_record(Passed)
  let expected_roles = [
    @evidence.RuntimeBuild,
    ExternalAdapterBuild,
    DirectPath,
    ExternalAdapterPath,
  ]
  for offset, role in expected_roles {
    guard record.evidence_at(11 + offset) is Some(entry) else {
      fail("missing independent build/path role")
    }
    assert_eq(entry.role(), role)
  }
}

///|
test "campaign digest changes with losses and declared status" {
  let passed = release_test_record(Passed)
  let pending = release_test_record(Pending)
  assert_true(passed.digest() != pending.digest())
  let changed = release_test_entries(Passed)
  changed[4] = @evidence.ReleaseEvidenceEntry::new(
    role=Performance,
    status=Passed,
    artifact=changed[4].artifact(),
    workload_count=9,
    losing_workload_count=4,
  )
  let changed_record = @evidence.ReleaseCampaignRecord::new(
    campaign_id=7UL,
    subject=release_test_subject(),
    image=release_test_image(),
    physical_target=release_test_target(),
    evidence=changed,
    reviewers=release_test_reviewers(Passed),
    limits=release_test_limits(),
  )
  assert_true(passed.digest() != changed_record.digest())
}
