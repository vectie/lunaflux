///|
test "group admission is rank-order canonical with rank-specific identities" {
  let first = bootstrap_test_group()
  let reordered = bootstrap_test_group(claims=[
    bootstrap_test_claim(1),
    bootstrap_test_claim(0),
  ])
  assert_eq(first.digest, reordered.digest)
  assert_eq(first.envelopes.length(), 2)
  assert_eq(first.envelopes[0].group_digest, first.envelopes[1].group_digest)
  assert_true(
    first.envelopes[0].device_plan_digest !=
    first.envelopes[1].device_plan_digest,
  )
  assert_true(
    first.envelopes[0].collective_digest != first.envelopes[1].collective_digest,
  )
}

///|
test "group admission rejects duplicate incomplete and substituted rank sets" {
  try
    bootstrap_test_group(claims=[
      bootstrap_test_claim(0),
      bootstrap_test_claim(0),
    ])
  catch {
    Rejected(rule=DuplicateRank, rank=0) => ()
    _ => fail("unexpected duplicate-rank rejection")
  } noraise {
    _ => fail("duplicate rank was admitted")
  }
  try
    bootstrap_test_group(claims=[bootstrap_test_claim(0)], world_size=2)
  catch {
    Rejected(rule=IncompleteRankSet, rank=-1) => ()
    _ => fail("unexpected incomplete-set rejection")
  } noraise {
    _ => fail("incomplete rank set was admitted")
  }
  let foreign = try! @spec.ModelIdentity::new(
    try! @spec.ContentDigest::from_sha256(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ),
    bootstrap_test_identity().plan(),
  )
  let substituted = bootstrap_test_claim_for(
    foreign, 1, 2, "6666666666666666666666666666666666666666666666666666666666666666",
    "8989898989898989898989898989898989898989898989898989898989898989",
  )
  try
    bootstrap_test_group(claims=[bootstrap_test_claim(0), substituted])
  catch {
    Rejected(rule=ModelIdentity, rank=1) => ()
    _ => fail("unexpected identity substitution rejection")
  } noraise {
    _ => fail("substituted model identity was admitted")
  }
}

///|
test "rank-local KV substitution changes the canonical group identity" {
  let first = bootstrap_test_group()
  let changed = bootstrap_test_group(claims=[
    bootstrap_test_claim(0),
    bootstrap_test_claim(
      1,
      kv_hex="abababababababababababababababababababababababababababababababab",
    ),
  ])
  assert_true(first.digest != changed.digest)
  assert_true(first.envelopes[1].kv_digest != changed.envelopes[1].kv_digest)
}

///|
test "execution-plan substitution changes group identity and malformed identity fails" {
  let first = bootstrap_test_group()
  let changed = bootstrap_test_group(claims=[
    bootstrap_test_claim(0),
    bootstrap_test_claim(
      1,
      execution_hex="abababababababababababababababababababababababababababababababab",
    ),
  ])
  assert_true(first.digest != changed.digest)
  assert_true(
    first.envelopes[1].execution_plan_digest_sha256 !=
    changed.envelopes[1].execution_plan_digest_sha256,
  )
  try
    bootstrap_test_group(claims=[
      bootstrap_test_claim(0),
      bootstrap_test_claim(1, execution_hex="not-a-digest"),
    ])
  catch {
    Rejected(rule=ExecutionPlan, rank=1) => ()
    _ => fail("unexpected execution-plan digest rejection")
  } noraise {
    _ => fail("malformed execution-plan identity was admitted")
  }
}
