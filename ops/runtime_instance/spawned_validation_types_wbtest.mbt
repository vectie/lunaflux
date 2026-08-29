///|
fn spawned_validation_wb_identity() -> @spec.ModelIdentity raise {
  @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    @spec.PlanDigest::from_sha256(
      "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
}

///|
test "spawned validation contract is exactly two-token and same-page bounded" {
  assert_true(
    spawned_validation_contract_values_valid(64, 1031, 2185, 1, 3, 8, 1024),
  )
  assert_false(
    spawned_validation_contract_values_valid(0, 1031, 2185, 1, 3, 8, 1024),
  )
  assert_false(
    spawned_validation_contract_values_valid(64, -1, 2185, 1, 3, 8, 1024),
  )
  assert_false(
    spawned_validation_contract_values_valid(64, 1031, 2185, 1, 2, 8, 1024),
  )
  assert_false(
    spawned_validation_contract_values_valid(64, 1031, 2185, 8, 10, 8, 1024),
  )
  assert_false(
    spawned_validation_contract_values_valid(64, 1031, 2185, 7, 9, 8, 1024),
  )
  assert_false(
    spawned_validation_contract_values_valid(64, 1031, 2185, 1, 3, 8, 0),
  )
}

///|
test "spawned validation contract copies caller-owned request bytes" {
  let source : FixedArray[Byte] = [b'a', b'b', b'c']
  let contract = RuntimeSpawnedValidationContract::new(
    source,
    spawned_validation_wb_identity(),
    1031,
    2185,
    1,
    3,
    8,
    1024,
  )
  source[0] = b'z'
  assert_eq(contract.frame[0], b'a')
}

///|
test "spawned validation rejects token and ordinal substitution" {
  assert_true(spawned_validation_token_matches(1031, 0, 1031, 0))
  assert_false(spawned_validation_token_matches(2049, 0, 1031, 0))
  assert_false(spawned_validation_token_matches(1031, 1, 1031, 0))
  assert_false(spawned_validation_token_matches(2185, 1, 688, 1))
}

///|
test "spawned KV observation accepts live or fully retired balanced state" {
  assert_true(spawned_validation_kv_state_valid(1, 1, 31, 32))
  assert_true(spawned_validation_kv_state_valid(0, 0, 32, 32))
  assert_false(spawned_validation_kv_state_valid(0, 1, 31, 32))
  assert_false(spawned_validation_kv_state_valid(1, 0, 32, 32))
  assert_false(spawned_validation_kv_state_valid(0, 0, 31, 32))
}
