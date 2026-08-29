///|
test "output exposes only inert bounded launch views" {
  let source = fixture()
  let profile = fixture_profile()
  let set = admit_fixture(
    source,
    profile,
    fixture_inputs(source.model, source.rank_plan, source.target, profile),
  )
  let contract = set.contracts()[4]
  assert_eq(contract.profile(), profile)
  assert_eq(contract.operation_kind(), CausalAttention)
  assert_eq(contract.query_head_count(), 2)
  assert_eq(contract.kv_head_count(), 1)
  assert_eq(contract.dimensions(), fixture_dimensions())
  assert_true(contract.operands().length() <= 16)
}
