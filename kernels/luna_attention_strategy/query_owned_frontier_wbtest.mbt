///|
test "query ownership search is Cartesian with unique stable identities" {
  let specs = query_owned_attention_specs()
  assert_eq(specs.length(), 12)
  let ids : Map[Int, Bool] = Map([])
  for spec in specs {
    assert_false(ids.contains(spec.stable_id))
    ids[spec.stable_id] = true
    assert_eq(spec.subgroups_per_block, spec.query_rows / 16)
  }
  for id in 318..<=324 {
    assert_true(ids.contains(id))
  }
  for q in [32, 64] {
    for k in [32, 64] {
      assert_eq(
        specs.filter(s => s.query_rows == q && s.key_value_rows == k).length(),
        3,
      )
    }
  }
}
