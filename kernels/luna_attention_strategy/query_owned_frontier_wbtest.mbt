///|
test "query ownership search is Cartesian with unique stable identities" {
  let specs = query_owned_attention_specs()
  assert_eq(specs.length(), 18)
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
    for k in [32, 64, 128] {
      assert_eq(
        specs.filter(s => s.query_rows == q && s.key_value_rows == k).length(),
        3,
      )
    }
  }
}

///|
test "approximate query ownership is an injective separate identity domain" {
  let strict = query_owned_attention_specs()
  let approximate = approximate_query_owned_attention_specs()
  assert_eq(approximate.length(), strict.length())
  for index in 0..<strict.length() {
    let old = strict[index]
    let next = approximate[index]
    assert_eq(next.stable_id, old.stable_id + 30000)
    assert_eq(next.compute, AttentionMatrixQkPvApproxExp)
    assert_eq(next.query_rows, old.query_rows)
    assert_eq(next.key_value_rows, old.key_value_rows)
    assert_eq(next.memory, old.memory)
    assert_eq(next.pipeline_stages, old.pipeline_stages)
  }
}
