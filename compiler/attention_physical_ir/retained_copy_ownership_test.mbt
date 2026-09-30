///|
test "retained copy slots are a deterministic disjoint vector cover" {
  for dimensions in [(1024, 64, 16), (1024, 256, 4), (19, 8, 3), (8, 32, 1)] {
    let (vectors, owners, slots) = dimensions
    let plan = @attention_physical_ir.RetainedCopyOwnership::new(
      vectors~,
      owners~,
    ).unwrap()
    assert_eq(plan.dimensions(), dimensions)
    let visited = Array::make(vectors, false)
    for slot in 0..<slots {
      for owner in 0..<owners {
        match plan.vector_index(slot, owner) {
          Some(vector) => {
            assert_false(visited[vector])
            visited[vector] = true
          }
          None => assert_true(slot * owners + owner >= vectors)
        }
      }
    }
    assert_true(visited.all(value => value))
    assert_eq(plan.vector_index(-1, 0), None)
    assert_eq(plan.vector_index(slots, 0), None)
    assert_eq(plan.vector_index(0, owners), None)
  }
  assert_eq(
    @attention_physical_ir.RetainedCopyOwnership::new(vectors=0, owners=32),
    None,
  )
  assert_eq(
    @attention_physical_ir.RetainedCopyOwnership::new(vectors=32, owners=0),
    None,
  )
}
