///|
test "finite transfer maps cover every vector exactly once including partial rounds" {
  for vectors in [1, 7, 31, 32, 33, 255, 256, 257, 512, 1024, 4096] {
    for owners in [1, 8, 16, 32, 64, 128, 256, 512] {
      let plan = ProjectionVectorTransferMap::new(vectors~, owners~).unwrap()
      let (extent, stride, rounds) = plan.geometry()
      assert_eq((extent, stride), (vectors, owners))
      let covered = FixedArray::make(vectors, 0)
      for owner in 0..<owners {
        for round in 0..<rounds {
          if plan.vector_at(owner, round) is Some(vector) {
            assert_eq(vector, owner + round * owners)
            covered[vector] += 1
          }
        }
        assert_eq(plan.vector_at(owner, rounds), None)
      }
      for count in covered {
        assert_eq(count, 1)
      }
    }
  }
}

///|
test "transfer map construction and indexing do not overflow at integer boundaries" {
  assert_eq(ProjectionVectorTransferMap::new(vectors=0, owners=1), None)
  assert_eq(ProjectionVectorTransferMap::new(vectors=1, owners=0), None)
  assert_eq(ProjectionVectorTransferMap::new(vectors=-1, owners=1), None)
  let maximum = 0x7fffffff
  let plan = ProjectionVectorTransferMap::new(vectors=maximum, owners=256).unwrap()
  let (_, _, rounds) = plan.geometry()
  assert_eq(plan.vector_at(254, rounds - 1), Some(maximum - 1))
  assert_eq(plan.vector_at(255, rounds - 1), None)
  assert_eq(plan.vector_at(-1, 0), None)
  assert_eq(plan.vector_at(256, 0), None)
  assert_eq(plan.vector_at(0, -1), None)
  assert_eq(plan.vector_at(0, maximum), None)
}
