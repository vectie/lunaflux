///|
test "weight transfer composes operand selection and striped ownership" {
  for counts in [[2048, 1024, 1024], [3, 5, 2], [17]] {
    let segments = plan_projection_concatenated_rows(counts).unwrap()
    for width in [8, 16, 64] {
      for owners in [1, 32, 64, 128, 256] {
        let map = ProjectionVectorTransferMap::new(
          vectors=7 * width / 8,
          owners~,
        ).unwrap()
        let plan = ProjectionWeightTransferPlan::new(
          map,
          segments,
          row_width=width,
          vector_width=8,
        ).unwrap()
        assert_eq(plan.vector_map(), map)
        assert_eq(
          plan.row_selection(),
          plan_projection_row_selection(segments).unwrap(),
        )
        assert_eq(plan.row_vectors(), width / 8)
        // Each live vector retains exactly the original owner and round.
        for index in 0..<(7 * width / 8) {
          assert_eq(
            plan.vector_map().vector_at(index % owners, index / owners),
            Some(index),
          )
        }
      }
    }
  }
}

///|
test "weight transfer rejects partial rows and invalid vector geometry" {
  let map = ProjectionVectorTransferMap::new(vectors=3, owners=4).unwrap()
  let segments = plan_projection_concatenated_rows([3, 7]).unwrap()
  for (width, vector) in [(0, 8), (8, 0), (7, 8), (16, 8)] {
    assert_eq(
      ProjectionWeightTransferPlan::new(
        map,
        segments,
        row_width=width,
        vector_width=vector,
      ),
      None,
    )
  }
  assert_eq(
    ProjectionWeightTransferPlan::new(map, [], row_width=8, vector_width=8),
    None,
  )
}
