///|
test "independent projection domains cover every segment without a reduction merge" {
  let plans = [3, 65, 128].map(columns => {
    @grouped_matrix_ir.MatrixTilePlan::new(
      rows=17,
      columns~,
      inner=33,
      law=ShapeAwareFloat32,
    )
  })
  let p = @grouped_matrix_ir.SegmentedMatrixPlan::new(plans)
  assert_eq(p.offsets().to_owned(), [0, 2, 19])
  assert_eq(p.tile_count(), 51)
  try @grouped_matrix_ir.SegmentedMatrixPlan::new([plans[0]]) catch {
    _ => ()
  } noraise {
    _ => fail("single segment accepted")
  }
}

///|
test "bounded indirect workers partition all live tiles including empty and tail groups" {
  for groups in [1, 17, 512] {
    for columns in [1, 4, 128] {
      let p = @grouped_matrix_ir.IndirectMatrixTraversal::new(
        capacity_groups=groups,
        column_tiles=columns,
      )
      assert_eq(p.workers(), groups)
      for live in [0, 1, groups] {
        let count = p.tile_count(live)
        let visited = Array::make(count, 0)
        for worker in 0..<p.workers() {
          for tile = worker; tile < count; tile = tile + p.workers() {
            visited[tile] += 1
          }
        }
        assert_true(visited.all(n => n == 1))
      }
    }
  }
}
