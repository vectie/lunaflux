///|
test "packed expert planes are disjoint from metadata and reuse one bounded epoch" {
  for transport in [@grouped_matrix_ir.PackedRows, PackedPipeline] {
    let grouping = @grouped_matrix_ir.ExpertGroupingPlan::new(
      maximum_rows=17,
      selected_experts=8,
      transport~,
    )
    let p = @grouped_matrix_ir.PackedExpertRows::new(
      grouping,
      after_bytes=9252L,
      payload_bytes=(2048, 512),
      scale_bytes=(256, 64),
    )
    assert_eq(grouping.transport(), transport)
    assert_eq(p.base(), 9264L)
    assert_eq(p.plane(false), (2048, 256, 4456448L))
    assert_eq(p.plane(true), (512, 64, 1114112L))
    assert_eq(p.total_bytes(), 5022768L)
    assert_eq(p.row(135, 15), Some(2175))
    assert_eq(p.row(136, 0), None)
    assert_eq(p.row(-1, 0), None)
    assert_eq(p.row(0, 16), None)
  }
}
