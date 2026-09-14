///|
test "scalar bucket bounds preserve immutable selection across later selects" {
  let table = build_execution_graph_bucket_table(8, 32, 4096, 1024)
  for phase in [PrefillGraph, DecodeGraph, MixedGraph] {
    for rows in [1, 3, 8] {
      let first = table.select(
        ExecutionGraphShape::new(
          phase~,
          batch_rows=rows,
          query_tokens=17,
          context_tokens=33,
        ),
      )
      let original_slot = first.slot()
      let second = table.select(
        ExecutionGraphShape::new(
          phase~,
          batch_rows=8,
          query_tokens=32,
          context_tokens=4096,
        ),
      )
      assert_eq(first.slot(), original_slot)
      assert_eq(
        (first.batch_rows(), first.query_tokens(), first.context_tokens()),
        first.upper_bounds(),
      )
      assert_eq(first.query_tokens(), 32)
      assert_eq(first.context_tokens(), 64)
      assert_eq(second.context_tokens(), 4096)
    }
  }
}
