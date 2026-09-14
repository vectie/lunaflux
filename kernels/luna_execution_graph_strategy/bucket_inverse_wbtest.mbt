///|
test "every prepared bucket round trips through immutable slot mapping" {
  let table = build_execution_graph_bucket_table(32, 2048, 65536, 65536)
  for slot in 0..<table.bucket_count {
    let bucket = table.bucket_at(slot)
    let shape = ExecutionGraphShape::new(
      phase=bucket.phase,
      batch_rows=bucket.batch_rows,
      query_tokens=bucket.query_tokens,
      context_tokens=bucket.context_tokens,
    )
    assert_eq(table.select(shape).slot, slot)
  }
}
