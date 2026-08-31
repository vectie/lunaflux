///|
test "execution graph buckets select exact phase and ceilings without scans" {
  let table = build_execution_graph_bucket_table(8, 32, 4096, 1024)
  let decode = table.select(
    ExecutionGraphShape::new(
      phase=DecodeGraph,
      batch_rows=5,
      query_tokens=5,
      context_tokens=3000,
    ),
  )
  assert_eq(decode.phase(), DecodeGraph)
  assert_eq(decode.upper_bounds(), (8, 8, 4096))
  let prefill = table.select(
    ExecutionGraphShape::new(
      phase=PrefillGraph,
      batch_rows=2,
      query_tokens=17,
      context_tokens=33,
    ),
  )
  assert_eq(prefill.phase(), PrefillGraph)
  assert_eq(prefill.upper_bounds(), (2, 32, 64))
  assert_true(decode.slot() != prefill.slot())
}

///|
test "execution graph bucket table rejects out of profile shapes" {
  let table = build_execution_graph_bucket_table(4, 16, 1024, 1024)
  try
    table.select(
      ExecutionGraphShape::new(
        phase=MixedGraph,
        batch_rows=5,
        query_tokens=4,
        context_tokens=32,
      ),
    )
  catch {
    InvalidExecutionGraphBucket(Limit) => ()
    _ => fail("unexpected bucket rejection")
  } noraise {
    _ => fail("expected bucket limit rejection")
  }
}
