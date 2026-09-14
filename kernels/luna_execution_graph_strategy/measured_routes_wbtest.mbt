///|
test "measured routes retain phase query batch and context distinctions" {
  let table = build_execution_graph_bucket_table(16, 4096, 8192, 65536)
  let shape = fn(phase, rows, tokens, context) raise {
    ExecutionGraphShape::new(
      phase~,
      batch_rows=rows,
      query_tokens=tokens,
      context_tokens=context,
    )
  }
  let short = shape(PrefillGraph, 8, 1528, 256)
  let long = shape(PrefillGraph, 8, 1528, 8192)
  let observation = fn(s, id, ns) raise {
    ExecutionGraphRouteMeasurement::new(
      s,
      candidate=id,
      median_ns=ns,
      samples=5,
    )
  }
  let records = [
    observation(short, 322, 91000L),
    observation(short, 324, 77000L),
    observation(long, 322, 500000L),
    observation(long, 324, 550000L),
  ]
  let selected = ExecutionGraphMeasuredRoutes::new(
    table,
    baseline=322,
    records[:],
  )
  assert_eq(selected.select(short), Some(324))
  assert_eq(selected.select(long), Some(322))
  assert_eq(selected.select(shape(MixedGraph, 8, 1528, 256)), None)
  assert_eq(selected.select(shape(PrefillGraph, 1, 1528, 256)), None)
  assert_eq(selected.select(shape(PrefillGraph, 8, 129, 256)), None)
  let reversed = records.rev()
  let reordered = ExecutionGraphMeasuredRoutes::new(
    table,
    baseline=322,
    reversed[:],
  )
  assert_eq(reordered.select(short), selected.select(short))
}

///|
test "measured routes require matched baselines and reject noise and duplicates" {
  let table = build_execution_graph_bucket_table(1, 1, 1, 3)
  let shape = ExecutionGraphShape::new(
    phase=DecodeGraph,
    batch_rows=1,
    query_tokens=1,
    context_tokens=1,
  )
  let record = fn(id, ns) raise {
    ExecutionGraphRouteMeasurement::new(
      shape,
      candidate=id,
      median_ns=ns,
      samples=3,
    )
  }
  let base = record(0, 10000L)
  let noise = record(1, 9999L)
  assert_eq(
    ExecutionGraphMeasuredRoutes::new(table, baseline=0, [base, noise]).select(
      shape,
    ),
    Some(0),
  )
  assert_eq(
    ExecutionGraphMeasuredRoutes::new(table, baseline=0, [
      base,
      record(1, 9900L),
    ]).select(shape),
    Some(1),
  )
  for invalid in [[noise], [base, base], [base, noise, noise]] {
    try ExecutionGraphMeasuredRoutes::new(table, baseline=0, invalid[:]) catch {
      InvalidExecutionGraphBucket(_) => ()
    } noraise {
      _ => fail("invalid route observations accepted")
    }
  }
  let large = record(0, 9223372036854775807L)
  assert_eq(
    ExecutionGraphMeasuredRoutes::new(table, baseline=0, [large, record(1, 1L)]).select(
      shape,
    ),
    Some(1),
  )
}
