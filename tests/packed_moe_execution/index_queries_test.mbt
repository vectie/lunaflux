///|
fn learned_index_query_fixture() -> @index_queries_frame.LearnedIndexQueriesFrame raise {
  @index_queries_frame.LearnedIndexQueriesFrame::new(
    @index_queries_source.learned_index_queries_source(
      @precision.LearnedIndexQueriesPrecision::new(
        hidden_width=128,
        query_rank=128,
        head_count=2,
        @precision.RotaryHadamardFp4Precision::new(
          head_count=2,
          head_width=128,
          rotary_width=64,
          maximum_rows=3,
          maximum_positions=1024,
          frequency=Base(theta=10000.0),
          simulation_block_width=32,
          maximum_workspace_bytes=3072L,
        ),
        maximum_workspace_bytes=4632L,
      ),
      symbol_prefix="prepared_index_queries",
    ),
  )
}

///|
test "learned query projections and transform share six allocation-free effects" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let operand = context.allocate(65536L)
  defer (operand.close() catch { _ => () })
  let frame = learned_index_query_fixture()
  defer (frame.close() catch { _ => () })
  frame.prepare(
    context,
    aot,
    counts=operand,
    positions=operand,
    hidden=operand,
    low_rank_queries=operand,
    weights=[operand, operand],
    maximum_workspace_bytes=4632L,
  )
  assert_eq(frame.outputs().queries.byte_count(), 1536L)
  assert_eq(frame.outputs().head_weights.byte_count(), 12L)
  assert_eq(frame.launches().length(), 6)
  let queue = context.create_ordered_kernel_executor(
    stream,
    frame.launches().to_owned(),
  )
  defer (queue.abort() catch { _ => () })
  defer (queue.close() catch { _ => () })
  let before = enqueues()
  let waits = blocking_syncs()
  begin_allocations()
  for _ in 0..<32 {
    for stage in 0..<6 {
      queue.enqueue(stage)
    }
    queue.record_completion()
    while !queue.poll_completion() {
      ()
    }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - before, 192UL)
  assert_eq(blocking_syncs(), waits)
  queue.close()
  frame.close()
  frame.close()
  operand.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "short learned index query borrowed positions fail before allocation" {
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let operand = context.allocate(65536L)
  defer (operand.close() catch { _ => () })
  let short = context.allocate(4L)
  defer (short.close() catch { _ => () })
  let frame = learned_index_query_fixture()
  defer (frame.close() catch { _ => () })
  let before = live_resources()
  try
    frame.prepare(
      context,
      aot,
      counts=operand,
      positions=short,
      hidden=operand,
      low_rank_queries=operand,
      weights=[operand, operand],
      maximum_workspace_bytes=4632L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("short positions accepted")
  }
  assert_eq(live_resources(), before)
  frame.close()
}
