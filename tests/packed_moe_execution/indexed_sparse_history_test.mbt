///|
test "request history composes one queue and independent owners without hot allocation" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let attention = @precision.IndexedSparseAttentionPrecision::new(
    index_head_count=1,
    index_head_dimension=4,
    query_head_count=2,
    key_value_head_count=1,
    query_key_dimension=4,
    value_dimension=4,
    pool_size=4,
    top_k=8,
    append_incomplete_tail=true,
    maximum_queries=8,
    maximum_history=32,
    maximum_workspace_bytes=100000L,
  )
  let source = @sparse_source.indexed_sparse_history_source(
    @precision.IndexedSparseHistoryPrecision::new(
      attention,
      maximum_state_bytes=1216L,
    ),
    symbol_prefix="retained",
  )
  let history = @sparse.IndexedSparseHistory::new(source)
  defer (history.close() catch { _ => () })
  let other = @sparse.IndexedSparseHistory::new(source)
  defer (other.close() catch { _ => () })
  let external = context.allocate(2048L)
  defer (external.close() catch { _ => () })
  for owner in [history, other] {
    owner.prepare(
      context,
      aot,
      maximum_state_bytes=1216L,
      query_counts=external,
      reset_flag=external,
      index_keys=external,
      pool_gate_scores=external,
      keys=external,
      values=external,
    )
  }
  assert_eq(history.state_bytes(), 1216L)
  assert_true(
    !history.append_descriptor().same_storage(other.append_descriptor()),
  )
  assert_true(!history.query_positions().same_storage(other.query_positions()))
  let frame = @sparse.IndexedSparseAttentionFrame::new(
    @sparse_source.indexed_sparse_attention_source(
      attention,
      symbol_prefix="retained_attention",
    ),
  )
  defer (frame.close() catch { _ => () })
  frame.prepare_retained(
    context,
    aot,
    history,
    maximum_workspace_bytes=580L,
    query_counts=external,
    index_queries=external,
    index_head_weights=external,
    pool_ape=external,
    queries=external,
    output=external,
  )
  assert_eq(frame.launches().length(), 6)
  let queue = context.create_ordered_kernel_executor(stream, frame.launches())
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
  queue.enqueue(0)
  queue.abort()
  queue.close()
  frame.close()
  history.close()
  history.close()
  other.close()
  external.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
