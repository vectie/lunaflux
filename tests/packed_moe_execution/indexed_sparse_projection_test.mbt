///|
test "hidden sparse projection cache attention uses one queue without hot allocation" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let plan = @precision.IndexedSparseAttentionPrecision::new(
    index_head_count=2,
    index_head_dimension=4,
    query_head_count=2,
    key_value_head_count=2,
    query_key_dimension=4,
    value_dimension=4,
    pool_size=4,
    top_k=8,
    append_incomplete_tail=true,
    maximum_queries=8,
    maximum_history=32,
    maximum_workspace_bytes=100000L,
  )
  let projection = @projection_frame.IndexedSparseProjectionFrame::new(
    @projection_source.indexed_sparse_projection_source(
      @precision.IndexedSparseProjectionPrecision::new(
        plan,
        hidden_width=8,
        query_rank=4,
        key_value_rank=2,
        rms_epsilon=1.0e-5,
        index_layer_norm_epsilon=1.0e-5,
        maximum_workspace_bytes=1312L,
      ),
      symbol_prefix="projection",
    ),
  )
  defer (projection.close() catch { _ => () })
  let external = context.allocate(4096L)
  defer (external.close() catch { _ => () })
  projection.prepare(
    context,
    aot,
    maximum_workspace_bytes=1312L,
    query_counts=external,
    hidden=external,
    weights=Array::make(12, external),
  )
  let history = @sparse.IndexedSparseHistory::new(
    @sparse_source.indexed_sparse_history_source(
      @precision.IndexedSparseHistoryPrecision::new(
        plan,
        maximum_state_bytes=10000L,
      ),
      symbol_prefix="history",
    ),
  )
  defer (history.close() catch { _ => () })
  let attention = @sparse.IndexedSparseAttentionFrame::new(
    @sparse_source.indexed_sparse_attention_source(
      plan,
      symbol_prefix="attention",
    ),
  )
  defer (attention.close() catch { _ => () })
  let launches = projection.prepare_retained_attention(
    context,
    aot,
    history,
    attention,
    maximum_state_bytes=10000L,
    maximum_attention_workspace_bytes=100000L,
    reset_flag=external,
    pool_ape=external,
    output=external,
  )
  assert_eq(launches.length(), 18)
  let queue = context.create_ordered_kernel_executor(stream, launches)
  defer (queue.abort() catch { _ => () })
  defer (queue.close() catch { _ => () })
  let before = enqueues()
  let waits = blocking_syncs()
  begin_allocations()
  for _ in 0..<32 {
    for stage in 0..<18 {
      queue.enqueue(stage)
    }
    queue.record_completion()
    while !queue.poll_completion() {
      ()
    }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - before, 576UL)
  assert_eq(blocking_syncs(), waits)
  queue.enqueue(0)
  queue.abort()
  queue.close()
  attention.close()
  history.close()
  projection.close()
  projection.close()
  external.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
