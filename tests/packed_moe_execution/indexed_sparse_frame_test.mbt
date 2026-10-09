///|
test "projected sparse frame composes three launches with bounded decode scratch" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let plan = @precision.IndexedSparseAttentionPrecision::new(
    index_head_count=1,
    index_head_dimension=4,
    query_head_count=2,
    key_value_head_count=1,
    query_key_dimension=4,
    value_dimension=4,
    pool_size=4,
    top_k=8,
    append_incomplete_tail=true,
    maximum_queries=1,
    maximum_history=32,
    maximum_workspace_bytes=100000L,
  )
  let frame = @sparse.IndexedSparseAttentionFrame::new(
    @sparse_source.indexed_sparse_attention_source(
      plan,
      symbol_prefix="sparse_frame",
    ),
  )
  defer (frame.close() catch { _ => () })
  assert_eq(frame.workspace_bytes(), 272L)
  let external = context.allocate(1024L)
  defer (external.close() catch { _ => () })
  frame.prepare(
    context,
    aot,
    maximum_workspace_bytes=272L,
    query_counts=external,
    history_counts=external,
    query_positions=external,
    index_queries=external,
    index_head_weights=external,
    index_keys=external,
    pool_gate_scores=external,
    pool_ape=external,
    history_valid=external,
    queries=external,
    keys=external,
    values=external,
    output=external,
  )
  ignore(frame.selected_indices())
  let consumer = @sparse.IndexedSparseAttentionFrame::for_shared_indices(
    @sparse_source.indexed_sparse_attention_source(
      plan,
      symbol_prefix="sparse_reuse",
    ),
  )
  defer (consumer.close() catch { _ => () })
  assert_eq(consumer.workspace_bytes(), 0L)
  consumer.prepare_shared_indices(
    aot,
    query_counts=external,
    history_counts=external,
    query_positions=external,
    history_valid=external,
    queries=external,
    keys=external,
    values=external,
    selected_indices=frame.selected_indices(),
    output=external,
  )
  let combined = frame.launches().to_owned()
  combined.append(consumer.launches())
  let queue = context.create_ordered_kernel_executor(stream, combined)
  defer (queue.abort() catch { _ => () })
  defer (queue.close() catch { _ => () })
  assert_eq(queue.kernel_count(), 4)
  let before = enqueues()
  let waits = blocking_syncs()
  begin_allocations()
  for _ in 0..<32 {
    for stage in 0..<4 {
      queue.enqueue(stage)
    }
    queue.record_completion()
    while !queue.poll_completion() {
      ()
    }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - before, 128UL)
  assert_eq(blocking_syncs(), waits)
  queue.enqueue(0)
  queue.abort()
  queue.close()
  consumer.close()
  frame.close()
  frame.close()
  external.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
