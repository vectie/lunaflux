///|
test "paged index refinement reaches one queue without a host completion" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let external = context.allocate(1048576L)
  defer (external.close() catch { _ => () })
  let plan = @precision.IndexedSparseAttentionPrecision::new(
    index_head_count=1, index_head_dimension=4,
    query_head_count=2, key_value_head_count=1,
    query_key_dimension=4, value_dimension=4,
    pool_size=4, top_k=8, append_incomplete_tail=true,
    maximum_queries=8, maximum_history=1024,
    maximum_workspace_bytes=1048576L,
    paged_history=@history_domain.PagedHistoryDomain::new(
      rows_per_page=16, physical_pages=128,
      maximum_context=1024, maximum_sequences=2,
    ).unwrap(),
  )
  let source = @sparse_source.indexed_sparse_attention_source(plan, symbol_prefix="partitioned")
  assert_eq(source.execution_order(), ReadOnlyArray::from_array([0,3,1,2]))
  assert_eq(source.launch(3), (16,128))
  let frame = @sparse.IndexedSparseAttentionFrame::new(source)
  defer (frame.close() catch { _ => () })
  frame.prepare(
    context, aot, maximum_workspace_bytes=1048576L,
    query_counts=external, history_counts=external,
    query_positions=external, index_queries=external,
    index_head_weights=external, index_keys=external,
    pool_gate_scores=external, pool_ape=external,
    history_valid=external, queries=external, keys=external,
    values=external, output=external,
    page_ports=@sparse.SparseHistoryPagePorts::new(
      row_sequences=external, page_offsets=external,
      page_ids=external, page_generations=external,
    ),
  )
  assert_eq(frame.launches().length(), 4)
  let queue = context.create_ordered_kernel_executor(stream, frame.launches())
  defer (queue.close() catch { _ => () })
  let before = enqueues()
  let waits = blocking_syncs()
  begin_allocations()
  for _ in 0..<32 {
    for stage in 0..<4 { queue.enqueue(stage) }
    queue.record_completion()
    while !queue.poll_completion() { () }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues()-before, 128UL)
  assert_eq(blocking_syncs(), waits)
  queue.close()
  frame.close()
  external.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
