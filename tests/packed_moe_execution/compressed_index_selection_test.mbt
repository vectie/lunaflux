///|
fn compressed_selection_fixture() -> @selection_frame.CompressedIndexSelectionFrame raise {
  @selection_frame.CompressedIndexSelectionFrame::new(
    @selection_source.compressed_index_selection_source(
      @precision.CompressedIndexSelectionPrecision::new(
        head_count=2,
        head_width=128,
        compression_ratio=4,
        maximum_rows=3,
        maximum_positions=16,
        top_k=2,
        maximum_workspace_bytes=60L,
      ),
      symbol_prefix="prepared_selection",
    ),
  )
}

///|
test "compressed index selection exposes two warm effects and validates all borrowed ports" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let operand = context.allocate(65536L)
  defer (operand.close() catch { _ => () })
  let short = context.allocate(8L)
  defer (short.close() catch { _ => () })
  let invalid = compressed_selection_fixture()
  let before_prepare = live_resources()
  try
    invalid.prepare(
      context,
      aot,
      counts=operand,
      positions=operand,
      cache_counts=operand,
      cache_append=operand,
      queries=operand,
      head_weights=operand,
      cache_values=operand,
      offsets=short,
      maximum_workspace_bytes=60L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("short offsets accepted")
  }
  assert_eq(live_resources(), before_prepare)
  invalid.close()
  let frame = compressed_selection_fixture()
  defer (frame.close() catch { _ => () })
  frame.prepare(
    context,
    aot,
    counts=operand,
    positions=operand,
    cache_counts=operand,
    cache_append=operand,
    queries=operand,
    head_weights=operand,
    cache_values=operand,
    offsets=operand,
    maximum_workspace_bytes=60L,
  )
  assert_eq(frame.outputs().row_ids.byte_count(), 24L)
  assert_eq(frame.outputs().row_counts.byte_count(), 12L)
  assert_eq(frame.launches().length(), 2)
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
    queue.enqueue(0)
    queue.enqueue(1)
    queue.record_completion()
    while !queue.poll_completion() {
      ()
    }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - before, 64UL)
  assert_eq(blocking_syncs(), waits)
  queue.close()
  frame.close()
  frame.close()
  short.close()
  operand.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
