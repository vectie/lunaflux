///|
test "seeded metadata is startup-owned and warmed parent steps allocate nothing" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let operand = context.allocate(32L)
  defer (operand.close() catch { _ => () })
  let output = context.allocate(24L)
  defer (output.close() catch { _ => () })
  let frame = @text_frame.SeededTokenBlockFrame::new(
    @precision.SeededTokenBlockPrecision::new(
      vocabulary_size=17,
      block_rows=5,
      maximum_main_rows=8,
      noise_token_id=16,
    ),
    symbol_prefix="seed_fixture",
  )
  defer (frame.close() catch { _ => () })
  let before = live_resources()
  try
    frame.prepare(
      context,
      aot,
      main_counts=operand,
      start_position=operand,
      seed_token=operand,
      output_chain=output,
      maximum_workspace_bytes=59L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("undersized seed workspace accepted")
  }
  assert_eq(live_resources(), before)
  frame.prepare(
    context,
    aot,
    main_counts=operand,
    start_position=operand,
    seed_token=operand,
    output_chain=output,
    maximum_workspace_bytes=60L,
  )
  assert_eq(frame.launches().length(), 1)
  assert_eq(frame.counts().byte_count(), 20L)
  assert_eq(frame.token_ids().byte_count(), 20L)
  assert_eq(frame.positions().byte_count(), 20L)
  let queue = context.create_ordered_kernel_executor(stream, frame.launches())
  defer (queue.abort() catch { _ => () })
  defer (queue.close() catch { _ => () })
  let calls = enqueues()
  begin_allocations()
  for _ in 0..<32 {
    queue.enqueue(0)
    queue.record_completion()
    while !queue.poll_completion() {
      ()
    }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - calls, 32UL)
  queue.close()
  frame.close()
  frame.close()
  output.close()
  operand.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "vector seed preparation validates the complete borrowed span" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let counts = context.allocate(32L)
  defer (counts.close() catch { _ => () })
  let scalar = context.allocate(4L)
  defer (scalar.close() catch { _ => () })
  let output = context.allocate(24L)
  defer (output.close() catch { _ => () })
  let frame = @text_frame.SeededTokenBlockFrame::new(
    @precision.SeededTokenBlockPrecision::new(
      vocabulary_size=17,
      block_rows=5,
      maximum_main_rows=8,
      maximum_seed_tokens=8,
      noise_token_id=16,
    ),
    symbol_prefix="vector_seed_fixture",
  )
  defer (frame.close() catch { _ => () })
  let before = live_resources()
  try
    frame.prepare(
      context,
      aot,
      main_counts=counts,
      start_position=scalar,
      seed_token=scalar,
      output_chain=output,
      maximum_workspace_bytes=60L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("scalar allocation admitted for eight seed tokens")
  }
  assert_eq(live_resources(), before)
  frame.prepare(
    context,
    aot,
    main_counts=counts,
    start_position=scalar,
    seed_token=counts,
    output_chain=output,
    maximum_workspace_bytes=60L,
  )
  assert_eq(frame.launches().length(), 1)
  frame.close()
  output.close()
  scalar.close()
  counts.close()
  aot.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
