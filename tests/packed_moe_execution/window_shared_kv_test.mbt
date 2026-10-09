///|
test "retained shared KV window contributes four effects and zero warm allocations" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  let stream = context.create_stream()
  let aot = context.load_module(b"fake\x00")
  let operand = context.allocate(65536L)
  let plan = @precision.WindowSharedKvPrecision::new(
    query_heads=2,
    head_width=128,
    window_tokens=128,
    maximum_rows=3,
    maximum_positions=1048576,
    maximum_state_bytes=40000L,
  )
  let frame = @window_frame.WindowSharedKvFrame::new(
    @window_source.window_shared_kv_source(plan, symbol_prefix="window_test"),
  )
  frame.prepare(
    context,
    aot,
    counts=operand,
    reset_flag=operand,
    positions=operand,
    queries=operand,
    shared_kv=operand,
    attention_sink=operand,
    maximum_state_bytes=40000L,
  )
  assert_eq(frame.launches().length(), 4)
  assert_eq(frame.output().byte_count(), 1536L)
  assert_eq(frame.append_descriptor().byte_count(), 12L)
  let queue = context.create_ordered_kernel_executor(stream, frame.launches())
  let submitted = enqueues()
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
  assert_eq(enqueues() - submitted, 128UL)
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
test "short borrowed window positions acquire no private allocation" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  let aot = context.load_module(b"fake\x00")
  let operand = context.allocate(65536L)
  let short = context.allocate(11L)
  let frame = @window_frame.WindowSharedKvFrame::new(
    @window_source.window_shared_kv_source(
      @precision.WindowSharedKvPrecision::new(
        query_heads=2,
        head_width=128,
        window_tokens=128,
        maximum_rows=3,
        maximum_positions=1048576,
        maximum_state_bytes=40000L,
      ),
      symbol_prefix="short_window",
    ),
  )
  let before = live_resources()
  try
    frame.prepare(
      context,
      aot,
      counts=operand,
      reset_flag=operand,
      positions=short,
      queries=operand,
      shared_kv=operand,
      attention_sink=operand,
      maximum_state_bytes=40000L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("accepted short positions")
  }
  assert_eq(live_resources(), before)
  frame.close()
  short.close()
  operand.close()
  aot.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "query rotary retained window and output use fifteen launches in one executor" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  let stream = context.create_stream()
  let aot = context.load_module(b"fake\x00")
  let operand = context.allocate(65536L)
  let prefix = query_kv_fixture()
  prefix.prepare(
    context,
    aot,
    counts=operand,
    hidden=operand,
    weights=[operand, operand, operand, operand, operand],
    maximum_workspace_bytes=6144L,
  )
  let projected = prefix.outputs()
  let rotary = rotary_kv_fixture()
  rotary.prepare(
    context,
    aot,
    counts=operand,
    positions=operand,
    queries=projected.queries,
    shared_kv=projected.shared_kv,
    maximum_workspace_bytes=2304L,
  )
  let rotated = rotary.outputs()
  let window = @window_frame.WindowSharedKvFrame::new(
    @window_source.window_shared_kv_source(
      @precision.WindowSharedKvPrecision::new(
        query_heads=2,
        head_width=128,
        window_tokens=128,
        maximum_rows=3,
        maximum_positions=1048576,
        maximum_state_bytes=40000L,
      ),
      symbol_prefix="composed_window",
    ),
  )
  window.prepare(
    context,
    aot,
    counts=operand,
    reset_flag=operand,
    positions=operand,
    queries=rotated.queries,
    shared_kv=rotated.shared_kv,
    attention_sink=operand,
    maximum_state_bytes=40000L,
  )
  let launches = prefix.launches().to_owned()
  launches.append(rotary.launches())
  launches.append(window.launches())
  assert_eq(launches.length(), 12)
  let output = output_fixture()
  output.prepare(
    context,
    aot,
    counts=operand,
    positions=operand,
    attention=window.output(),
    weights=[operand, operand],
    maximum_workspace_bytes=10000L,
  )
  launches.append(output.launches())
  assert_eq(launches.length(), 15)
  let queue = context.create_ordered_kernel_executor(stream, launches)
  let before = enqueues()
  begin_allocations()
  for _ in 0..<32 {
    for stage in 0..<15 {
      queue.enqueue(stage)
    }
    queue.record_completion()
    while !queue.poll_completion() {
      ()
    }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - before, 480UL)
  queue.close()
  output.close()
  window.close()
  rotary.close()
  prefix.close()
  operand.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
