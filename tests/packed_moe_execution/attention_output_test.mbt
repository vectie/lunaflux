///|
fn output_fixture() -> @output_frame.GroupedAttentionOutputFrame raise {
  @output_frame.GroupedAttentionOutputFrame::new(
    @output_source.grouped_attention_output_source(
      @precision.GroupedAttentionOutputPrecision::new(
        head_count=2,
        head_width=128,
        rotary_width=64,
        group_count=2,
        output_rank=128,
        hidden_width=128,
        maximum_rows=3,
        maximum_positions=1048576,
        frequency=Base(theta=10000.0),
        maximum_workspace_bytes=10000L,
      ),
      symbol_prefix="prepared_output",
    ),
  )
}

///|
test "grouped output contributes three launches with no warm allocations or extra wait" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  let stream = context.create_stream()
  let aot = context.load_module(b"fake\x00")
  let operand = context.allocate(65536L)
  let frame = output_fixture()
  frame.prepare(
    context,
    aot,
    counts=operand,
    positions=operand,
    attention=operand,
    weights=[operand, operand],
    maximum_workspace_bytes=10000L,
  )
  assert_eq(frame.launches().length(), 3)
  assert_eq(frame.output().byte_count(), 768L)
  let queue = context.create_ordered_kernel_executor(stream, frame.launches())
  let before = enqueues()
  let syncs = blocking_syncs()
  begin_allocations()
  for _ in 0..<32 {
    for stage in 0..<3 {
      queue.enqueue(stage)
    }
    queue.record_completion()
    while !queue.poll_completion() {
      ()
    }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - before, 96UL)
  assert_eq(blocking_syncs(), syncs)
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
test "short grouped output scale bank acquires no scratch" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  let aot = context.load_module(b"fake\x00")
  let operand = context.allocate(65536L)
  let short = context.allocate(32769L)
  let frame = output_fixture()
  let before = live_resources()
  try
    frame.prepare(
      context,
      aot,
      counts=operand,
      positions=operand,
      attention=operand,
      weights=[short, operand],
      maximum_workspace_bytes=10000L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("accepted short Output-A scale plane")
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
test "grouped output borrows the exact parent result without allocating or releasing it" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  let stream = context.create_stream()
  let aot = context.load_module(b"fake\x00")
  let operand = context.allocate(65536L)
  let result = context.allocate(768L)
  let frame = output_fixture()
  assert_eq(
    frame.workspace_bytes() - frame.workspace_bytes(output_borrowed=true),
    768L,
  )
  let before = live_resources()
  frame.prepare(
    context,
    aot,
    counts=operand,
    positions=operand,
    attention=operand,
    weights=[operand, operand],
    output=result,
    maximum_workspace_bytes=frame.workspace_bytes(output_borrowed=true),
  )
  assert_true(frame.output().same_storage(result))
  // Two scratch buffers, rather than the old three owned outputs.
  assert_eq(live_resources() - before, 5)
  let queue = context.create_ordered_kernel_executor(stream, frame.launches())
  let waits = blocking_syncs()
  begin_allocations()
  for _ in 0..<32 {
    for stage in 0..<3 {
      queue.enqueue(stage)
    }
    queue.record_completion()
    while !queue.poll_completion() {
      ()
    }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(blocking_syncs(), waits)
  queue.close()
  frame.close()
  assert_eq(live_resources(), before)
  result.close()
  operand.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "short borrowed grouped output is rejected before acquiring scratch" {
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let operand = context.allocate(65536L)
  defer (operand.close() catch { _ => () })
  let short = context.allocate(767L)
  defer (short.close() catch { _ => () })
  let frame = output_fixture()
  defer (frame.close() catch { _ => () })
  let before = live_resources()
  try
    frame.prepare(
      context,
      aot,
      counts=operand,
      positions=operand,
      attention=operand,
      weights=[operand, operand],
      output=short,
      maximum_workspace_bytes=frame.workspace_bytes(output_borrowed=true),
    )
  catch {
    @device.DeviceError::InvalidArgument(_) => ()
    e => raise e
  } noraise {
    _ => fail("short borrowed result accepted")
  }
  assert_eq(live_resources(), before)
}
