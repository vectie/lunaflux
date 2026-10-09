///|
fn pool_fixture() -> @pool_frame.LearnedGatedPoolFrame raise {
  @pool_frame.LearnedGatedPoolFrame::new(
    @pool_source.learned_gated_pool_source(
      @precision.LearnedGatedPoolPrecision::new(
        input_width=8,
        feature_width=16,
        ratio=4,
        preceding_window_overlap=true,
        maximum_rows=5,
        maximum_positions=1024,
        norm_epsilon=1.0e-6,
        maximum_state_bytes=1048576L,
        maximum_workspace_bytes=1048576L,
      ),
      symbol_prefix="prepared_pool",
    ),
  )
}

///|
test "learned gated pooling owns persistent state with five warm allocation-free effects" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  let stream = context.create_stream()
  let aot = context.load_module(b"fake\x00")
  let operand = context.allocate(65536L)
  let frame = pool_fixture()
  frame.prepare(
    context,
    aot,
    counts=operand,
    positions=operand,
    reset_flag=operand,
    input=operand,
    weights=[operand, operand, operand, operand],
    maximum_device_bytes=frame.required_device_bytes(),
  )
  assert_eq(frame.launches().length(), 5)
  assert_eq(frame.output().byte_count(), 64L)
  assert_eq(frame.output_counts().byte_count(), 20L)
  assert_eq(frame.output_positions().byte_count(), 8L)
  assert_eq(frame.append_descriptor().byte_count(), 24L)
  let queue = context.create_ordered_kernel_executor(stream, frame.launches())
  let before = enqueues()
  let syncs = blocking_syncs()
  begin_allocations()
  for _ in 0..<32 {
    for stage in 0..<5 {
      queue.enqueue(stage)
    }
    queue.record_completion()
    while !queue.poll_completion() {
      ()
    }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - before, 160UL)
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
test "short learned-pool APE is rejected before private allocations" {
  let context = @device.open_context(0)
  let aot = context.load_module(b"fake\x00")
  let operand = context.allocate(65536L)
  let ape = context.allocate(511L)
  let frame = pool_fixture()
  let baseline = live_resources()
  try
    frame.prepare(
      context,
      aot,
      counts=operand,
      positions=operand,
      reset_flag=operand,
      input=operand,
      weights=[operand, operand, ape, operand],
      maximum_device_bytes=frame.required_device_bytes(),
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("short APE accepted")
  }
  assert_eq(live_resources(), baseline)
  frame.close()
  ape.close()
  operand.close()
  aot.close()
  context.close()
}
