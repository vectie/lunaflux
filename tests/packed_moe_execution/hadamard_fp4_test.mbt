///|
test "rotary Hadamard FP4 prepared effects do not allocate or synchronize on warm steps" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let operand = context.allocate(65536L)
  defer (operand.close() catch { _ => () })
  let frame = @hadamard_frame.RotaryHadamardFp4Frame::new(
    @hadamard_source.rotary_hadamard_fp4_source(
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
      symbol_prefix="prepared_hadamard",
    ),
  )
  defer (frame.close() catch { _ => () })
  frame.prepare(
    context,
    aot,
    counts=operand,
    positions=operand,
    input=operand,
    maximum_workspace_bytes=3072L,
  )
  assert_eq(frame.output().byte_count(), 1536L)
  assert_eq(frame.workspace_bytes(), 3072L)
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
test "learned index operand pool transform and retention compose eleven effects" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let operand = context.allocate(65536L)
  defer (operand.close() catch { _ => () })
  let pool = pool_fixture()
  defer (pool.close() catch { _ => () })
  pool.prepare(
    context,
    aot,
    counts=operand,
    positions=operand,
    reset_flag=operand,
    input=operand,
    weights=[operand, operand, operand, operand],
    maximum_device_bytes=pool.required_device_bytes(),
  )
  let transform = @hadamard_frame.RotaryHadamardFp4Frame::new(
    @hadamard_source.rotary_hadamard_fp4_source(
      @precision.RotaryHadamardFp4Precision::new(
        head_count=1,
        head_width=16,
        rotary_width=8,
        maximum_rows=2,
        maximum_positions=1024,
        frequency=Base(theta=10000.0),
        simulation_block_width=8,
        maximum_workspace_bytes=128L,
      ),
      symbol_prefix="index_pool_transform",
    ),
  )
  defer (transform.close() catch { _ => () })
  transform.prepare(
    context,
    aot,
    counts=pool.output_counts(),
    positions=pool.output_positions(),
    input=pool.output(),
    maximum_workspace_bytes=128L,
  )
  let retained = retained_fixture(3)
  defer (retained.close() catch { _ => () })
  retained.prepare(
    context,
    aot,
    counts=pool.output_counts(),
    positions=pool.output_positions(),
    reset_flag=operand,
    upstream_append=pool.append_descriptor(),
    input=transform.output(),
    maximum_device_bytes=retained.required_device_bytes(),
  )
  let launches = pool.launches().to_owned()
  launches.append(transform.launches())
  launches.append(retained.launches())
  assert_eq(launches.length(), 11)
  let queue = context.create_ordered_kernel_executor(stream, launches)
  defer (queue.abort() catch { _ => () })
  defer (queue.close() catch { _ => () })
  let before = enqueues()
  let waits = blocking_syncs()
  begin_allocations()
  for _ in 0..<32 {
    for stage in 0..<11 {
      queue.enqueue(stage)
    }
    queue.record_completion()
    while !queue.poll_completion() {
      ()
    }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - before, 352UL)
  assert_eq(blocking_syncs(), waits)
  queue.close()
  retained.close()
  transform.close()
  pool.close()
  operand.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
