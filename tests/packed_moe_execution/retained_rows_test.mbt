///|
fn retained_fixture(capacity : Int) -> @retained_frame.RetainedRowsFrame raise {
  @retained_frame.RetainedRowsFrame::new(
    @retained_source.retained_rows_source(
      @precision.RetainedRowsPrecision::new(
        word_width=16,
        position_stride=4,
        maximum_input_rows=2,
        maximum_stored_rows=capacity,
        maximum_state_bytes=1048576L,
        maximum_workspace_bytes=12L,
      ),
      symbol_prefix="prepared_retained",
    ),
  )
}

///|
test "learned pool transform and retained append use ten warm allocation-free effects" {
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
  let transform = compact_cache_fixture()
  defer (transform.close() catch { _ => () })
  transform.prepare(
    context,
    aot,
    counts=pool.output_counts(),
    positions=pool.output_positions(),
    input=pool.output(),
    maximum_workspace_bytes=64L,
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
  assert_eq(retained.required_device_bytes(), 136L)
  assert_eq(retained.values().byte_count(), 96L)
  assert_eq(retained.counts().byte_count(), 20L)
  assert_eq(retained.append_descriptor().byte_count(), 12L)
  let launches = pool.launches().to_owned()
  launches.append(transform.launches())
  launches.append(retained.launches())
  assert_eq(launches.length(), 10)
  let queue = context.create_ordered_kernel_executor(stream, launches)
  defer (queue.abort() catch { _ => () })
  defer (queue.close() catch { _ => () })
  let before = enqueues()
  let waits = blocking_syncs()
  begin_allocations()
  for _ in 0..<32 {
    for stage in 0..<10 {
      queue.enqueue(stage)
    }
    queue.record_completion()
    while !queue.poll_completion() {
      ()
    }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - before, 320UL)
  assert_eq(blocking_syncs(), waits)
  queue.close()
  retained.close()
  retained.close()
  transform.close()
  pool.close()
  operand.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "retained prepare rejects short error ports before allocation and supports zero logical capacity" {
  let context = @device.open_context(0)
  defer context.close()
  let aot = context.load_module(b"fake\x00")
  defer aot.close()
  let operand = context.allocate(65536L)
  defer operand.close()
  let short = context.allocate(11L)
  defer short.close()
  let baseline = live_resources()
  let invalid = retained_fixture(3)
  try
    invalid.prepare(
      context,
      aot,
      counts=operand,
      positions=operand,
      reset_flag=operand,
      upstream_append=short,
      input=operand,
      maximum_device_bytes=invalid.required_device_bytes(),
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("short error descriptor accepted")
  }
  invalid.close()
  assert_eq(live_resources(), baseline)
  let empty = retained_fixture(0)
  empty.prepare(
    context,
    aot,
    counts=operand,
    positions=operand,
    reset_flag=operand,
    upstream_append=operand,
    input=operand,
    maximum_device_bytes=empty.required_device_bytes(),
  )
  assert_eq(empty.values().byte_count(), 32L)
  assert_eq(empty.required_device_bytes(), 72L)
  empty.close()
  assert_eq(live_resources(), baseline)
}
