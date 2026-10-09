///|
test "serial transfer double rejects host reads and reuse before second poll" {
  let context = @device.open_context(0)
  let allocation = context.allocate(4L)
  let pool = context.prepare_transfer_pool(
    allocation,
    host_bytes=4L,
    lanes=1,
    max_regions_per_lane=1,
  )
  pool.check_prepared()
  pool.enqueue(
    lane=0,
    host_offset=0L,
    device_offset=0L,
    bytes=4L,
    to_device=false,
  )
  pool.record(0)
  assert_false(pool.poll(0))
  let destination = FixedArray::make(4, b'\x00')
  try
    pool.read_host(destination, host_offset=0L, destination_offset=0, bytes=4)
  catch {
    @device.DeviceError::Busy(_) => ()
    error => raise error
  } noraise {
    _ => fail("read host bytes before transfer completion")
  }
  try pool.reset(0) catch {
    @device.DeviceError::InvalidArgument(_) => ()
    error => raise error
  } noraise {
    _ => fail("reused a pending lane")
  }
  assert_true(pool.poll(0))
  pool.read_host(destination, host_offset=0L, destination_offset=0, bytes=4)
  pool.reset(0)
  pool.close()
  allocation.close()
  context.close()
}

///|
test "serial worker retires input model and output events without warm allocations" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let host = serial_host()
  let worker = @serial_worker.SerialDecoderWorker::new(host)
  defer (worker.close() catch { _ => () })
  assert_eq(worker.required_device_bytes(), 68L)
  worker.prepare_ports(context, maximum_device_bytes=68L)
  let counts = worker.input(Counts)
  let samples = worker.sampled_tokens()
  context.copy_from_fixed_host(samples, [7, 0, 0, 0], 0L, 0L, 4L)
  let error = context.allocate(12L)
  defer (error.close() catch { _ => () })
  let (queue, function) = serial_queue(context, aot, stream, counts)
  defer (function.close() catch { _ => () })
  bind_serial_queue(worker, context, queue, [error])
  defer (worker.close() catch { _ => () })
  let plans = Array::makei(32, i => serial_wire_plan(i.to_uint64() + 1UL))
  let before = enqueues()
  let waits = blocking_syncs()
  begin_allocations()
  for (submitted, frame, writer) in plans {
    worker.begin(frame, writer)
    for ;; {
      if worker.progress() {
        let completion = worker.completion()
        completion.validate_for(submitted)
        assert_eq(completion.entry(0).token_id(), 7)
        break
      }
    }
    worker.release_request()
  }
  let allocations = end_allocations()
  assert_eq(allocations, 0UL)
  assert_eq(enqueues() - before, 32UL)
  assert_eq(blocking_syncs(), waits)
  let bytes = FixedArray::make(20, b'\x00')
  context.copy_to_fixed_host(counts, bytes, 0L, 0L, 20L)
  assert_eq(bytes[0], b'\x01')
  assert_eq(bytes[8], b'\x01')
  assert_eq(bytes[12], b'\x02')
  let (_, frame, writer) = serial_wire_plan(100UL)
  worker.begin(frame, writer)
  worker.close()
  worker.close()
  try writer.submit() catch {
    _ => ()
  } noraise {
    _ => fail("cancelled completion remained open")
  }
  error.close()
  function.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "serial worker publishes sparse append failures and rejects invalid samples" {
  for token in [7, -1, 16] {
    let baseline = live_resources()
    let context = @device.open_context(0)
    let stream = context.create_stream()
    let aot = context.load_module(b"fake\x00")
    let worker = @serial_worker.SerialDecoderWorker::new(serial_host())
    worker.prepare_ports(context, maximum_device_bytes=68L)
    let sample_bytes = FixedArray::make(4, b'\x00')
    let bits = token.reinterpret_as_uint()
    for i in 0..<4 {
      sample_bytes[i] = ((bits >> (i * 8)) & 255U).to_byte()
    }
    context.copy_from_fixed_host(
      worker.sampled_tokens(),
      sample_bytes,
      0L,
      0L,
      4L,
    )
    let error = context.allocate(12L)
    if token == 7 {
      context.copy_from_fixed_host(error, [1, 0, 0, 0], 0L, 8L, 4L)
    }
    let (queue, function) = serial_queue(
      context,
      aot,
      stream,
      worker.input(Counts),
    )
    bind_serial_queue(worker, context, queue, [error])
    let (submitted, frame, writer) = serial_wire_plan(7UL)
    worker.begin(frame, writer)
    for ;; {
      if worker.progress() {
        let completion = worker.completion()
        completion.validate_for(submitted)
        assert_eq(
          completion.entry(0).kind(),
          @protocol.CompletionEntryKind::FailedCompletion,
        )
        assert_eq(
          completion.entry(0).failure(),
          @protocol.WorkerFailure::ExecutionFailed,
        )
        break
      }
    }
    try worker.release_request() catch {
      @serial_frame.SerialDecoderError::Busy => ()
      error => raise error
    } noraise {
      _ => fail("failed GPU history reused")
    }
    worker.close()
    function.close()
    error.close()
    aot.close()
    stream.close()
    context.close()
    assert_eq(live_resources(), baseline)
  }
}

///|
test "serial worker memory ceiling fails before acquiring transfer resources" {
  let context = @device.open_context(0)
  let worker = @serial_worker.SerialDecoderWorker::new(serial_host())
  let baseline = live_resources()
  try worker.prepare_ports(context, maximum_device_bytes=67L) catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
    error => raise error
  } noraise {
    _ => fail("ignored request IO ceiling")
  }
  assert_eq(live_resources(), baseline)
  worker.close()
  context.close()
}
