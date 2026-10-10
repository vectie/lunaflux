///|
test "position-enabled worker shifts output ports and retains warmed allocation law" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  let stream = context.create_stream()
  let aot = context.load_module(b"fake\x00")
  let host = @serial_frame.SerialDecoderFrame::new(
    maximum_rows=4,
    maximum_context=12,
    vocabulary_size=16,
    model_generation=9UL,
    with_positions=true,
  )
  let worker = @serial_worker.SerialDecoderWorker::new(host)
  assert_eq(worker.required_device_bytes(), 84L)
  worker.prepare_ports(context, maximum_device_bytes=84L)
  context.copy_from_fixed_host(
    worker.sampled_tokens(),
    [11, 0, 0, 0],
    0L,
    0L,
    4L,
  )
  let positions = worker.input(Positions)
  let error = context.allocate(12L)
  let (queue, function) = serial_queue(
    context,
    aot,
    stream,
    worker.input(Counts),
  )
  bind_serial_queue(worker, context, queue, [error])
  let plans = Array::makei(32, i => serial_wire_plan(i.to_uint64() + 1UL))
  let waits = blocking_syncs()
  begin_allocations()
  for (submitted, frame, writer) in plans {
    worker.begin(frame, writer)
    for ;; {
      if worker.progress() {
        let completion = worker.completion()
        completion.validate_for(submitted)
        assert_eq(completion.entry(0).token_id(), 11)
        break
      }
    }
    worker.release_request()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(blocking_syncs(), waits)
  let bytes = FixedArray::make(16, b'\xff')
  context.copy_to_fixed_host(positions, bytes, 0L, 0L, 16L)
  assert_eq(bytes[0], b'\x00')
  assert_eq(bytes[4], b'\x01')
  // The append error moved after the sample port, not into its former slot.
  context.copy_from_fixed_host(error, [1, 0, 0, 0], 0L, 8L, 4L)
  let (submitted, frame, writer) = serial_wire_plan(100UL)
  worker.begin(frame, writer)
  for ;; {
    if worker.progress() {
      break
    }
  }
  let completion = worker.completion()
  completion.validate_for(submitted)
  assert_eq(
    completion.entry(0).kind(),
    @protocol.CompletionEntryKind::FailedCompletion,
  )
  worker.close()
  worker.close()
  function.close()
  error.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "legacy worker rejects position binding instead of indexing past prepared ports" {
  let context = @device.open_context(0)
  let worker = @serial_worker.SerialDecoderWorker::new(serial_host())
  worker.prepare_ports(context, maximum_device_bytes=68L)
  try worker.input(Positions) catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
    e => raise e
  } noraise {
    _ => fail("borrowed absent positions")
  }
  worker.close()
  context.close()
}
