///|
fn bind_rank_queue(
  rank : @serial_worker.SerialDecoderRank,
  context : @device.Context,
  queue : @device.OrderedKernelExecutor,
  descriptors : ArrayView[@device.Allocation],
) -> Unit raise {
  rank.bind_executor(
    context,
    submit=rows => {
      assert_true(rows == 1 || rows == 2)
      queue.enqueue(0)
      queue.record_completion()
    },
    poll=() => {
      if queue.poll_completion() {
        queue.reset()
        true
      } else {
        false
      }
    },
    close=() => {
      queue.abort()
      queue.close()
      for descriptor in descriptors {
        descriptor.close()
      }
    },
    append_descriptors=descriptors,
  )
}

///|
test "rank metadata upload cannot submit or publish a partial model token" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let host = serial_host()
  let rank = @serial_worker.SerialDecoderRank::new(host, outputs_token=false)
  defer (rank.close() catch { _ => () })
  let ports = rank.ports()
  ports.prepare_ports(context, maximum_device_bytes=68L)
  let (queue, function) = serial_queue(
    context,
    aot,
    stream,
    ports.input(Counts),
  )
  defer (function.close() catch { _ => () })
  bind_rank_queue(rank, context, queue, [])
  let plans = Array::makei(32, i => serial_wire_plan(i.to_uint64() + 1UL))
  let enqueued = enqueues()
  let waits = blocking_syncs()
  begin_allocations()
  for (_, frame, writer) in plans {
    rank.begin(frame, writer)
    while !rank.inputs_ready() {
      ()
    }
    rank.submit()
    while !rank.progress() {
      ()
    }
    assert_false(rank.execution_failed())
    rank.commit(pipeline_failed=false)
    rank.release_request()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - enqueued, 32UL)
  assert_eq(blocking_syncs(), waits)
  assert_eq(host.processed_tokens(), 0)
  let (_, frame, writer) = serial_wire_plan(99UL)
  rank.begin(frame, writer)
  let before = enqueues()
  try rank.submit() catch {
    @serial_frame.SerialDecoderError::Busy => ()
    error => raise error
  } noraise {
    _ => fail("submitted before metadata retirement")
  }
  while !rank.inputs_ready() {
    ()
  }
  assert_eq(enqueues(), before)
  try ports.progress() catch {
    @serial_frame.SerialDecoderError::Busy => ()
    error => raise error
  } noraise {
    _ => fail("partial rank entered whole-model execution")
  }
  rank.submit()
  while !rank.progress() {
    ()
  }
  try rank.sampled_token() catch {
    @serial_frame.SerialDecoderError::Busy => ()
    error => raise error
  } noraise {
    _ => fail("ingress returned a model token")
  }
  try ports.completion() catch {
    @serial_frame.SerialDecoderError::Busy => ()
    error => raise error
  } noraise {
    _ => fail("partial rank published completion")
  }
  try rank.begin(frame, writer) catch {
    @serial_frame.SerialDecoderError::Busy => ()
    error => raise error
  } noraise {
    _ => fail("rank reused before pipeline commit")
  }
  rank.commit(pipeline_failed=true)
  assert_true(host.is_faulted())
  rank.close()
  function.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "egress rank returns a validated token but cannot complete the model itself" {
  for index, token in [7, -1, 16, 7, 7, -1, 16, 7] {
    let baseline = live_resources()
    let context = @device.open_context(0)
    let stream = context.create_stream()
    let aot = context.load_module(b"fake\x00")
    let host = serial_host(with_positions=index >= 4)
    let rank = @serial_worker.SerialDecoderRank::new(host, outputs_token=true)
    let ports = rank.ports()
    ports.prepare_ports(
      context,
      maximum_device_bytes=ports.required_device_bytes(),
    )
    let bytes = FixedArray::make(4, b'\x00')
    let bits = token.reinterpret_as_uint()
    for i in 0..<4 {
      bytes[i] = ((bits >> (i * 8)) & 255U).to_byte()
    }
    context.copy_from_fixed_host(ports.sampled_tokens(), bytes, 0L, 0L, 4L)
    let (queue, function) = serial_queue(
      context,
      aot,
      stream,
      ports.input(Counts),
    )
    let descriptor = context.allocate(12L)
    if index % 4 == 3 {
      context.copy_from_fixed_host(descriptor, [1, 0, 0, 0], 0L, 8L, 4L)
    }
    bind_rank_queue(rank, context, queue, [descriptor])
    let (_, frame, writer) = serial_wire_plan(7UL)
    rank.begin(frame, writer)
    while !rank.inputs_ready() {
      ()
    }
    rank.submit()
    while !rank.progress() {
      ()
    }
    if index % 4 == 0 {
      assert_false(rank.execution_failed())
      assert_eq(rank.sampled_token(), 7)
    } else {
      assert_true(rank.execution_failed())
      try rank.sampled_token() catch {
        @serial_frame.SerialDecoderError::Busy => ()
        error => raise error
      } noraise {
        _ => fail("egress accepted an invalid token")
      }
    }
    rank.commit(pipeline_failed=false)
    assert_eq(host.is_faulted(), index % 4 != 0)
    rank.close()
    function.close()
    aot.close()
    stream.close()
    context.close()
    assert_eq(live_resources(), baseline)
  }
}

///|
test "rank cancellation drains staged metadata and aborts its local writer" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  let rank = @serial_worker.SerialDecoderRank::new(
    serial_host(),
    outputs_token=false,
  )
  rank.ports().prepare_ports(context, maximum_device_bytes=68L)
  let closed = Ref(false)
  rank.bind_executor(
    context,
    submit=_ => fail("cancelled metadata submitted its queue"),
    poll=() => false,
    close=() => closed.val = true,
    append_descriptors=[],
  )
  let (_, plan, writer) = serial_wire_plan(7UL)
  rank.begin(plan, writer)
  rank.close()
  rank.close()
  assert_true(closed.val)
  try writer.submit() catch {
    _ => ()
  } noraise {
    _ => fail("cancelled rank retained a completion lease")
  }
  context.close()
  assert_eq(live_resources(), baseline)
}
