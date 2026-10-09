///|
async test "two-rank coordinator waits for DMA execution and remote commit before token publication" {
  device_fixture_lane.acquire()
  defer device_fixture_lane.release()
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let local_host = serial_host()
  let remote_host = serial_host()
  let ingress = @serial_worker.SerialDecoderRank::new(
    local_host,
    outputs_token=false,
  )
  defer (ingress.close() catch { _ => () })
  let remote_rank = @serial_worker.SerialDecoderRank::new(
    remote_host,
    outputs_token=true,
  )
  defer (remote_rank.close() catch { _ => () })
  for rank in [ingress, remote_rank] {
    rank.ports().prepare_ports(context, maximum_device_bytes=68L)
  }
  let (local_queue, local_function) = serial_queue(
    context,
    aot,
    stream,
    ingress.ports().input(Counts),
  )
  defer (local_function.close() catch { _ => () })
  let (remote_queue, remote_function) = serial_queue(
    context,
    aot,
    stream,
    remote_rank.ports().input(Counts),
  )
  defer (remote_function.close() catch { _ => () })
  let local_retired = Ref(false)
  ingress.bind_executor(
    context,
    submit=rows => {
      assert_true(rows == 1 || rows == 2)
      local_retired.val = false
      local_queue.enqueue(0)
      local_queue.record_completion()
    },
    poll=() => {
      if local_queue.poll_completion() {
        local_queue.reset()
        local_retired.val = true
        true
      } else {
        false
      }
    },
    close=() => {
      local_queue.abort()
      local_queue.close()
    },
    append_descriptors=[],
  )
  bind_rank_queue(remote_rank, context, remote_queue, [])
  let samples = remote_rank.ports().sampled_tokens()
  context.copy_from_fixed_host(samples, [7, 0, 0, 0], 0L, 0L, 4L)
  let source = context.allocate(16L)
  defer (source.close() catch { _ => () })
  let destination = context.allocate(16L)
  defer (destination.close() catch { _ => () })
  let original : FixedArray[Byte] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  ]
  context.copy_from_fixed_host(source, original, 0L, 0L, 16L)
  let listener = @socket.TcpServer::TcpServer(
    @socket.Addr::parse("127.0.0.1:0"),
  )
  defer listener.close()
  let send_connection = @socket.Tcp::connect(listener.addr)
  let (receive_connection, _) = listener.accept()
  let activation_budget = @activation.required_host_bytes(
    maximum_rows=2,
    row_bytes=8,
  )
  let sender = @activation.ActivationLink::sender(
    send_connection,
    maximum_rows=2,
    row_bytes=8,
    maximum_host_bytes=activation_budget,
    timeout_millis=1000UL,
    producer_ready=() => local_retired.val,
  )
  defer (sender.close() catch { _ => () })
  let receiver = @activation.ActivationLink::receiver(
    receive_connection,
    maximum_rows=2,
    row_bytes=8,
    maximum_host_bytes=activation_budget,
    timeout_millis=1000UL,
  )
  defer (receiver.close() catch { _ => () })
  sender.prepare(context, source)
  receiver.prepare(context, destination)
  let control_connection = @socket.Tcp::connect(listener.addr)
  let (control_peer, _) = listener.accept()
  let limits = serial_limits()
  let budget = @stage_control.required_buffer_bytes(limits)
  let client = @stage_control.DecoderStageClient::new(
    control_connection,
    limits,
    outputs_token=true,
    maximum_buffer_bytes=budget,
    timeout_millis=1000UL,
  )
  defer (client.close() catch { _ => () })
  let incoming_active = Ref(false)
  let incoming_done = Ref(false)
  let server = @stage_control.DecoderStageServer::new(
    remote_rank,
    control_peer,
    limits,
    serial_inference_limits(),
    maximum_buffer_bytes=budget,
    timeout_millis=1000UL,
    local_now_millis=0UL,
    ready_to_submit=() => incoming_done.val,
    arm_activation=(epoch, rows, now) => {
      incoming_done.val = false
      receiver.begin(epoch~, live_rows=rows, local_now_millis=now)
      incoming_active.val = true
    },
  )
  defer (server.close() catch { _ => () })
  let pipeline = @serial_pipeline.TwoRankDecoderPipeline::new(
    serial_host(),
    ingress,
    client,
    limits,
    serial_inference_limits(),
    begin_activation=(epoch, rows, now) => {
      sender.begin(epoch~, live_rows=rows, local_now_millis=now)
    },
    poll_activation=now => sender.poll(local_now_millis=now),
    close_activation=() => sender.close(),
  )
  defer (pipeline.close() catch { _ => () })
  let enqueued = enqueues()
  let generation : Ref[@serial_generation.SerialGreedyGeneration?] = Ref(None)
  let completion_owner = @wire.CompletionFrameBuffer::new(
    limits,
    serial_inference_limits(),
  )
  for step in 1..<=4 {
    let (plan, writer) = if step <= 2 {
      let (_, plan, writer) = serial_wire_plan(
        7UL,
        position=if step == 1 { 0 } else { 2 },
        tokens=[2, 3],
        produces=step == 2,
      )
      (plan, writer)
    } else {
      guard generation.val is Some(owner) else { fail("missing generation") }
      let plan = owner.next_decode()
      assert_eq(plan.token_at(0), if step == 3 { 7 } else { 9 })
      // Distinct numerical outputs prove the next input comes from the
      // preceding actual terminal completion, not a constant replay token.
      context.copy_from_fixed_host(
        samples,
        [if step == 3 { 9 } else { 11 }, 0, 0, 0],
        0L,
        0L,
        4L,
      )
      (plan, completion_owner.writer_for(plan))
    }
    pipeline.begin(plan, writer, epoch=step.to_uint64(), local_now_millis=0UL)
    let mut completed = false
    for _ in 0..<512 {
      if incoming_active.val && !incoming_done.val {
        incoming_done.val = receiver.poll(local_now_millis=0UL)
      }
      server.poll(local_now_millis=0UL)
      if pipeline.progress(local_now_millis=0UL) {
        completed = true
        break
      }
      try pipeline.completion() catch {
        @serial_frame.SerialDecoderError::Busy => ()
        error => raise error
      } noraise {
        _ => fail("published before remote commit")
      }
      @async.sleep(1)
    }
    assert_true(completed)
    let completion = pipeline.completion()
    if step > 1 {
      assert_eq(
        completion.entry(0).token_id(),
        if step == 2 {
          7
        } else if step == 3 {
          9
        } else {
          11
        },
      )
    } else {
      assert_eq(
        completion.entry(0).kind(),
        @protocol.CompletionEntryKind::PrefillCompletion,
      )
    }
    if step == 2 {
      generation.val = Some(
        @serial_generation.SerialGreedyGeneration::new(
          plan,
          completion,
          limits,
          serial_inference_limits(),
          maximum_new_tokens=3,
          maximum_context=16,
          stop_tokens=[],
        ),
      )
    } else if step > 2 {
      guard generation.val is Some(owner) else { fail("missing generation") }
      owner.accept(completion)
    }
    let processed = if step > 2 { step + 2 } else { step * 2 }
    assert_eq(local_host.processed_tokens(), processed)
    assert_eq(remote_host.processed_tokens(), processed)
    let actual = FixedArray::make(16, b'\x00')
    context.copy_to_fixed_host(destination, actual, 0L, 0L, 16L)
    assert_eq(actual, original)
  }
  guard generation.val is Some(owner) else { fail("missing generation") }
  assert_eq(owner.finish_reason(), @serial_generation.GenerationFinish::Length)
  for index, expected in [7, 9, 11] {
    assert_eq(owner.token_at(index), expected)
  }
  pipeline.release_request(local_now_millis=0UL)
  let mut released = false
  for _ in 0..<512 {
    server.poll(local_now_millis=0UL)
    if pipeline.progress(local_now_millis=0UL) {
      released = true
      break
    }
    @async.sleep(1)
  }
  assert_true(released)
  assert_eq(local_host.processed_tokens(), 0)
  assert_eq(remote_host.processed_tokens(), 0)
  assert_eq(enqueues() - enqueued, 8UL)
  // A terminal-stage numerical failure invalidates every rank's retained state.
  context.copy_from_fixed_host(samples, [255, 255, 255, 255], 0L, 0L, 4L)
  let (submitted, plan, writer) = serial_wire_plan(4UL)
  pipeline.begin(plan, writer, epoch=5UL, local_now_millis=0UL)
  let mut failed = false
  for _ in 0..<512 {
    if incoming_active.val && !incoming_done.val {
      incoming_done.val = receiver.poll(local_now_millis=0UL)
    }
    server.poll(local_now_millis=0UL)
    if pipeline.progress(local_now_millis=0UL) {
      failed = true
      break
    }
    @async.sleep(1)
  }
  assert_true(failed)
  let completion = pipeline.completion()
  completion.validate_for(submitted)
  assert_eq(
    completion.entry(0).kind(),
    @protocol.CompletionEntryKind::FailedCompletion,
  )
  assert_true(local_host.is_faulted() && remote_host.is_faulted())
  pipeline.close()
  pipeline.close()
  server.close()
  receiver.close()
  remote_rank.close()
  source.close()
  destination.close()
  local_function.close()
  remote_function.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
