///|
async fn control_exchange(
  client : @stage_control.DecoderStageClient,
  server : @stage_control.DecoderStageServer,
) -> Unit {
  for _ in 0..<512 {
    server.poll(local_now_millis=0UL)
    if client.poll(local_now_millis=0UL) {
      return
    }
    @async.sleep(1)
  }
  fail("control exchange did not retire")
}

///|
async test "plain stage control carries exact metadata waits for activation and commits tokens" {
  device_fixture_lane.acquire()
  defer device_fixture_lane.release()
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let host = serial_host()
  let rank = @serial_worker.SerialDecoderRank::new(host, outputs_token=true)
  defer (rank.close() catch { _ => () })
  let ports = rank.ports()
  ports.prepare_ports(context, maximum_device_bytes=68L)
  context.copy_from_fixed_host(ports.sampled_tokens(), [7, 0, 0, 0], 0L, 0L, 4L)
  let counts = ports.input(Counts)
  let (queue, function) = serial_queue(context, aot, stream, counts)
  defer (function.close() catch { _ => () })
  bind_rank_queue(rank, context, queue, [])
  let listener = @socket.TcpServer::TcpServer(
    @socket.Addr::parse("127.0.0.1:0"),
  )
  defer listener.close()
  let connection = @socket.Tcp::connect(listener.addr)
  let (peer, _) = listener.accept()
  let limits = serial_limits()
  let budget = @stage_control.required_buffer_bytes(limits)
  try
    @stage_control.DecoderStageClient::new(
      connection,
      limits,
      outputs_token=true,
      maximum_buffer_bytes=budget - 1L,
      timeout_millis=1000UL,
    )
  catch {
    @stage_control.StageControlError::InvalidConfiguration => ()
    error => raise error
  } noraise {
    _ => fail("allocated control storage beyond ceiling")
  }
  let client = @stage_control.DecoderStageClient::new(
    connection,
    limits,
    outputs_token=true,
    maximum_buffer_bytes=budget,
    timeout_millis=1000UL,
  )
  defer (client.close() catch { _ => () })
  let activation_ready = Ref(false)
  let server = @stage_control.DecoderStageServer::new(
    rank,
    peer,
    limits,
    serial_inference_limits(),
    maximum_buffer_bytes=budget,
    timeout_millis=1000UL,
    local_now_millis=0UL,
    ready_to_submit=() => activation_ready.val,
  )
  defer (server.close() catch { _ => () })
  let coordinator = serial_host()
  defer coordinator.close()
  let (submitted, plan, writer) = serial_wire_plan(7UL)
  coordinator.stage(plan, writer)
  let enqueued = enqueues()
  client.prepare(plan, epoch=1UL, local_now_millis=0UL)
  control_exchange(client, server)
  assert_eq(enqueues(), enqueued)
  let metadata = FixedArray::make(20, b'\x00')
  context.copy_to_fixed_host(counts, metadata, 0L, 0L, 20L)
  assert_eq(metadata[0], b'\x01')
  assert_eq(metadata[8], b'\x01')
  assert_eq(metadata[12], b'\x02')
  client.run(local_now_millis=0UL)
  for _ in 0..<8 {
    server.poll(local_now_millis=0UL)
    assert_false(client.poll(local_now_millis=0UL))
    @async.sleep(1)
  }
  assert_eq(enqueues(), enqueued)
  activation_ready.val = true
  control_exchange(client, server)
  assert_eq(enqueues() - enqueued, 1UL)
  assert_false(client.execution_failed())
  assert_eq(client.sampled_token(), 7)
  assert_eq(host.processed_tokens(), 0)
  let token = client.sampled_token()
  client.commit(pipeline_failed=false, local_now_millis=0UL)
  control_exchange(client, server)
  assert_eq(host.processed_tokens(), 2)
  let completion = coordinator.complete(
    sampled_token=token,
    execution_failed=false,
  )
  completion.validate_for(submitted)
  assert_eq(completion.entry(0).token_id(), 7)
  client.release_request(local_now_millis=0UL)
  control_exchange(client, server)
  assert_eq(host.processed_tokens(), 0)
  client.prepare(plan, epoch=2UL, local_now_millis=0UL)
  control_exchange(client, server)
  client.run(local_now_millis=0UL)
  control_exchange(client, server)
  client.commit(pipeline_failed=true, local_now_millis=0UL)
  control_exchange(client, server)
  assert_true(host.is_faulted())
  assert_true(server.is_terminal())
  server.close()
  client.close()
  rank.close()
  function.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
async test "stage server idle lifetime is independent of transfer timeout" {
  let listener = @socket.TcpServer::TcpServer(
    @socket.Addr::parse("127.0.0.1:0"),
  )
  defer listener.close()
  let connection = @socket.Tcp::connect(listener.addr)
  defer connection.close()
  let (peer, _) = listener.accept()
  let rank = @serial_worker.SerialDecoderRank::new(
    serial_host(),
    outputs_token=true,
  )
  defer rank.close()
  let limits = serial_limits()
  let server = @stage_control.DecoderStageServer::new(
    rank,
    peer,
    limits,
    serial_inference_limits(),
    maximum_buffer_bytes=@stage_control.required_buffer_bytes(limits),
    timeout_millis=10UL,
    local_now_millis=0UL,
    ready_to_submit=() => true,
  )
  defer server.close()
  server.poll(local_now_millis=100000UL)
  assert_false(server.is_terminal())
}

///|
async test "remote rank waiting for activation times out without submitting" {
  device_fixture_lane.acquire()
  defer device_fixture_lane.release()
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let rank = @serial_worker.SerialDecoderRank::new(
    serial_host(),
    outputs_token=true,
  )
  defer (rank.close() catch { _ => () })
  rank.ports().prepare_ports(context, maximum_device_bytes=68L)
  rank.bind_executor(
    context,
    submit=_ => fail("queue submitted before incoming activation"),
    poll=() => false,
    close=() => (),
    append_descriptors=[],
  )
  let listener = @socket.TcpServer::TcpServer(
    @socket.Addr::parse("127.0.0.1:0"),
  )
  defer listener.close()
  let connection = @socket.Tcp::connect(listener.addr)
  let (peer, _) = listener.accept()
  let limits = serial_limits()
  let budget = @stage_control.required_buffer_bytes(limits)
  let client = @stage_control.DecoderStageClient::new(
    connection,
    limits,
    outputs_token=true,
    maximum_buffer_bytes=budget,
    timeout_millis=10UL,
  )
  defer (client.close() catch { _ => () })
  let server = @stage_control.DecoderStageServer::new(
    rank,
    peer,
    limits,
    serial_inference_limits(),
    maximum_buffer_bytes=budget,
    timeout_millis=10UL,
    local_now_millis=0UL,
    ready_to_submit=() => false,
  )
  defer (server.close() catch { _ => () })
  let (_, plan, _) = serial_wire_plan(7UL)
  client.prepare(plan, epoch=1UL, local_now_millis=0UL)
  control_exchange(client, server)
  client.run(local_now_millis=0UL)
  for _ in 0..<8 {
    server.poll(local_now_millis=0UL)
    assert_false(client.poll(local_now_millis=0UL))
    @async.sleep(1)
  }
  try server.poll(local_now_millis=10UL) catch {
    @channel.RemoteChannelError::TimedOut => ()
    error => raise error
  } noraise {
    _ => fail("activation wait had no bounded lifetime")
  }
  assert_true(server.is_terminal())
  client.close()
  server.close()
  rank.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
