///|
async test "stage verification transports every output and retires device restore before host rollback" {
  device_fixture_lane.acquire()
  defer device_fixture_lane.release()
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let host = verification_host()
  let rank = @serial_worker.SerialDecoderRank::new(host, outputs_token=true)
  defer (rank.close() catch { _ => () })
  let ports = rank.ports()
  ports.prepare_ports(context, maximum_device_bytes=108L)
  store_verification_samples(context, ports, [7, 9])
  let (queue, function) = serial_queue(
    context,
    aot,
    stream,
    ports.input(Counts),
  )
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
  let client = @stage_control.DecoderStageClient::new(
    connection,
    limits,
    outputs_token=true,
    maximum_buffer_bytes=budget,
    timeout_millis=1000UL,
    maximum_output_rows=4,
  )
  defer (client.close() catch { _ => () })
  let state_phase = Ref(0)
  let save_ready = Ref(false)
  let restore_ready = Ref(false)
  let saved = Ref(0)
  let restored = Ref(0)
  let kept = Ref(0)
  let committed_effects = Ref(0)
  let armed = Ref(0)
  let transaction = @stage_control.StageStateTransaction::new(
    begin_save=() => {
      assert_eq(state_phase.val, 0)
      state_phase.val = 1
      saved.val += 1
    },
    poll_save=() => {
      assert_eq(state_phase.val, 1)
      if save_ready.val {
        state_phase.val = 2
      }
      save_ready.val
    },
    commit=() => {
      assert_eq(state_phase.val, 2)
      state_phase.val = 0
      kept.val += 1
    },
    begin_restore=() => {
      assert_eq(state_phase.val, 2)
      state_phase.val = 3
      restored.val += 1
    },
    poll_restore=() => {
      assert_eq(state_phase.val, 3)
      if restore_ready.val {
        state_phase.val = 0
      }
      restore_ready.val
    },
  )
  let server = @stage_control.DecoderStageServer::new(
    rank,
    peer,
    limits,
    serial_inference_limits(),
    maximum_buffer_bytes=budget,
    timeout_millis=1000UL,
    local_now_millis=0UL,
    ready_to_submit=() => true,
    arm_activation=(_, _, _) => armed.val += 1,
    state_transaction=transaction,
    committed_effect=_ => committed_effects.val += 1,
  )
  defer (server.close() catch { _ => () })
  let (_, verify, _) = serial_wire_plan(7UL)
  let enqueued = enqueues()
  client.prepare(verify, epoch=1UL, local_now_millis=0UL, tentative=true)
  for _ in 0..<512 {
    server.poll(local_now_millis=0UL)
    assert_false(client.poll(local_now_millis=0UL))
    if saved.val == 1 {
      break
    }
    @async.sleep(1)
  }
  assert_eq(saved.val, 1)
  assert_eq(armed.val, 0)
  assert_eq(enqueues(), enqueued)
  save_ready.val = true
  control_exchange(client, server)
  assert_eq(armed.val, 1)
  client.run(local_now_millis=0UL)
  control_exchange(client, server)
  assert_false(client.execution_failed())
  assert_eq(client.output_rows(), 2)
  assert_eq(client.sampled_token_at(0), 7)
  assert_eq(client.sampled_token_at(1), 9)
  try client.sampled_token() catch {
    @stage_control.StageControlError::InvalidState => ()
    error => raise error
  } noraise {
    _ => fail("vector result silently collapsed to one token")
  }
  try client.sampled_token_at(2) catch {
    @stage_control.StageControlError::InvalidState => ()
    error => raise error
  } noraise {
    _ => fail("sample accessor read beyond the live result")
  }
  assert_eq(host.processed_tokens(), 0)
  client.rollback(local_now_millis=0UL)
  for _ in 0..<512 {
    server.poll(local_now_millis=0UL)
    assert_false(client.poll(local_now_millis=0UL))
    if restored.val == 1 {
      break
    }
    @async.sleep(1)
  }
  assert_eq(restored.val, 1)
  assert_eq(host.processed_tokens(), 0)
  assert_eq(committed_effects.val, 0)
  restore_ready.val = true
  control_exchange(client, server)
  assert_eq(host.processed_tokens(), 0)
  assert_eq(state_phase.val, 0)
  // Replay only the accepted input prefix, with no stale suffix or output.
  let (_, replay, _) = serial_wire_plan(7UL, tokens=[4], produces=false)
  client.prepare(replay, epoch=2UL, local_now_millis=0UL)
  control_exchange(client, server)
  client.run(local_now_millis=0UL)
  control_exchange(client, server)
  assert_eq(client.output_rows(), 0)
  client.commit(pipeline_failed=false, local_now_millis=0UL)
  control_exchange(client, server)
  assert_eq(host.processed_tokens(), 1)
  assert_eq(saved.val, 1)
  assert_eq(committed_effects.val, 1)
  // A later all-accepted transaction retains its state, without restoration.
  store_verification_samples(context, ports, [6, 8])
  let (_, accepted, _) = serial_wire_plan(7UL, position=1, tokens=[5, 6])
  client.prepare(accepted, epoch=3UL, local_now_millis=0UL, tentative=true)
  control_exchange(client, server)
  client.run(local_now_millis=0UL)
  control_exchange(client, server)
  assert_eq(client.sampled_token_at(0), 6)
  assert_eq(client.sampled_token_at(1), 8)
  client.commit(pipeline_failed=false, local_now_millis=0UL)
  control_exchange(client, server)
  assert_eq(host.processed_tokens(), 3)
  assert_eq(saved.val, 2)
  assert_eq(restored.val, 1)
  assert_eq(kept.val, 1)
  assert_eq(state_phase.val, 0)
  assert_eq(committed_effects.val, 2)
  client.release_request(local_now_millis=0UL)
  control_exchange(client, server)
  assert_eq(host.processed_tokens(), 0)
  // An invalid later sample is failure, not an ordinary rejected draft.
  store_verification_samples(context, ports, [7, 16])
  client.prepare(verify, epoch=4UL, local_now_millis=0UL, tentative=true)
  control_exchange(client, server)
  client.run(local_now_millis=0UL)
  control_exchange(client, server)
  assert_true(client.execution_failed())
  assert_eq(client.output_rows(), 0)
  try client.rollback(local_now_millis=0UL) catch {
    @stage_control.StageControlError::InvalidState => ()
    error => raise error
  } noraise {
    _ => fail("execution failure was treated as a retryable rejection")
  }
  client.commit(pipeline_failed=false, local_now_millis=0UL)
  control_exchange(client, server)
  assert_true(host.is_faulted())
  assert_true(client.is_terminal())
  assert_true(server.is_terminal())
  assert_eq(state_phase.val, 0)
  assert_eq(committed_effects.val, 2)
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
async test "tentative execution requires a startup-bound state transaction" {
  let listener = @socket.TcpServer::TcpServer(
    @socket.Addr::parse("127.0.0.1:0"),
  )
  defer listener.close()
  let connection = @socket.Tcp::connect(listener.addr)
  let (peer, _) = listener.accept()
  let rank = @serial_worker.SerialDecoderRank::new(
    verification_host(),
    outputs_token=true,
  )
  defer rank.close()
  let limits = serial_limits()
  let budget = @stage_control.required_buffer_bytes(limits)
  let client = @stage_control.DecoderStageClient::new(
    connection,
    limits,
    outputs_token=true,
    maximum_buffer_bytes=budget,
    timeout_millis=1000UL,
    maximum_output_rows=4,
  )
  defer client.close()
  let server = @stage_control.DecoderStageServer::new(
    rank,
    peer,
    limits,
    serial_inference_limits(),
    maximum_buffer_bytes=budget,
    timeout_millis=1000UL,
    local_now_millis=0UL,
    ready_to_submit=() => false,
  )
  defer server.close()
  let (_, plan, _) = serial_wire_plan(7UL)
  client.prepare(plan, epoch=1UL, local_now_millis=0UL, tentative=true)
  try control_exchange(client, server) catch {
    @stage_control.StageControlError::InvalidMessage => ()
    error => raise error
  } noraise {
    _ => fail("tentative work ran without a state backup")
  }
  assert_true(server.is_terminal())
}

///|
async test "stage protocol cannot silently narrow all-position output to a legacy token" {
  device_fixture_lane.acquire()
  defer device_fixture_lane.release()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let host = verification_host()
  let rank = @serial_worker.SerialDecoderRank::new(host, outputs_token=true)
  defer (rank.close() catch { _ => () })
  let ports = rank.ports()
  ports.prepare_ports(context, maximum_device_bytes=108L)
  store_verification_samples(context, ports, [7, 9])
  let (queue, function) = serial_queue(
    context,
    aot,
    stream,
    ports.input(Counts),
  )
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
  let client = @stage_control.DecoderStageClient::new(
    connection,
    limits,
    outputs_token=true,
    maximum_buffer_bytes=budget,
    timeout_millis=1000UL,
  )
  defer (client.close() catch { _ => () })
  let server = @stage_control.DecoderStageServer::new(
    rank,
    peer,
    limits,
    serial_inference_limits(),
    maximum_buffer_bytes=budget,
    timeout_millis=1000UL,
    local_now_millis=0UL,
    ready_to_submit=() => true,
  )
  defer (server.close() catch { _ => () })
  let (_, plan, _) = serial_wire_plan(7UL)
  client.prepare(plan, epoch=1UL, local_now_millis=0UL)
  control_exchange(client, server)
  client.run(local_now_millis=0UL)
  try control_exchange(client, server) catch {
    @stage_control.StageControlError::InvalidMessage => ()
    error => raise error
  } noraise {
    _ => fail("legacy client consumed a truncated verification result")
  }
  assert_true(client.is_terminal())
  assert_eq(host.processed_tokens(), 0)
}
