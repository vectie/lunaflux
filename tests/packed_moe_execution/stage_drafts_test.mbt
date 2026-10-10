///|
async test "committed draft control waits for prediction and readback without advancing KV" {
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
  rank
  .ports()
  .prepare_ports(
    context,
    maximum_device_bytes=rank.ports().required_device_bytes(),
  )
  let (queue, function) = serial_queue(
    context,
    aot,
    stream,
    rank.ports().input(Counts),
  )
  defer (function.close() catch { _ => () })
  rank.bind_executor(
    context,
    submit=rows => {
      assert_true(rows > 0 && rows <= 4)
      queue.enqueue(0)
      queue.record_completion()
    },
    poll=() => {
      if !queue.poll_completion() {
        return false
      }
      queue.reset()
      true
    },
    close=() => {
      queue.abort()
      queue.close()
    },
    append_descriptors=[],
  )
  let source = context.allocate(12L)
  defer (source.close() catch { _ => () })
  context.copy_from_fixed_host(
    source,
    [4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0],
    0L,
    0L,
    12L,
  )
  let readback = @draft_readback.DraftTokenReadback::new(
    context,
    source,
    token_count=3,
    vocabulary_size=16,
    maximum_host_bytes=12L,
  )
  defer (readback.close() catch { _ => () })
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
    maximum_draft_tokens=3,
  )
  defer (client.close() catch { _ => () })
  let predictions = Ref(0)
  let ready = Ref(false)
  let copying = Ref(false)
  let commits = Ref(0)
  let server = @stage_control.DecoderStageServer::new(
    rank,
    peer,
    limits,
    serial_inference_limits(),
    maximum_buffer_bytes=budget,
    timeout_millis=1000UL,
    local_now_millis=0UL,
    ready_to_submit=() => true,
    committed_effect=_ => commits.val += 1,
    draft_execution=@stage_control.StageDraftExecution::new(
      maximum_tokens=3,
      begin=() => {
        assert_eq(host.processed_tokens(), 2)
        predictions.val += 1
      },
      begin_seeded=seed => {
        // Output-free accepted replay has no live output count. Its committed
        // seed is a new input, not a stale sample-vector lookup.
        assert_eq(host.processed_tokens(), 5)
        assert_eq(seed, 8)
        context.copy_from_fixed_host(
          source,
          [8, 0, 0, 0, 9, 0, 0, 0, 10, 0, 0, 0],
          0L,
          0L,
          12L,
        )
        copying.val = false
        predictions.val += 1
      },
      poll=() => {
        if !ready.val {
          return false
        }
        if !copying.val {
          readback.begin()
          copying.val = true
        }
        readback.poll()
      },
      count=() => readback.token_count(),
      token_at=row => readback.token_at(row),
    ),
  )
  defer (server.close() catch { _ => () })
  try client.predict(local_now_millis=0UL) catch {
    _ => ()
  } noraise {
    _ => fail("prediction ran without a committed prefix")
  }
  let (_, prompt, _) = serial_wire_plan(7UL, produces=false)
  client.prepare(prompt, epoch=1UL, local_now_millis=0UL)
  control_exchange(client, server)
  try client.predict(local_now_millis=0UL) catch {
    _ => ()
  } noraise {
    _ => fail("prediction ran during an uncommitted request")
  }
  client.run(local_now_millis=0UL)
  control_exchange(client, server)
  client.commit(pipeline_failed=false, local_now_millis=0UL)
  control_exchange(client, server)
  assert_eq(host.processed_tokens(), 2)
  assert_eq(predictions.val, 0)
  client.predict(local_now_millis=0UL)
  for _ in 0..<8 {
    server.poll(local_now_millis=0UL)
    assert_false(client.poll(local_now_millis=0UL))
    try client.draft_count() catch {
      _ => ()
    } noraise {
      _ => fail("drafts escaped before prediction retired")
    }
    @async.sleep(1)
  }
  ready.val = true
  control_exchange(client, server)
  assert_eq(predictions.val, 1)
  assert_eq(host.processed_tokens(), 2)
  assert_eq(client.draft_count(), 3)
  for i in 0..<3 {
    assert_eq(client.draft_token_at(i), i + 4)
  }
  try client.draft_token_at(3) catch {
    _ => ()
  } noraise {
    _ => fail("draft receipt exposed padded output")
  }
  let (_, next, _) = serial_wire_plan(
    7UL,
    position=2,
    tokens=[4, 5, 6],
    produces=false,
  )
  client.prepare(next, epoch=2UL, local_now_millis=0UL)
  try client.draft_count() catch {
    _ => ()
  } noraise {
    _ => fail("prepare retained the overwritten draft receipt")
  }
  control_exchange(client, server)
  client.run(local_now_millis=0UL)
  control_exchange(client, server)
  client.commit(pipeline_failed=false, local_now_millis=0UL)
  control_exchange(client, server)
  assert_eq(host.processed_tokens(), 5)
  assert_eq(commits.val, 2)
  assert_eq(predictions.val, 1)
  try client.predict(local_now_millis=0UL, committed_seed=-1) catch {
    _ => ()
  } noraise {
    _ => fail("invalid seed transported")
  }
  client.predict(local_now_millis=0UL, committed_seed=8)
  control_exchange(client, server)
  assert_eq(client.draft_count(), 3)
  for i in 0..<3 {
    assert_eq(client.draft_token_at(i), i + 8)
  }
  assert_eq(predictions.val, 2)
  assert_eq(host.processed_tokens(), 5)
  client.release_request(local_now_millis=0UL)
  control_exchange(client, server)
  assert_eq(host.processed_tokens(), 0)
  server.close()
  client.close()
  readback.close()
  source.close()
  rank.close()
  function.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
