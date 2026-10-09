///|
async test "peer EOF is clean only after acknowledged request retirement" {
  device_fixture_lane.acquire()
  defer device_fixture_lane.release()
  for released in [false, true] {
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
    rank.ports().prepare_ports(context, maximum_device_bytes=68L)
    context.copy_from_fixed_host(
      rank.ports().sampled_tokens(),
      [7, 0, 0, 0],
      0L,
      0L,
      4L,
    )
    let (queue, function) = serial_queue(
      context,
      aot,
      stream,
      rank.ports().input(Counts),
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
    control_exchange(client, server)
    client.commit(pipeline_failed=false, local_now_millis=0UL)
    control_exchange(client, server)
    if released {
      client.release_request(local_now_millis=0UL)
      control_exchange(client, server)
      assert_eq(host.processed_tokens(), 0)
    }
    client.close()
    let mut failed = false
    for _ in 0..<16 {
      server.poll(local_now_millis=0UL) catch {
        @channel.RemoteChannelError::Disconnected => {
          failed = true
          break
        }
        error => raise error
      }
      if server.is_terminal() {
        break
      }
      @async.sleep(1)
    }
    assert_eq(failed, !released)
    assert_true(server.is_terminal())
    server.close()
    rank.close()
    function.close()
    aot.close()
    stream.close()
    context.close()
    assert_eq(live_resources(), baseline)
  }
}
