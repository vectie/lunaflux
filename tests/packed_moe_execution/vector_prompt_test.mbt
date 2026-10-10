///|
async test "ordinary prompt publication takes the last live output of a vector terminal rank" {
  device_fixture_lane.acquire()
  defer device_fixture_lane.release()
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let ingress = @serial_worker.SerialDecoderRank::new(
    verification_host(),
    outputs_token=false,
  )
  defer (ingress.close() catch { _ => () })
  let egress = @serial_worker.SerialDecoderRank::new(
    verification_host(),
    outputs_token=true,
  )
  defer (egress.close() catch { _ => () })
  for rank in [ingress, egress] {
    rank
    .ports()
    .prepare_ports(
      context,
      maximum_device_bytes=rank.ports().required_device_bytes(),
    )
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
    egress.ports().input(Counts),
  )
  defer (remote_function.close() catch { _ => () })
  bind_rank_queue(ingress, context, local_queue, [])
  bind_rank_queue(egress, context, remote_queue, [])
  store_verification_samples(context, egress.ports(), [5, 6])
  let listener = @socket.TcpServer::TcpServer(
    @socket.Addr::parse("127.0.0.1:0"),
  )
  defer listener.close()
  let connection = @socket.Tcp::connect(listener.addr)
  let (peer, _) = listener.accept()
  let limits = serial_limits()
  let inference = serial_inference_limits()
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
  let server = @stage_control.DecoderStageServer::new(
    egress,
    peer,
    limits,
    inference,
    maximum_buffer_bytes=budget,
    timeout_millis=1000UL,
    local_now_millis=0UL,
    ready_to_submit=() => true,
  )
  defer (server.close() catch { _ => () })
  let scalar = @serial_frame.SerialDecoderFrame::new(
    maximum_rows=4,
    maximum_context=16,
    vocabulary_size=16,
    model_generation=9UL,
    with_positions=true,
  )
  let pipeline = @serial_pipeline.TwoRankDecoderPipeline::new(
    scalar,
    ingress,
    client,
    limits,
    inference,
    begin_activation=(_, _, _) => (),
    poll_activation=_ => true,
    close_activation=() => (),
  )
  defer (pipeline.close() catch { _ => () })
  let (_, prompt, writer) = serial_wire_plan(7UL)
  pipeline.begin(prompt, writer, epoch=1UL, local_now_millis=0UL)
  let mut retired = false
  for _ in 0..<512 {
    server.poll(local_now_millis=0UL)
    if pipeline.progress(local_now_millis=0UL) {
      retired = true
      break
    }
    @async.sleep(1)
  }
  assert_true(retired)
  let completion = pipeline.completion()
  assert_eq(completion.entry(0).token_id(), 6)
  assert_eq(completion.greedy_token_for(prompt), 6)
  pipeline.release_request(local_now_millis=0UL)
  retired = false
  for _ in 0..<512 {
    server.poll(local_now_millis=0UL)
    if pipeline.progress(local_now_millis=0UL) {
      retired = true
      break
    }
    @async.sleep(1)
  }
  assert_true(retired)
  pipeline.close()
  server.close()
  egress.close()
  local_function.close()
  remote_function.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
