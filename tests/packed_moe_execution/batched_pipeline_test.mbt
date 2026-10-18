///|
async fn batch_pipeline_exchange(
  pipeline : @batch_pipeline.BatchedTensorPipeline,
  server : @stage_control.DecoderStageServer,
) -> Unit {
  for _ in 0..<512 {
    server.poll(local_now_millis=0UL)
    if pipeline.progress(local_now_millis=0UL) {
      return
    }
    @async.sleep(1)
  }
  fail("batched whole-step publication did not retire")
}

///|
async test "batched TP publication waits for both ranks and preserves other live requests" {
  device_fixture_lane.acquire()
  defer device_fixture_lane.release()
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let local = batch_rank_fixture(outputs_token=false)
  defer (local.close() catch { _ => () })
  let terminal = batch_rank_fixture()
  defer (terminal.close() catch { _ => () })
  local.prepare_ports(
    context,
    maximum_device_bytes=local.required_device_bytes(),
  )
  terminal.prepare_ports(
    context,
    maximum_device_bytes=terminal.required_device_bytes(),
  )
  context.copy_from_fixed_host(
    terminal.sampled_tokens(),
    [7, 0, 0, 0, 8, 0, 0, 0],
    0L,
    0L,
    8L,
  )
  let retired = Ref(false)
  local.bind_executor(
    context,
    submit=_ => (),
    poll=() => retired.val,
    close=() => (),
    append_descriptors=[],
  )
  terminal.bind_executor(
    context,
    submit=_ => (),
    poll=() => true,
    close=() => (),
    append_descriptors=[],
  )
  let listener = @socket.TcpServer::TcpServer(
    @socket.Addr::parse("127.0.0.1:0"),
  )
  defer listener.close()
  let connection = @socket.Tcp::connect(listener.addr)
  let (peer, _) = listener.accept()
  let limits = batch_control_limits()
  let budget = @stage_control.required_buffer_bytes(limits)
  let remote = @stage_control.DecoderStageClient::new(
    connection,
    limits,
    outputs_token=true,
    maximum_output_rows=2,
    batched_outputs=true,
    maximum_buffer_bytes=budget,
    timeout_millis=1000UL,
  )
  defer (remote.close() catch { _ => () })
  let server = @stage_control.DecoderStageServer::new_batched(
    terminal,
    peer,
    limits,
    serial_inference_limits(),
    maximum_buffer_bytes=budget,
    timeout_millis=1000UL,
    local_now_millis=0UL,
  )
  defer (server.close() catch { _ => () })
  let host = @batch_frame.BatchedDecoderFrame::new(
    maximum_rows=4,
    maximum_sequences=2,
    cache_slots=2,
    maximum_context=16,
    vocabulary_size=16,
    model_generation=9UL,
  )
  defer (host.close() catch { _ => () })
  let pipeline = @batch_pipeline.BatchedTensorPipeline::new(
    host,
    local,
    remote,
    limits,
    serial_inference_limits(),
  )
  defer (pipeline.close() catch { _ => () })
  pipeline.release_request(99UL, 0UL, local_now_millis=0UL)
  batch_pipeline_exchange(pipeline, server)
  let (submitted, first, writer) = batch_control_plan([
    (7UL, 0UL, 0, true),
    (8UL, 0UL, 0, true),
  ])
  pipeline.begin(first, writer, epoch=1UL, local_now_millis=0UL)
  for _ in 0..<64 {
    server.poll(local_now_millis=0UL)
    assert_false(pipeline.progress(local_now_millis=0UL))
    @async.sleep(1)
  }
  try pipeline.completion() catch {
    @batch_frame.BatchFrameError::Busy => ()
    error => raise error
  } noraise {
    _ => fail("published remote tokens before local execution retired")
  }
  retired.val = true
  batch_pipeline_exchange(pipeline, server)
  let completion = pipeline.completion()
  completion.validate_for(submitted)
  assert_eq(completion.entry(0).token_id(), 7)
  assert_eq(completion.entry(1).token_id(), 8)
  assert_eq(host.processed_tokens(7UL, 0UL), 2)
  assert_eq(host.processed_tokens(8UL, 0UL), 2)
  pipeline.release_request(7UL, 0UL, local_now_millis=0UL)
  batch_pipeline_exchange(pipeline, server)
  assert_eq(host.processed_tokens(8UL, 0UL), 2)
  let (next, reused, next_writer) = batch_control_plan([
    (7UL, 1UL, 0, true),
    (8UL, 0UL, 2, true),
  ])
  pipeline.begin(reused, next_writer, epoch=2UL, local_now_millis=0UL)
  batch_pipeline_exchange(pipeline, server)
  pipeline.completion().validate_for(next)
  assert_eq(host.processed_tokens(7UL, 1UL), 2)
  assert_eq(host.processed_tokens(8UL, 0UL), 3)
  try pipeline.release_request(7UL, 0UL, local_now_millis=0UL) catch {
    @batch_frame.BatchFrameError::GenerationMismatch => ()
    error => raise error
  } noraise {
    _ => fail("stale generation released current history")
  }
  assert_eq(host.processed_tokens(7UL, 1UL), 2)
  pipeline.close()
  server.close()
  terminal.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
