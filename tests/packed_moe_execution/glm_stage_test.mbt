///|
async fn exercise_glm_stage(
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  limits : @streaming.StreamingSafetensorsLimits,
  weights : Int64,
) -> Unit {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let rank = @serial_worker.SerialDecoderRank::new(
    @serial_frame.SerialDecoderFrame::new(
      maximum_rows=32,
      maximum_context=64,
      vocabulary_size=@glm53_spec.Glm53FlashSpec::official().vocabulary_size(),
      model_generation=1UL,
    ),
    outputs_token=false,
  )
  defer (rank.close() catch { _ => () })
  let ports = rank.ports()
  ports.prepare_ports(
    context,
    maximum_device_bytes=ports.required_device_bytes(),
  )
  let stage = @glm53_packed_expert_upload.GlmDecoderStage::new(
    @glm53_spec.Glm53FlashSpec::official(),
    @pipeline.DecoderStageRange::new(layer_count=45, first_layer=0, end_layer=1),
    maximum_rows=32,
    maximum_history=64,
    index_layer_norm_epsilon=1.0e-5,
    maximum_layer_state_bytes=10000000L,
    maximum_layer_workspace_bytes=20000000L,
    maximum_layer_expert_bytes=5000000000L,
    maximum_text_workspace_bytes=0L,
    accuracy=@precision.Accuracy::new(absolute=0.125, relative=0.125),
  )
  defer (stage.close() catch { _ => () })
  let total = stage.required_device_bytes(checkpoint)
  assert_true(total > weights)
  assert_true(total < weights + 20000000L)
  let before = live_resources()
  try
    stage.prepare(
      context,
      aot,
      stream,
      checkpoint,
      root,
      ports,
      maximum_device_bytes=total - 1L,
      chunk_bytes=1048573,
      limits~,
    )
  catch {
    @precision.PrecisionPlanError::MemoryBudgetExceeded => ()
    e => raise e
  } noraise {
    _ => fail("stage allocated before aggregate budget check")
  }
  assert_eq(live_resources(), before)
  let foreign = @serial_worker.SerialDecoderWorker::new(
    @serial_frame.SerialDecoderFrame::new(
      maximum_rows=16,
      maximum_context=64,
      vocabulary_size=@glm53_spec.Glm53FlashSpec::official().vocabulary_size(),
      model_generation=1UL,
    ),
  )
  defer foreign.close()
  try
    stage.prepare(
      context,
      aot,
      stream,
      checkpoint,
      root,
      foreign,
      maximum_device_bytes=total,
      chunk_bytes=1048573,
      limits~,
    )
  catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
    e => raise e
  } noraise {
    _ => fail("different row geometry bound to prepared stage")
  }
  assert_eq(live_resources(), before)
  stage.prepare(
    context,
    aot,
    stream,
    checkpoint,
    root,
    ports,
    maximum_device_bytes=total,
    chunk_bytes=1048573,
    limits~,
  )
  assert_eq(stage.launch_count(), 27)
  assert_eq(stage.append_descriptors().length(), 0)
  assert_eq(stage.input_residual().byte_count(), 1048576L)
  assert_false(stage.input_residual().same_storage(stage.output_residual()))
  try stage.submit(live_rows=33) catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue) => ()
    e => raise e
  } noraise {
    _ => fail("stage accepted a row count beyond its plan")
  }
  assert_eq(stage.input_residual().byte_count(), 1048576L)
  // The fixture intentionally has no norm/head. Only ingress weights are bound.
  try stage.bind_worker(ports, context) catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
    e => raise e
  } noraise {
    _ => fail("partial stage published a whole-model token")
  }
  let queued = enqueues()
  let waits = blocking_syncs()
  begin_allocations()
  for _ in 0..<32 {
    stage.submit(live_rows=32)
    while !stage.poll() {
      ()
    }
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - queued, 864UL)
  assert_eq(blocking_syncs(), waits)
  let listener = @socket.TcpServer::TcpServer(
    @socket.Addr::parse("127.0.0.1:0"),
  )
  defer listener.close()
  let client = @socket.Tcp::connect(listener.addr)
  let (peer, _) = listener.accept()
  let edge = @glm53_packed_expert_upload.GlmActivationEdge::new(
    stage,
    client,
    outgoing=true,
    maximum_host_bytes=5000000L,
    timeout_millis=1000UL,
  )
  defer (edge.close() catch { _ => () })
  edge.prepare(context)
  let incoming = context.allocate(1048576L)
  defer (incoming.close() catch { _ => () })
  let receiver = @activation.ActivationLink::receiver(
    peer,
    maximum_rows=32,
    row_bytes=32768,
    maximum_host_bytes=5000000L,
    timeout_millis=1000UL,
  )
  defer (receiver.close() catch { _ => () })
  receiver.prepare(context, incoming)
  stage.bind_rank(rank, context)
  let (_, plan, writer) = serial_wire_plan(7UL, model_generation=1UL)
  rank.begin(plan, writer)
  while !rank.inputs_ready() {
    ()
  }
  rank.submit()
  try edge.begin(epoch=1UL, live_rows=3, local_now_millis=0UL) catch {
    @activation.ActivationLinkError::InvalidState => ()
    e => raise e
  } noraise {
    _ => fail("transferred inactive producer rows")
  }
  edge.begin(epoch=1UL, live_rows=2, local_now_millis=0UL)
  receiver.begin(epoch=1UL, live_rows=2, local_now_millis=0UL)
  try stage.close() catch {
    @device.DeviceError::Busy(OrderedExecutorClose) => ()
    e => raise e
  } noraise {
    _ => fail("closed a transfer-leased stage")
  }
  let mut sent = false
  let mut received = false
  let mut retired = false
  for _ in 0..<128 {
    if !retired {
      retired = rank.progress()
    }
    if !sent {
      sent = edge.poll(local_now_millis=0UL)
      if !sent {
        try stage.submit(live_rows=1) catch {
          @device.DeviceError::Busy(OrderedExecutorEnqueue) => ()
          e => raise e
        } noraise {
          _ => fail("overwrote a pending activation transfer")
        }
      }
    }
    if !received {
      received = receiver.poll(local_now_millis=0UL)
    }
    if sent && received && retired {
      break
    }
    @async.sleep(1)
  }
  assert_true(sent && received && retired)
  assert_false(rank.execution_failed())
  rank.commit(pipeline_failed=false)
  try edge.begin(epoch=2UL, live_rows=2, local_now_millis=0UL) catch {
    @activation.ActivationLinkError::InvalidState => ()
    error => raise error
  } noraise {
    _ => fail("transferred the same completed submission as a different step")
  }
  receiver.close()
  incoming.close()
  stage.submit(live_rows=1)
  edge.begin(epoch=2UL, live_rows=1, local_now_millis=1UL)
  edge.close()
  stage.close()
  stage.close()
  rank.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
async test "GLM ingress stage streams only embedding and its dense decoder into one queue" {
  exercise_glm_checkpoint_owner(with_dense_block=true, with_decoder_stage=true)
}
