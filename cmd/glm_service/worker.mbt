///|
async fn run_rank(
  plan : ServicePlan,
  root : @approved_fs.ApprovedRoot,
  kernels : @approved_fs.ApprovedRoot,
  channel : @process.InheritedChannel?,
  startup : @wire.WorkerStartupContract?,
) -> Unit {
  let ordinal = if channel is Some(_) { 0 } else { 1 }
  let spec = inspect_spec(root)
  let limits = plan.worker_limits(spec)
  let inference = plan.inference(spec, plan.load_tokenizer(root, spec))
  if startup is Some(s) {
    guard s.worker_limits() == limits &&
      s.inference_limits() == inference &&
      s.model_generation_value() == 1UL else {
      fail("GLM startup capacity mismatch")
    }
  }
  let checkpoint = inspect_checkpoint(root, kernels, plan)
  let stage = plan.stage(spec, ordinal)
  defer stage.close()
  guard stage.required_device_bytes(checkpoint) <= plan.budget - plan.reserve else {
    fail("GLM preflight exceeds device budget")
  }
  let context = @device.open_context(0, runtime_interoperable=true)
  defer context.close()
  let stream = context.create_stream()
  defer stream.close()
  let aot = context.load_module(
    read_bounded(
      kernels,
      if ordinal == 0 {
        plan.module0
      } else {
        plan.module1
      },
      plan.reserve / 4L,
    ),
  )
  defer aot.close()
  let sampling = context.load_module(
    read_bounded(kernels, plan.sampling_module, plan.reserve / 8L),
  )
  defer sampling.close()
  let rank = @rank.BatchedDecoderRank::new(
    plan.frame(spec),
    outputs_token=ordinal == 1,
  )
  defer rank.close()
  let listeners = if ordinal == 1 {
    Some(bind_rank_listeners(plan.control_address, plan.rendezvous_address))
  } else {
    None
  }
  defer (if listeners is Some((a, b)) {
    a.close()
    b.close()
  })
  let (connection, rendezvous) = match listeners {
    Some((a, b)) => {
      let (c, _) = a.accept()
      let (r, _) = b.accept()
      (c, r)
    }
    None =>
      (
        @socket.Tcp::connect(@socket.Addr::parse(plan.control_address)),
        @socket.Tcp::connect(@socket.Addr::parse(plan.rendezvous_address)),
      )
  }
  defer connection.close()
  defer rendezvous.close()
  let collective = connect_collective(context, stage, rendezvous, ordinal)
  defer {
    // Graph user objects retain NCCL callbacks even when no step is in flight.
    // The execution owner first drops graphs and aborts pending communication;
    // worker DMA leases and model allocations are released only afterwards.
    stage.abort_execution()
    collective.abort()
    rank.close()
    stage.close()
  }
  prepare_rank(
    plan, spec, ordinal, context, stream, aot, sampling, root, checkpoint, stage,
    rank, collective,
  ) catch {
    error => {
      // Record the original failure before teardown can itself fail. This is
      // startup/failure diagnostics, never text on the framed result channel.
      @stdio.stderr.write(
        "GLM rank \{ordinal} preparation failed: \{Repr(error)}\n",
      )
      raise error
    }
  }
  // Ready is a two-rank statement. Keep this startup rendezvous outside the
  // token step and do not expose HTTP before the remote weights/ports exist.
  @async.with_timeout(300000, () => {
    let ready = FixedArray::make(1, b'R')
    guard rendezvous.write_once(b"R", offset=0, len=1) == 1 else {
      fail("rank readiness write")
    }
    guard rendezvous.read(ready, offset=0, max_len=1) == 1 && ready[0] == b'R' else {
      fail("rank readiness read")
    }
  })
  let clock = @clock.MonotonicClock::system()
  if channel is Some(channel) && startup is Some(startup) {
    let remote = @control.DecoderStageClient::new(
      connection,
      limits,
      outputs_token=true,
      maximum_output_rows=plan.slots,
      batched_outputs=true,
      maximum_buffer_bytes=plan.reserve / 8L,
      timeout_millis=300000UL,
    )
    defer remote.close()
    let pipeline = @pipeline.BatchedTensorPipeline::new(
      plan.frame(spec),
      rank,
      remote,
      limits,
      inference,
    )
    defer pipeline.close()
    worker_exchange(
      channel,
      startup,
      pipeline,
      clock,
      stage.replay_graph_count() > 0,
    ) catch {
      error => {
        stage.abort_execution()
        collective.abort()
        raise error
      }
    }
  } else {
    let server = @control.DecoderStageServer::new_batched(
      rank,
      connection,
      limits,
      inference,
      maximum_buffer_bytes=plan.reserve / 8L,
      timeout_millis=300000UL,
      local_now_millis=clock.now_millis(),
    )
    defer server.close()
    println("GLM batched rank 1 ready")
    try {
      while !server.is_terminal() {
        server.poll(local_now_millis=clock.now_millis())
        if !server.wait_idle_receive() {
          @async.pause()
        }
      }
    } catch {
      error => {
        stage.abort_execution()
        collective.abort()
        raise error
      }
    }
  }
}

///|
async fn worker_exchange(
  channel : @process.InheritedChannel,
  startup : @wire.WorkerStartupContract,
  pipeline : @pipeline.BatchedTensorPipeline,
  clock : @clock.MonotonicClock,
  graphs : Bool,
) -> Unit {
  let startup_owner = @wire.StartupFrameBuffer::new()
  let ready = startup_owner.encode(Ready, startup)
  let ready_bytes = FixedArray::make(startup_owner.capacity(), b'\x00')
  ready.copy_to(ready_bytes, destination_offset=0)
  channel.write_frame(ready_bytes, offset=0, byte_count=ready.length())
  let plans = @wire.PlanFrameBuffer::new(
    startup.worker_limits(),
    startup.inference_limits(),
  )
  let completions = @wire.CompletionFrameBuffer::new(
    startup.worker_limits(),
    startup.inference_limits(),
  )
  let input = FixedArray::make(plans.capacity().max(40), b'\x00')
  let input_staging = FixedArray::make(input.length(), b'\x00')
  let output = FixedArray::make(completions.capacity().max(40), b'\x00')
  let telemetry = FixedArray::make(
    @wire.worker_graph_telemetry_frame_capacity(),
    b'\x00',
  )
  let mut previous = startup.predecessor_value()
  let mut hits = 0UL
  let mut misses = 0UL
  while true {
    channel.begin_wait_frame_or_eof(input, staging=input_staging)
    let count = for ;; {
      if channel.progress_read_frame_or_eof() is Some(n) {
        break n
      }
      // All device and peer-commit effects have retired at this boundary.
      // Wait for the actual plan/retirement pipe instead of imposing a timer
      // even when the next frame arrives during that millisecond. The bounded
      // wait consumes no bytes and does not start/extend the idle-frame
      // deadline; the next progress call retains framing and EOF ownership.
      ignore(channel.wait_readable(maximum_wait_millis=1))
      @async.pause()
    }
    if count == 0 {
      break
    }
    if @wire.is_request_retirement(input, count) {
      let release = @wire.decode_request_retirement(input, count)
      guard !release.acknowledged && release.predecessor == previous else {
        fail("stale request retirement")
      }
      pipeline.release_request(
        release.request_id,
        release.generation,
        local_now_millis=clock.now_millis(),
      )
      while !pipeline.progress(local_now_millis=clock.now_millis()) {
        @async.pause()
      }
      @wire.RequestRetirement::{ ..release, acknowledged: true, }.encode(output)
      channel.write_frame(output, offset=0, byte_count=40)
      continue
    }
    let frame = plans.load(input, count)
    guard previous < 0xffffffffffffffffUL &&
      frame.sequence_value() == previous + 1UL else {
      fail("GLM worker plan sequence mismatch")
    }
    pipeline.begin(
      frame,
      completions.writer_for(frame),
      epoch=frame.sequence_value(),
      local_now_millis=clock.now_millis(),
    )
    while !pipeline.progress(local_now_millis=clock.now_millis()) {
      @async.pause()
    }
    let completion = pipeline.completion()
    completion.copy_to(output, destination_offset=0)
    channel.write_frame(output, offset=0, byte_count=completion.length())
    previous = frame.sequence_value()
    let replayed = graphs && frame.prefill_count() == 0
    if replayed {
      hits += 1UL
    } else {
      misses += 1UL
    }
    if @wire.worker_graph_telemetry_sample_required(previous) {
      let graph = @wire.WorkerGraphTelemetry::without_shape(
        sequence=previous,
        path=if replayed {
          WorkerGraphCaptured
        } else {
          WorkerGraphOrderedEager
        },
        graph_hits=hits,
        graph_misses=misses,
      )
      let length = @wire.encode_worker_graph_telemetry(graph, telemetry)
      channel.write_frame(telemetry, offset=0, byte_count=length)
    }
  }
}

///|
async fn inherited_worker() -> Unit {
  let channel = @process.open_inherited(
    @process.ProcessLimits::new(
      max_frame_bytes=67108864,
      io_timeout_millis=300000,
      shutdown_timeout_millis=300000,
    ),
  )
  let owner = @wire.StartupFrameBuffer::new()
  let input = FixedArray::make(owner.capacity(), b'\x00')
  let configured = owner.load(input, channel.read_frame(input))
  guard configured.kind() == Configure else { fail("GLM startup kind") }
  let startup = configured.contract()
  let bytes = FixedArray::make(@wire.bootstrap_source_capacity(), b'\x00')
  let source = @wire.decode_bootstrap_source(bytes, channel.read_frame(bytes))
  guard source.recipe() == PackedGlmBatchedTensorV14 &&
    source.digest() == startup.bootstrap_source_digest() else {
    fail("GLM bootstrap recipe")
  }
  let approval = FixedArray::make(
    @wire.parent_approval_attestation_capacity(),
    b'\x00',
  )
  guard @wire.decode_parent_approval_attestation_v1(
      approval,
      channel.read_frame(approval),
    )
    is None else {
    fail("GLM unexpected legacy artifact approval")
  }
  let root = @approved_fs.ApprovedRoot::open_inherited(Model)
  defer root.close()
  let kernels = @approved_fs.ApprovedRoot::open_inherited(Kernel)
  defer kernels.close()
  root.require_absolute_identity(source.roots().model_root())
  kernels.require_absolute_identity(source.roots().kernel_root())
  let plan = parse_plan(
    read_bounded(kernels, source.kernel().manifest_locator(), 65536L),
  )
  guard plan.model_root == source.roots().model_root() &&
    plan.kernel_root == source.roots().kernel_root() else {
    fail("GLM root binding mismatch")
  }
  run_rank(plan, root, kernels, Some(channel), Some(startup))
}
