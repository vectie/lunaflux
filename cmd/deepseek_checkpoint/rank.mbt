///|
fn prepare_rank(
  startup : Startup,
  ordinal : Int,
  context : @device.Context,
  aot : @device.Module,
  stream : @device.Stream,
  root : @approved_fs.ApprovedRoot,
  rank : @worker.SerialDecoderRank,
  stage : @deepseek.DeepSeekDecoderStage,
) -> Unit raise {
  let ports = rank.ports()
  guard ports.required_device_bytes() <= startup.reserve else {
    fail("request ports exceed the startup reserve")
  }
  ports.prepare_ports(
    context,
    maximum_device_bytes=ports.required_device_bytes(),
  )
  startup.checkpoint.with_transfer_session(root, startup.limits, checkpoint => {
    stage.prepare(
      context,
      aot,
      stream,
      checkpoint,
      root,
      ports,
      maximum_device_bytes=startup.budgets[ordinal] - startup.reserve,
      maximum_host_bytes=startup.reserve / 4L,
      chunk_bytes=1048576,
      limits=startup.limits,
    )
  })
  stage.bind_rank(rank, context)
}

///|
fn load_module(
  context : @device.Context,
  root_name : String,
  locator : String,
  reserve : Int64,
) -> @device.Module raise {
  let root = @approved_fs.ApprovedRoot::open_absolute(root_name)
  defer root.close()
  // Module bytes and all protocol/DMA storage must fit the non-weight reserve.
  let bytes = read_bounded(root, locator, reserve / 4L)
  context.load_module(bytes)
}

///|
async fn run_egress(
  startup : Startup,
  root : @approved_fs.ApprovedRoot,
  arguments : ArrayView[String],
) -> Unit {
  let context = @device.open_context(0)
  defer context.close()
  let stream = context.create_stream()
  defer stream.close()
  let aot = load_module(context, arguments[0], arguments[1], startup.reserve)
  defer aot.close()
  let stage = stage_plan(startup, 1)
  defer stage.close()
  let rank = @worker.SerialDecoderRank::new(
    new_frame(startup),
    outputs_token=true,
  )
  defer rank.close()
  prepare_rank(startup, 1, context, aot, stream, root, rank, stage)
  let prediction = if startup.dspark {
    let owner = prediction_plan(
      startup.spec,
      startup.rows,
      startup.history,
      startup.budgets[1] - startup.reserve,
    )
    errdefer owner.close()
    let required = owner.required_device_bytes(startup.checkpoint)
    startup.checkpoint.with_transfer_session(root, startup.limits, checkpoint => {
      owner.prepare(
        context,
        aot,
        stream,
        checkpoint,
        root,
        stage,
        rank.ports(),
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        maximum_device_bytes=required,
        chunk_bytes=1048576,
        limits=startup.limits,
      )
    })
    Some(owner)
  } else {
    None
  }
  defer (if prediction is Some(owner) { owner.close() })
  let control_listener = @socket.TcpServer::TcpServer(
    @socket.Addr::parse(arguments[2]),
  )
  defer control_listener.close()
  let activation_listener = @socket.TcpServer::TcpServer(
    @socket.Addr::parse(arguments[3]),
  )
  defer activation_listener.close()
  println("egress loaded; waiting for control and activation peers")
  let (connection, _) = control_listener.accept()
  defer connection.close()
  let (activation_connection, _) = activation_listener.accept()
  defer activation_connection.close()
  let edge = @deepseek.DeepSeekActivationEdge::new(
    stage,
    activation_connection,
    outgoing=false,
    maximum_host_bytes=startup.reserve / 4L,
    timeout_millis=300000UL,
  )
  defer edge.close()
  edge.prepare(context)
  let clock = @clock.MonotonicClock::system()
  let active = Ref(false)
  let ready = Ref(false)
  let prediction_blocks = Ref(0)
  let server = @control.DecoderStageServer::new(
    rank,
    connection,
    worker_limits(startup),
    inference_limits(startup),
    maximum_buffer_bytes=startup.reserve / 4L,
    timeout_millis=300000UL,
    local_now_millis=clock.now_millis(),
    ready_to_submit=() => ready.val,
    arm_activation=(epoch, rows, now) => {
      edge.begin(epoch~, live_rows=rows, local_now_millis=now)
      ready.val = false
      active.val = true
    },
    committed_effect?=if prediction is Some(owner) {
      Some(decode => {
        owner.submit(if decode { Predict } else { Prime })
        if decode {
          prediction_blocks.val += 1
        }
      })
    } else {
      None
    },
    poll_committed_effect=() => {
      match prediction {
        Some(owner) => owner.poll()
        None => true
      }
    },
  )
  defer server.close()
  // Cancellation/EOF drains through explicit owners; there is no poll counter
  // that turns a healthy idle service into a lifecycle failure.
  while !server.is_terminal() {
    let now = clock.now_millis()
    if active.val && edge.poll(local_now_millis=now) {
      active.val = false
      ready.val = true
    }
    server.poll(local_now_millis=now)
    @async.pause()
  }
  if prediction_blocks.val > 0 && prediction is Some(owner) {
    let tokens = FixedArray::make(24, b'\x00')
    let confidence = FixedArray::make(20, b'\x00')
    context.copy_to_fixed_host(owner.tokens(), tokens, 0L, 0L, 24L)
    context.copy_to_fixed_host(owner.confidence(), confidence, 0L, 0L, 20L)
    for row in 0..<6 {
      let offset = row * 4
      let token = tokens[offset].to_int() |
        (tokens[offset + 1].to_int() << 8) |
        (tokens[offset + 2].to_int() << 16) |
        (tokens[offset + 3].to_int() << 24)
      guard token >= 0 && token < startup.spec.vocabulary_size() else {
        fail("prediction returned an invalid dependent token")
      }
    }
    for row in 0..<5 {
      let offset = row * 4
      let exponent = ((confidence[offset + 3].to_int() & 0x7f) << 1) |
        (confidence[offset + 2].to_int() >> 7)
      guard exponent != 255 else {
        fail("prediction returned non-finite confidence")
      }
    }
    println(
      "dspark_prediction_blocks=\{prediction_blocks.val} tokens_i32_le=\{@crypto.bytes_to_hex_string(Bytes::from_array(tokens))} confidence_f32_le=\{@crypto.bytes_to_hex_string(Bytes::from_array(confidence))}",
    )
    println(
      "drafts are diagnostic output; no speculative accept/reject/commit is claimed",
    )
  }
}

///|
async fn run_ingress(
  startup : Startup,
  root : @approved_fs.ApprovedRoot,
  arguments : ArrayView[String],
  generate~ : Bool,
) -> Unit {
  let count = @string.parse_int(arguments[5])
  guard count > 0 && count <= startup.history else {
    fail("invalid bounded frame count")
  }
  let maximum_new_tokens = if generate {
    @string.parse_int(arguments[6])
  } else {
    1
  }
  let stop = if generate { @string.parse_int(arguments[7]) } else { -1 }
  guard maximum_new_tokens > 0 &&
    maximum_new_tokens <= startup.history &&
    stop >= -1 &&
    stop < startup.spec.vocabulary_size() else {
    fail("invalid generation length or stop token")
  }
  let output_directory = arguments[if generate { 8 } else { 6 }]
  let frame_root = @approved_fs.ApprovedRoot::open_absolute(arguments[4])
  defer frame_root.close()
  @fs.mkdir(output_directory)
  let context = @device.open_context(0)
  defer context.close()
  let stream = context.create_stream()
  defer stream.close()
  let aot = load_module(context, arguments[0], arguments[1], startup.reserve)
  defer aot.close()
  let stage = stage_plan(startup, 0)
  defer stage.close()
  let rank = @worker.SerialDecoderRank::new(
    new_frame(startup),
    outputs_token=false,
  )
  defer rank.close()
  prepare_rank(startup, 0, context, aot, stream, root, rank, stage)
  let connection = @socket.Tcp::connect(@socket.Addr::parse(arguments[2]))
  defer connection.close()
  let client = @control.DecoderStageClient::new(
    connection,
    worker_limits(startup),
    outputs_token=true,
    maximum_buffer_bytes=startup.reserve / 4L,
    timeout_millis=300000UL,
  )
  defer client.close()
  let activation_connection = @socket.Tcp::connect(
    @socket.Addr::parse(arguments[3]),
  )
  defer activation_connection.close()
  let edge = @deepseek.DeepSeekActivationEdge::new(
    stage,
    activation_connection,
    outgoing=true,
    maximum_host_bytes=startup.reserve / 4L,
    timeout_millis=300000UL,
  )
  defer edge.close()
  edge.prepare(context)
  let limits = worker_limits(startup)
  let inference = inference_limits(startup)
  let pipeline = @pipeline.TwoRankDecoderPipeline::new(
    new_frame(startup),
    rank,
    client,
    limits,
    inference,
    begin_activation=(epoch, rows, now) => {
      edge.begin(epoch~, live_rows=rows, local_now_millis=now)
    },
    poll_activation=now => edge.poll(local_now_millis=now),
    close_activation=() => edge.close(),
  )
  defer pipeline.close()
  let plans = @wire.PlanFrameBuffer::new(limits, inference)
  let completions = @wire.CompletionFrameBuffer::new(limits, inference)
  let scratch = FixedArray::make(plans.capacity(), b'\x00')
  let output = FixedArray::make(completions.capacity(), b'\x00')
  let clock = @clock.MonotonicClock::system()
  let generation : Ref[@generation.SerialGreedyGeneration?] = Ref(None)
  for index in 0..<count {
    let input = read_bounded(
      frame_root,
      "plan-\{index}.bin",
      plans.capacity().to_int64(),
    )
    scratch.fill(b'\x00')
    for i, byte in input {
      scratch[i] = byte
    }
    let plan = plans.load(scratch, input.length())
    if generate {
      guard plan.prefill_count() == 1 &&
        plan.decode_count() == 0 &&
        plan.prefill_row(0).produces_token() == (index == count - 1) else {
        fail(
          "generation requires prompt-only chunks followed by one final prefill",
        )
      }
      if index == count - 1 {
        let row = plan.prefill_row(0)
        let prompt = row.sequence_token_start() + row.token_length()
        guard maximum_new_tokens <= startup.history - prompt else {
          fail(
            "prompt and generated decode positions exceed the reserved context",
          )
        }
      }
    }
    pipeline.begin(
      plan,
      completions.writer_for(plan),
      epoch=index.to_uint64() + 1UL,
      local_now_millis=clock.now_millis(),
    )
    while !pipeline.progress(local_now_millis=clock.now_millis()) {
      @async.pause()
    }
    let completion = pipeline.completion()
    completion.copy_to(output, destination_offset=0)
    @fs.write_file(
      output_directory + "/completion-\{index}.bin",
      Bytes::from_array(output[:completion.length()].to_owned()),
      create_mode=CreateNew,
    )
    let entry = completion.entry(0)
    guard entry.kind() != @protocol.CompletionEntryKind::FailedCompletion else {
      fail("model execution returned a failed completion")
    }
    if generate && index == count - 1 {
      generation.val = Some(
        @generation.SerialGreedyGeneration::new(
          plan,
          completion,
          limits,
          inference,
          maximum_new_tokens~,
          maximum_context=startup.history,
          stop_tokens=if stop == -1 { [] } else { [stop] },
        ),
      )
    }
    if entry.kind() == @protocol.CompletionEntryKind::PrefillCompletion {
      println("frame=\{index} processed=\{entry.processed_tokens()}")
    } else {
      println(
        "frame=\{index} token=\{entry.token_id()} processed=\{entry.processed_tokens()}",
      )
    }
  }
  if generate {
    guard generation.val is Some(owner) else { fail("no completed prompt") }
    while owner.finish_reason() is Running {
      let plan = owner.next_decode()
      pipeline.begin(
        plan,
        completions.writer_for(plan),
        epoch=(count + owner.token_count()).to_uint64(),
        local_now_millis=clock.now_millis(),
      )
      while !pipeline.progress(local_now_millis=clock.now_millis()) {
        @async.pause()
      }
      owner.accept(pipeline.completion())
    }
  }
  pipeline.release_request(local_now_millis=clock.now_millis())
  while !pipeline.progress(local_now_millis=clock.now_millis()) {
    @async.pause()
  }
  if generation.val is Some(owner) {
    // Diagnostic rendering/filesystem output happens only after generation and
    // both ranks' request retirement, not on its decode submission path.
    let rendered = StringBuilder()
    for index in 0..<owner.token_count() {
      rendered <+ "\{owner.token_at(index)}\n"
    }
    @fs.write_file(
      output_directory + "/tokens.txt",
      rendered.to_string(),
      create_mode=CreateNew,
    )
    println(
      "generated=\{owner.token_count()} finish=\{Repr(owner.finish_reason())}",
    )
  }
}
