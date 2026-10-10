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
  defer close_rank_context(context, "egress")
  let stream = context.create_stream()
  defer stream.close()
  let aot = load_module(context, arguments[0], arguments[1], startup.reserve)
  defer aot.close()
  let stage = stage_plan(startup, 1)
  defer stage.close()
  let rank = @worker.SerialDecoderRank::new(
    new_frame(startup, verification=startup.dspark),
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
  let backup = if startup.dspark {
    Some(
      stage.prepare_state_snapshot(
        context,
        stream,
        maximum_backup_bytes=stage.state_backup_bytes(),
      ),
    )
  } else {
    None
  }
  defer (if backup is Some(owner) {
    edge.close()
    if prediction is Some(predictor) {
      predictor.close()
    }
    stage.drain()
    owner.close_after_retirement()
  })
  let readback = match prediction {
    Some(owner) =>
      Some(
        @draft_readback.DraftTokenReadback::new(
          context,
          owner.borrow_draft_output(),
          token_count=startup.spec.dspark_block_size() + 1,
          vocabulary_size=startup.spec.vocabulary_size(),
          maximum_host_bytes=startup.reserve / 4L,
        ),
      )
    None => None
  }
  defer (if readback is Some(owner) { owner.close() })
  let clock = @clock.MonotonicClock::system()
  let active = Ref(false)
  let ready = Ref(false)
  let prediction_blocks = Ref(0)
  let reading_drafts = Ref(false)
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
      Some(_ => owner.submit(Prime))
    } else {
      None
    },
    poll_committed_effect=() => {
      match prediction {
        Some(owner) => owner.poll()
        None => true
      }
    },
    state_transaction?=backup.map(owner => state_transaction(owner)),
    draft_execution?=match (prediction, readback) {
      (Some(predictor), Some(output)) =>
        Some(
          @control.StageDraftExecution::new(
            maximum_tokens=startup.spec.dspark_block_size() + 1,
            begin=() => fail("DSpark requires an explicit committed seed"),
            begin_seeded=seed => {
              reading_drafts.val = false
              predictor.submit_from_committed(seed)
              prediction_blocks.val += 1
            },
            poll=() => {
              if !reading_drafts.val {
                if !predictor.poll() {
                  return false
                }
                output.begin()
                reading_drafts.val = true
              }
              output.poll()
            },
            count=() => output.token_count(),
            token_at=row => output.token_at(row),
          ),
        )
      _ => None
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
  println("dspark_prediction_blocks=\{prediction_blocks.val}")
}

///|
async fn run_ingress(
  startup : Startup,
  root : @approved_fs.ApprovedRoot,
  arguments : ArrayView[String],
  generate~ : Bool,
  speculate? : Bool = true,
  sequence? : Bool = false,
) -> Unit {
  let speculative_enabled = startup.dspark && speculate
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
  let limits = worker_limits(startup)
  let inference = inference_limits(startup)
  let prepared = if generate {
    let prompts = if sequence {
      prepare_prompt_sequence(
        frame_root,
        count,
        limits,
        inference,
        maximum_new_tokens~,
        maximum_context=startup.history,
        maximum_bytes=startup.reserve / 4L,
      )
    } else {
      ReadOnlyArray::from_array([
        @prompt_file.load_generation_prompt(
          frame_root,
          count,
          limits,
          inference,
          maximum_new_tokens~,
          maximum_context=startup.history,
          maximum_bytes=startup.reserve / 4L,
          model_generation=1UL,
        ),
      ])
    }
    for index, prompt in prompts {
      println(
        "request_prepared: index=\{index} input_tokens=\{prompt.input_token_count()} maximum_new_tokens=\{prompt.maximum_new_tokens()} context=\{startup.history}",
      )
    }
    Some(prompts)
  } else {
    None
  }
  if prepared is Some(_) {
    frame_root.close()
  }
  @fs.mkdir(output_directory)
  let context = @device.open_context(0)
  defer close_rank_context(context, "ingress")
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
    maximum_output_rows=if startup.dspark { startup.rows } else { 1 },
    maximum_draft_tokens=if startup.dspark {
      startup.spec.dspark_block_size() + 1
    } else {
      0
    },
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
  let backup = if startup.dspark {
    Some(
      stage.prepare_state_snapshot(
        context,
        stream,
        maximum_backup_bytes=stage.state_backup_bytes(),
      ),
    )
  } else {
    None
  }
  defer (if backup is Some(owner) {
    edge.close()
    stage.drain()
    owner.close_after_retirement()
  })
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
    close_activation=() => {
      edge.close()
      if backup is Some(saved) {
        stage.drain()
        saved.close_after_retirement()
      }
    },
  )
  defer pipeline.close()
  let verifier = if speculative_enabled {
    guard backup is Some(saved) else { fail("missing verification snapshot") }
    Some(
      prepare_verification_session(
        startup,
        rank,
        client,
        stage,
        edge,
        saved,
        stop~,
      ),
    )
  } else {
    None
  }
  defer (if verifier is Some(owner) { owner.close() })
  let clock = @clock.MonotonicClock::system()
  match prepared {
    Some(prompts) =>
      // The model, rank, activation edge and connection owners outlive every
      // request. Each completed request retires before the next starts at zero.
      for index, prompt in prompts; next_epoch = 1UL {
        let destination = if sequence {
          let path = output_directory + "/request-\{index}"
          @fs.mkdir(path)
          path
        } else {
          output_directory
        }
        let next = execute_ingress_request(
          startup,
          frame_root,
          Some(prompt),
          prompt.frame_count(),
          pipeline,
          client,
          verifier,
          clock,
          generate~,
          speculative_enabled~,
          maximum_new_tokens~,
          stop~,
          output_directory=destination,
          first_epoch=next_epoch,
        )
        println("request_retired: index=\{index} next_epoch=\{next}")
        continue next
      }
    None =>
      ignore(
        execute_ingress_request(
          startup,
          frame_root,
          None,
          count,
          pipeline,
          client,
          verifier,
          clock,
          generate~,
          speculative_enabled~,
          maximum_new_tokens~,
          stop~,
          output_directory~,
          first_epoch=1UL,
        ),
      )
  }
}
