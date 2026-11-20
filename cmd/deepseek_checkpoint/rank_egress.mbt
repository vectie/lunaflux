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
  // The head stores one vocabulary row per physical token row. Copy the full
  // retired allocation: row zero is not generally the row sampled at the end
  // of a chunked prefill. The offline caller supplies the exact live-row index.
  let logit_bytes = if arguments.length() == 5 {
    stage.borrow_text_logits().byte_count().to_int()
  } else {
    0
  }
  let logit_capture = if arguments.length() == 5 {
    Some(
      context.prepare_transfer_pool(
        stage.borrow_text_logits(),
        host_bytes=logit_bytes.to_int64(),
        lanes=1,
        max_regions_per_lane=1,
      ),
    )
  } else {
    None
  }
  defer (if logit_capture is Some(pool) {
    pool.drain(0)
    pool.close()
  })
  // The diagnostic reads DSpark's existing target-layer means after EOF.
  // No extra graph node, AOT source, layer snapshot or token-step copy.
  let prediction_capture = if logit_capture is Some(_) &&
    stage.prediction_input() is Some(input) {
    Some(
      (
        context.prepare_transfer_pool(
          input,
          host_bytes=input.byte_count(),
          lanes=1,
          max_regions_per_lane=1,
        ),
        input.byte_count().to_int(),
      ),
    )
  } else {
    None
  }
  defer (if prediction_capture is Some((pool, _)) {
    pool.drain(0)
    pool.close()
  })
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
  if logit_capture is Some(pool) {
    capture_retired_logits(pool, logit_bytes, arguments[4])
    if prediction_capture is Some((pool, bytes)) {
      capture_retired_logits(
        pool,
        bytes,
        arguments[4],
        filename="prediction-inputs.bf16",
        create_directory=false,
      )
      println(
        "retired_prediction_layers=\{Repr(stage.prediction_capture_layers())}",
      )
    }
  }
}
