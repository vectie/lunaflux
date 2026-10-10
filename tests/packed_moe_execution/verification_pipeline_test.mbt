///|
fn fixture_state_transaction(
  value : Ref[Int],
  saves : Ref[Int],
  restores : Ref[Int],
  keeps : Ref[Int],
) -> @stage_control.StageStateTransaction {
  let backup = Ref(0)
  let phase = Ref(0)
  @stage_control.StageStateTransaction::new(
    begin_save=() => {
      assert_eq(phase.val, 0)
      backup.val = value.val
      phase.val = 1
      saves.val += 1
    },
    poll_save=() => {
      assert_eq(phase.val, 1)
      phase.val = 2
      true
    },
    commit=() => {
      assert_eq(phase.val, 2)
      phase.val = 0
      keeps.val += 1
    },
    begin_restore=() => {
      assert_eq(phase.val, 2)
      phase.val = 3
      restores.val += 1
    },
    poll_restore=() => {
      assert_eq(phase.val, 3)
      value.val = backup.val
      phase.val = 0
      true
    },
  )
}

///|
async fn run_verification_pipeline_case(
  target : ArrayView[Int],
  accepted : Int,
  remaining? : Int = 10,
  stop? : Int = -1,
  failure? : Bool = false,
  fail_replay? : Bool = false,
  reuse? : Bool = false,
) -> Unit {
  device_fixture_lane.acquire()
  defer device_fixture_lane.release()
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let local_host = serial_host()
  let remote_host = verification_host()
  let ingress = @serial_worker.SerialDecoderRank::new(
    local_host,
    outputs_token=false,
  )
  defer (ingress.close() catch { _ => () })
  let egress = @serial_worker.SerialDecoderRank::new(
    remote_host,
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
  let local_state = Ref(0)
  let remote_state = Ref(0)
  let local_runs = Ref(0)
  let remote_runs = Ref(0)
  ingress.bind_executor(
    context,
    submit=rows => {
      local_state.val += rows
      local_runs.val += 1
      local_queue.enqueue(0)
      local_queue.record_completion()
    },
    poll=() => {
      if local_queue.poll_completion() {
        local_queue.reset()
        true
      } else {
        false
      }
    },
    close=() => {
      local_queue.abort()
      local_queue.close()
    },
    append_descriptors=[],
  )
  egress.bind_executor(
    context,
    submit=rows => {
      if fail_replay && remote_runs.val == 2 {
        fail("replay submission failure")
      }
      remote_state.val += rows
      remote_runs.val += 1
      remote_queue.enqueue(0)
      remote_queue.record_completion()
    },
    poll=() => {
      if remote_queue.poll_completion() {
        remote_queue.reset()
        true
      } else {
        false
      }
    },
    close=() => {
      remote_queue.abort()
      remote_queue.close()
    },
    append_descriptors=[],
  )
  // Both actual rank frontiers begin with a committed two-token prompt.
  for rank in [ingress, egress] {
    let (_, prompt, writer) = serial_wire_plan(7UL, produces=false)
    rank.begin(prompt, writer)
    while !rank.inputs_ready() {
      ()
    }
    rank.submit()
    while !rank.progress() {
      ()
    }
    rank.commit(pipeline_failed=false)
  }
  store_verification_samples(context, egress.ports(), target)
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
  let local_saves = Ref(0)
  let remote_saves = Ref(0)
  let local_restores = Ref(0)
  let remote_restores = Ref(0)
  let local_keeps = Ref(0)
  let remote_keeps = Ref(0)
  let committed_effects = Ref(0)
  let server = @stage_control.DecoderStageServer::new(
    egress,
    peer,
    limits,
    serial_inference_limits(),
    maximum_buffer_bytes=budget,
    timeout_millis=1000UL,
    local_now_millis=0UL,
    ready_to_submit=() => true,
    state_transaction=fixture_state_transaction(
      remote_state, remote_saves, remote_restores, remote_keeps,
    ),
    committed_effect=_ => committed_effects.val += 1,
  )
  defer (server.close() catch { _ => () })
  let closed_state = Ref(false)
  let coordinator = @verify_pipeline.TwoRankGreedyVerification::new(
    ingress,
    client,
    limits,
    serial_inference_limits(),
    local_state=fixture_state_transaction(
      local_state, local_saves, local_restores, local_keeps,
    ),
    close_local_state=() => {
      local_queue.abort()
      closed_state.val = true
    },
    begin_activation=(_, _, _) => (),
    poll_activation=_ => true,
    close_activation=() => (),
    stop_tokens=if stop < 0 { [] } else { [stop] },
  )
  defer (coordinator.close() catch { _ => () })
  let (_, plan, _) = serial_wire_plan(7UL, position=2, tokens=[4, 5, 6])
  coordinator.begin(
    plan,
    epoch=1UL,
    remaining_new_tokens=remaining,
    local_now_millis=0UL,
  )
  let mut retired = false
  let mut replay_failed = false
  for _ in 0..<512 {
    if fail_replay {
      server.poll(local_now_millis=0UL) catch {
        _ => {
          replay_failed = true
          break
        }
      }
    } else {
      server.poll(local_now_millis=0UL)
    }
    if coordinator.progress(local_now_millis=0UL) {
      retired = true
      break
    }
    try coordinator.result() catch {
      @stage_control.StageControlError::InvalidState => ()
      error => raise error
    } noraise {
      _ => fail("target tokens escaped before both ranks committed")
    }
    @async.sleep(1)
  }
  if fail_replay {
    assert_true(replay_failed)
    assert_false(retired)
    try coordinator.result() catch {
      @stage_control.StageControlError::InvalidState => ()
      error => raise error
    } noraise {
      _ => fail("failed replay published tentative tokens")
    }
  } else if failure {
    assert_true(retired)
    assert_true(coordinator.is_failed())
    assert_true(local_host.is_faulted())
    assert_true(remote_host.is_faulted())
    assert_eq(committed_effects.val, 0)
    assert_eq(local_restores.val, 0)
    assert_eq(remote_restores.val, 0)
  } else {
    assert_true(retired)
    assert_false(coordinator.is_failed())
    let decision = coordinator.result()
    assert_eq(decision.accepted_input_rows(), accepted)
    assert_eq(decision.emitted_token_count(), accepted)
    assert_eq(decision.next_seed(), target[accepted - 1])
    assert_eq(local_host.processed_tokens(), 2 + accepted)
    assert_eq(remote_host.processed_tokens(), 2 + accepted)
    assert_eq(local_state.val, 2 + accepted)
    assert_eq(remote_state.val, 2 + accepted)
    assert_eq(local_saves.val, 1)
    assert_eq(remote_saves.val, 1)
    assert_eq(committed_effects.val, 1)
    let rejected = accepted < 3
    assert_eq(local_restores.val, if rejected { 1 } else { 0 })
    assert_eq(remote_restores.val, if rejected { 1 } else { 0 })
    assert_eq(local_keeps.val, if rejected { 0 } else { 1 })
    assert_eq(remote_keeps.val, if rejected { 0 } else { 1 })
    assert_eq(coordinator.last_epoch(), if rejected { 2UL } else { 1UL })
    assert_eq(local_runs.val, if rejected { 3 } else { 2 })
    assert_eq(remote_runs.val, if rejected { 3 } else { 2 })
    for row in 0..<accepted {
      assert_eq(coordinator.token_at(row), target[row])
    }
    // Saved target outputs survive the replay's output-free protocol receipt.
    assert_eq(coordinator.token_at(accepted - 1), target[accepted - 1])
    if stop >= 0 {
      assert_eq(
        decision.finish_reason(),
        @greedy_verification.GreedyVerificationFinish::StopToken,
      )
    } else if remaining == accepted {
      assert_eq(
        decision.finish_reason(),
        @greedy_verification.GreedyVerificationFinish::Length,
      )
    }
  }
  if !failure && !fail_replay {
    coordinator.release_request(local_now_millis=0UL)
    try coordinator.result() catch {
      @stage_control.StageControlError::InvalidState => ()
      error => raise error
    } noraise {
      _ => fail("released request retained readable target tokens")
    }
    let mut released = false
    for _ in 0..<512 {
      server.poll(local_now_millis=0UL)
      if coordinator.progress(local_now_millis=0UL) {
        released = true
        break
      }
      @async.sleep(1)
    }
    assert_true(released)
    assert_eq(local_host.processed_tokens(), 0)
    assert_eq(remote_host.processed_tokens(), 0)
    assert_false(closed_state.val)
    if reuse {
      let resident = live_resources()
      let retained_local = local_state.val
      let retained_remote = remote_state.val
      // A fresh request prefills through the ordinary coordinator, then hands
      // the same rank/client owners back to the retained verification session.
      let pipeline = @serial_pipeline.TwoRankDecoderPipeline::new(
        serial_host(),
        ingress,
        client,
        limits,
        serial_inference_limits(),
        begin_activation=(_, _, _) => (),
        poll_activation=_ => true,
        close_activation=() => (),
      )
      defer (pipeline.close() catch { _ => () })
      let epoch = coordinator.last_epoch() + 1UL
      let (_, prompt, writer) = serial_wire_plan(
        99UL,
        tokens=[10, 11],
        produces=false,
      )
      pipeline.begin(prompt, writer, epoch~, local_now_millis=0UL)
      let mut prefilled = false
      for _ in 0..<512 {
        server.poll(local_now_millis=0UL)
        if pipeline.progress(local_now_millis=0UL) {
          prefilled = true
          break
        }
        @async.sleep(1)
      }
      assert_true(prefilled)
      assert_eq(local_host.processed_tokens(), 2)
      assert_eq(remote_host.processed_tokens(), 2)
      store_verification_samples(context, egress.ports(), [5, 9, 0])
      let (_, next, _) = serial_wire_plan(99UL, position=2, tokens=[4, 5, 6])
      coordinator.begin(
        next,
        epoch=epoch + 1UL,
        remaining_new_tokens=10,
        local_now_millis=0UL,
      )
      let mut committed = false
      for _ in 0..<512 {
        server.poll(local_now_millis=0UL)
        if coordinator.progress(local_now_millis=0UL) {
          committed = true
          break
        }
        @async.sleep(1)
      }
      assert_true(committed)
      assert_eq(coordinator.result().accepted_input_rows(), 2)
      assert_eq(coordinator.token_at(1), 9)
      assert_eq(coordinator.last_epoch(), epoch + 2UL)
      assert_eq(local_host.processed_tokens(), 4)
      assert_eq(remote_host.processed_tokens(), 4)
      assert_eq(local_state.val, retained_local + 4)
      assert_eq(remote_state.val, retained_remote + 4)
      coordinator.release_request(local_now_millis=0UL)
      let mut second_released = false
      for _ in 0..<512 {
        server.poll(local_now_millis=0UL)
        if coordinator.progress(local_now_millis=0UL) {
          second_released = true
          break
        }
        @async.sleep(1)
      }
      assert_true(second_released)
      assert_eq(local_host.processed_tokens(), 0)
      assert_eq(remote_host.processed_tokens(), 0)
      assert_false(closed_state.val)
      assert_eq(live_resources(), resident)
      // Verification released the ranks, not the ordinary coordinator's host
      // frontier. Retire that publication locally without another peer release.
      pipeline.retire_publication()
      let (_, third, writer) = serial_wire_plan(
        100UL,
        tokens=[12, 13],
        produces=false,
      )
      pipeline.begin(
        third,
        writer,
        epoch=coordinator.last_epoch() + 1UL,
        local_now_millis=0UL,
      )
      let mut third_prefilled = false
      for _ in 0..<512 {
        server.poll(local_now_millis=0UL)
        if pipeline.progress(local_now_millis=0UL) {
          third_prefilled = true
          break
        }
        @async.sleep(1)
      }
      assert_true(third_prefilled)
      assert_eq(local_host.processed_tokens(), 2)
      assert_eq(remote_host.processed_tokens(), 2)
      coordinator.close()
    }
  }
  coordinator.close()
  assert_true(closed_state.val)
  server.close()
  egress.close()
  local_function.close()
  remote_function.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
async test "two-rank verification commits all accepted inputs before exposing target tokens" {
  run_verification_pipeline_case([5, 6, 9], 3)
}

///|
async test "resident verification session survives distinct request prefill and rejection replay" {
  run_verification_pipeline_case([5, 6, 9], 3, reuse=true)
}

///|
async test "two-rank verification restores both ranks and replays exactly a rejected prefix" {
  run_verification_pipeline_case([5, 9, 0], 2)
  run_verification_pipeline_case([9, 0, 0], 1)
}

///|
async test "two-rank verification rolls back fully matched suffixes truncated by stop or length" {
  run_verification_pipeline_case([5, 6, 9], 1, remaining=1)
  run_verification_pipeline_case([5, 6, 9], 1, stop=5)
}

///|
async test "failed target execution poisons both ranks instead of ordinary rejection" {
  run_verification_pipeline_case([5, 16, 9], 0, failure=true)
}

///|
async test "failed accepted-prefix replay never exposes target tokens" {
  run_verification_pipeline_case([5, 9, 0], 2, fail_replay=true)
}
