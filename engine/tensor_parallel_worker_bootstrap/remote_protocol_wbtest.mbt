///|
fn bridge_remote_mailbox(
  from : @remote_mailbox.RemoteMailbox,
  to : @remote_mailbox.RemoteMailbox,
  bytes : FixedArray[Byte],
) -> Unit raise {
  if to.can_deliver() {
    match from.outgoing(bytes) {
      Some(length) => {
        to.deliver(bytes, length)
        from.sent()
      }
      None => ()
    }
  }
}

///|
fn exercise_remote_group(
  remote : @remote_startup.RemoteGroupStartup,
  admitted : TensorParallelGroupBootstrap,
  disconnect_after_accept? : Bool = false,
) -> Unit raise {
  let startup = remote.rank(0).unwrap().worker()
  let limits = @rank_group_wire.RankGroupWireLimits::new(
    max_payload_bytes=16384,
  )
  let plans = [0, 1].map(_ => {
    @worker_protocol.SchedulePlanBuffer::new(startup.worker_limits())
  })
  let group = @remote_group.RemoteGroup::new(
    remote,
    admitted,
    [plans[0].identity(), plans[1].identity()],
    limits,
    timeout_millis=10000UL,
  )
  let workers = [0, 1].map(_ => {
    @remote_mailbox.RemoteMailbox::new(limits.max_frame_bytes())
  })
  let controls = workers.map(mailbox => {
    @rank_child_control.open_remote_rank_child_control(mailbox, limits)
  })
  let bytes = FixedArray::make(limits.max_frame_bytes(), b'\x00')
  for rank in 0..<2 {
    controls[rank].begin_receive()
    bridge_remote_mailbox(group.mailbox(rank), workers[rank], bytes)
    assert_true(controls[rank].progress_receive().has_inbound())
    let inbound = controls[rank].inbound()
    inbound.copy_payload_to(bytes, destination_offset=0)
    remote.rank(rank).unwrap().verify_configure(bytes, inbound.payload_length())
    controls[rank].begin_ready(
      inbound,
      inbound.binding(),
      previous_sequence_value=0UL,
    )
    bridge_remote_mailbox(workers[rank], group.mailbox(rank), bytes)
    assert_true(controls[rank].progress_send().is_complete())
  }
  for _ in 0..<8 {
    if group.poll_startup() {
      break
    }
  }
  assert_true(group.lifecycle() == @remote_group.Ready)
  let allocator = @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=2,
      max_run_pages=2,
      max_active_references_per_page=2,
      max_cached_references_per_page=2,
    ),
  )
  plans[0].append_prefill(
    request_id=@inference.RequestId::new(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[1],
    pages=[allocator.allocate_one()],
    capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
    completion_slot=@worker_protocol.CompletionSlot::new(
      0,
      startup.worker_limits(),
    ),
  )
  let generation = @worker_protocol.ModelPlanGeneration::from_wire(
    startup.model_generation_value(),
  )
  let plan = plans[0].submit(
    sequence=@worker_protocol.PlanSequence::from_wire(1UL),
    model_plan_generation=generation,
    token_budget=1,
    None,
    generation,
  )
  let completion_owner = @worker_protocol.CompletionBuffer::new(
    startup.worker_limits(),
  )
  completion_owner.append_prefill_completed(
    slot=@worker_protocol.CompletionSlot::new(0, startup.worker_limits()),
    request_id=@inference.RequestId::new(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    processed_tokens=1,
  )
  let completion = completion_owner.submit_for(plan)
  let completion_buffer = @worker_wire.CompletionFrameBuffer::new(
    startup.worker_limits(),
    startup.inference_limits(),
  )
  let encoded = completion_buffer.encode_completion(completion)
  let response = FixedArray::make(limits.max_payload_bytes(), b'\x00')
  encoded.copy_to(response, destination_offset=0)
  group.begin(plan)
  let phases = FixedArray::make(2, 0)
  for control in controls {
    control.begin_receive()
  }
  for _ in 0..<100 {
    for rank in 0..<2 {
      bridge_remote_mailbox(group.mailbox(rank), workers[rank], bytes)
      if phases[rank] <= 1 && controls[rank].progress_receive().has_inbound() {
        let inbound = controls[rank].inbound()
        if phases[rank] == 0 {
          controls[rank].accept_submit(inbound)
          controls[rank].begin_receive()
          phases[rank] = 1
        } else {
          if rank == 0 {
            controls[rank].begin_leader_complete(
              inbound,
              response,
              payload_length=encoded.length(),
            )
          } else {
            controls[rank].begin_follower_complete(inbound)
          }
          phases[rank] = 2
        }
      }
      bridge_remote_mailbox(workers[rank], group.mailbox(rank), bytes)
      if phases[rank] == 2 && controls[rank].progress_send().is_complete() {
        controls[rank].begin_graph_telemetry_sidecar(
          @worker_wire.WorkerGraphTelemetry::without_shape(
            sequence=1UL,
            path=WorkerGraphOrderedEager,
            graph_hits=0UL,
            graph_misses=1UL,
          ),
        )
        phases[rank] = 3
      } else if phases[rank] == 3 &&
        controls[rank].progress_send().is_complete() {
        phases[rank] = 4
      }
    }
    if group.progress(plan) {
      break
    }
  }
  assert_true(group.lifecycle() == @remote_group.CompletionReady)
  group.completion_frame(1UL).validate_for(plan)
  group.accept_completion(plan, completion)
  if disconnect_after_accept {
    group.transport_lost()
    assert_true(group.lifecycle() == @remote_group.Accepted)
  }
  group.retire_after_scheduler_commit(1UL)
  assert_eq(group.previous_plan_sequence_value(), 1UL)
  if disconnect_after_accept {
    assert_true(group.lifecycle() == @remote_group.Faulted)
    assert_eq(group.take_failure().plan_sequence_value(), 0UL)
    return
  }
  group.begin_close()
  for control in controls {
    control.begin_receive()
  }
  phases.fill(0)
  for _ in 0..<100 {
    for rank in 0..<2 {
      bridge_remote_mailbox(group.mailbox(rank), workers[rank], bytes)
      if (phases[rank] == 0 || phases[rank] == 2) &&
        controls[rank].progress_receive().has_inbound() {
        let inbound = controls[rank].inbound()
        if phases[rank] == 0 {
          controls[rank].begin_drained(inbound)
        } else {
          controls[rank].begin_closed(inbound)
        }
        phases[rank] += 1
      }
      bridge_remote_mailbox(workers[rank], group.mailbox(rank), bytes)
      if (phases[rank] == 1 || phases[rank] == 3) &&
        controls[rank].progress_send().is_complete() {
        phases[rank] += 1
        if phases[rank] == 2 {
          controls[rank].begin_receive()
        }
      }
    }
    if group.progress_close() {
      break
    }
  }
  assert_true(group.lifecycle() == @remote_group.Closed)
  let failed = @remote_group.RemoteGroup::new(
    remote,
    admitted,
    [plans[0].identity(), plans[1].identity()],
    limits,
    timeout_millis=10000UL,
  )
  failed.mailbox(1).fail()
  try failed.poll_startup() catch {
    _ => ()
  } noraise {
    _ => fail("lost rank accepted")
  }
  assert_true(failed.lifecycle() == @remote_group.Faulted)
  assert_true(failed.mailbox(0).is_failed())
  assert_false(failed.recovery_cleanup_complete())
  assert_eq(
    failed.take_failure().group_generation_value(),
    remote.rank(0).unwrap().group_generation_value(),
  )
  try failed.take_failure() catch {
    _ => ()
  } noraise {
    _ => fail("failure published twice")
  }
}
