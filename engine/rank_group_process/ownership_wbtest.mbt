///|
fn ownership_identity() -> @spec.ModelIdentity raise {
  @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    @spec.PlanDigest::from_sha256(
      "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
  )
}

///|
fn ownership_worker_limits() -> @worker_protocol.WorkerProtocolLimits raise {
  @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=2,
    max_decode_rows=2,
    max_plan_rows=2,
    max_plan_tokens=8,
    max_plan_pages=8,
    max_capabilities=8,
    max_completion_slots=2,
    max_sequence_tokens=16,
    max_token_id=31,
  )
}

///|
fn ownership_inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=8,
    max_input_tokens=8,
    max_new_tokens=8,
    max_context_tokens=16,
    max_token_id=31,
    max_stop_token_ids=1,
    max_stop_strings=1,
    max_stop_string_bytes=1,
    max_trace_bytes=1,
    max_cache_scope_bytes=1,
    max_decoded_delta_bytes=1,
    max_deadline_millis=1000UL,
    max_top_k=31,
    max_temperature=4.0,
  )
}

///|
fn ownership_prepared_for(
  plan_a : @worker_protocol.SchedulePlanBuffer,
  plan_b : @worker_protocol.SchedulePlanBuffer,
) -> PreparedRankGroupProcess raise {
  prepare_rank_group_process(
    @worker_executable_fixture.cat_path(),
    [
      RankGroupProcessRankStartup::new(ownership_configure(0)),
      RankGroupProcessRankStartup::new(ownership_configure(1)),
    ],
    [plan_a.identity(), plan_b.identity()],
    previous_plan_sequence_value=0UL,
    RankGroupProcessLimits::new(
      max_payload_bytes=4096,
      startup_io_timeout_millis=1000,
      io_timeout_millis=1000,
      shutdown_timeout_millis=10,
      collective_exchange_timeout_millis=1000,
    ),
  )
}

///|
fn ownership_prepared() -> PreparedRankGroupProcess raise {
  ownership_prepared_for(
    @worker_protocol.SchedulePlanBuffer::new(ownership_worker_limits()),
    @worker_protocol.SchedulePlanBuffer::new(ownership_worker_limits()),
  )
}

///|
fn ownership_plan(
  owner : @worker_protocol.SchedulePlanBuffer,
) -> @worker_protocol.SubmittedSchedulePlan raise {
  let allocator = @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=2,
      max_run_pages=2,
      max_active_references_per_page=2,
      max_cached_references_per_page=2,
    ),
  )
  owner.append_prefill(
    request_id=@inference.RequestId::new(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[1],
    pages=[allocator.allocate_one()],
    capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
    completion_slot=@worker_protocol.CompletionSlot::new(
      0,
      ownership_worker_limits(),
    ),
  )
  let generation = @worker_protocol.ModelPlanGeneration::first()
  owner.submit(
    sequence=@worker_protocol.PlanSequence::from_wire(1UL),
    model_plan_generation=generation,
    token_budget=1,
    None,
    generation,
  )
}

///|
fn ownership_completion(
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> @worker_protocol.SubmittedCompletion raise {
  let owner = @worker_protocol.CompletionBuffer::new(ownership_worker_limits())
  owner.append_prefill_completed(
    slot=@worker_protocol.CompletionSlot::new(0, ownership_worker_limits()),
    request_id=@inference.RequestId::new(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    processed_tokens=1,
  )
  owner.submit_for(plan)
}

///|
fn ownership_prime_wire_ready(
  supervisor : RankGroupProcessSupervisor,
) -> Unit raise {
  let payload = FixedArray::make(1, b'x')
  for rank in 0..<2 {
    let length = @rank_group_wire.encode_rank_group_wire_payload_into(
      supervisor.tx,
      supervisor.wire_limits,
      ownership_binding(rank),
      Configure,
      ParentToChild,
      None,
      0UL,
      payload,
      1,
    )
    ignore(
      supervisor.wire_transcript.accept_bytes(
        supervisor.tx,
        length,
        supervisor.wire_limits,
      ),
    )
  }
  for rank in 0..<2 {
    let length = @rank_group_wire.encode_rank_group_wire_control_into(
      supervisor.rx,
      supervisor.wire_limits,
      ownership_binding(rank),
      Ready,
      ChildToParent,
      None,
      0UL,
    )
    ignore(
      supervisor.wire_transcript.accept_bytes(
        supervisor.rx,
        length,
        supervisor.wire_limits,
      ),
    )
  }
}

///|
fn ownership_prime_wire_submitted(
  supervisor : RankGroupProcessSupervisor,
) -> Unit raise {
  let payload = FixedArray::make(1, b'p')
  for rank in 0..<2 {
    let length = @rank_group_wire.encode_rank_group_wire_payload_into(
      supervisor.tx,
      supervisor.wire_limits,
      ownership_binding(rank),
      Submit,
      ParentToChild,
      Some(A),
      1UL,
      payload,
      1,
    )
    ignore(
      supervisor.wire_transcript.accept_bytes(
        supervisor.tx,
        length,
        supervisor.wire_limits,
      ),
    )
  }
}

///|
fn ownership_prime_active(
  supervisor : RankGroupProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
  deadline_millis : UInt64,
) -> @rank_group_protocol.SubmittedRankGroupPlan raise {
  let submitted = supervisor.protocol.begin(
    plan,
    started_tick=1UL,
    deadline_tick=deadline_millis,
  )
  let slot = supervisor.slot_for_plan(plan)
  let frame = supervisor.plan_frame.encode_plan(plan)
  supervisor.active_slot_code = if slot == A { 0 } else { 1 }
  supervisor.active_sequence_value = plan.sequence_wire()
  supervisor.encoded_plan_length = frame.length()
  supervisor.active_deadline_millis = deadline_millis
  supervisor.active_frame_deadline_millis = 0L
  supervisor.exchange_epoch += 1UL
  submitted
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "parent preallocates exactly one tx and one rx plus staging" {
  let PreparedRankGroupProcess(supervisor) = ownership_prepared()
  assert_eq(supervisor.tx.length(), supervisor.wire_limits.max_frame_bytes())
  assert_eq(supervisor.rx.length(), supervisor.wire_limits.max_frame_bytes())
  assert_eq(
    supervisor.rx_staging.length(),
    supervisor.wire_limits.max_frame_bytes(),
  )
  assert_false(physical_equal(supervisor.tx, supervisor.rx))
  assert_false(physical_equal(supervisor.rx, supervisor.rx_staging))
  assert_eq(supervisor.plan_identities.length(), 2)
  assert_false(
    supervisor.plan_identities[0].same_owner(supervisor.plan_identities[1]),
  )
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "cleanup start failure stays retryable and preserves primary evidence" {
  let PreparedRankGroupProcess(supervisor) = ownership_prepared()
  let process_limits = @process.ProcessLimits::new(
    max_frame_bytes=4096,
    io_timeout_millis=1000,
    shutdown_timeout_millis=10,
  )
  let first = spawn_rank_fixture_child(
    @worker_executable_fixture.cat_path(),
    process_limits,
  )
  let second = spawn_rank_fixture_child(
    @worker_executable_fixture.cat_path(),
    process_limits,
  )
  supervisor.children[0] = LiveChild(first)
  supervisor.children[1] = LiveChild(second)
  first.begin_shutdown_maintenance()
  supervisor.failure_value = Some(TransportLost(rank=0))
  supervisor.lifecycle_value = Faulted
  supervisor.begin_cleanup()
  assert_false(supervisor.cleanup_started[0])
  assert_true(supervisor.cleanup_started[1])
  assert_eq(supervisor.cleanup_failure(), Some(RankChildCleanupFailed(rank=0)))
  assert_eq(supervisor.failure(), Some(TransportLost(rank=0)))
  assert_eq(supervisor.progress_cleanup(), CleanupPending)
  let mut first_closed = false
  for _ in 0..<100000 {
    if !first_closed {
      let progress = first.progress_shutdown_maintenance()
      if progress.is_complete() {
        first_closed = true
      }
    }
  }
  assert_true(first_closed)
  let mut group_closed = false
  for _ in 0..<100000 {
    if !group_closed {
      match supervisor.progress_cleanup() {
        CleanupComplete => group_closed = true
        CleanupStuck(rank~) => fail("unexpected stuck rank \{rank}")
        _ => ()
      }
    }
  }
  assert_true(group_closed)
  assert_eq(supervisor.failure(), Some(TransportLost(rank=0)))
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "authenticated failure remains cooperative while transport loss seals" {
  let plan_owner = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let plan_b = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let PreparedRankGroupProcess(cooperative) = ownership_prepared_for(
    plan_owner, plan_b,
  )
  let plan = ownership_plan(plan_owner)
  let submitted = ownership_prime_active(cooperative, plan, 10UL)
  cooperative.lifecycle_value = Polling
  cooperative.protocol.report_collective_failure(submitted, 1)
  cooperative.admit_cooperative_failure(CollectiveFailed(rank=1))
  assert_eq(cooperative.lifecycle(), Draining)
  assert_eq(cooperative.failure(), Some(CollectiveFailed(rank=1)))
  let cooperative_report = cooperative.take_group_failure()
  assert_eq(cooperative_report.failed_rank(), 1)
  assert_eq(cooperative_report.reason(), CollectiveFailure)

  let conservative_owner = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let conservative_b = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let PreparedRankGroupProcess(conservative) = ownership_prepared_for(
    conservative_owner, conservative_b,
  )
  let conservative_plan = ownership_plan(conservative_owner)
  ignore(ownership_prime_active(conservative, conservative_plan, 10UL))
  conservative.lifecycle_value = Polling
  try {
    let _ : Unit = conservative.active_transport_lost(conservative_plan, 1)
  } catch {
    Failed(TransportLost(rank=1)) => ()
    _ => fail("unexpected transport loss error")
  } noraise {
    _ => fail("transport loss did not terminate the group")
  }
  assert_eq(conservative.lifecycle(), Faulted)
  let conservative_report = conservative.take_group_failure()
  assert_eq(conservative_report.failed_rank(), 1)
  assert_eq(conservative_report.reason(), RankLost)
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "deadline faults abort cooperatively before conservative fallback" {
  let plan_owner = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let plan_b = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let PreparedRankGroupProcess(supervisor) = ownership_prepared_for(
    plan_owner, plan_b,
  )
  let plan = ownership_plan(plan_owner)
  let submitted = ownership_prime_active(supervisor, plan, 2UL)
  supervisor.lifecycle_value = Polling
  assert_true(supervisor.protocol.expire_if_due(submitted, 2UL))
  supervisor.admit_deadline_failure()
  assert_eq(supervisor.lifecycle(), Aborting)
  assert_eq(supervisor.failure(), Some(DeadlineExpired))
  let report = supervisor.take_group_failure()
  assert_eq(report.reason(), DeadlineExpired)
  assert_eq(report.failed_rank(), -1)
  assert_eq(supervisor.protocol.lifecycle(), Faulted)

  // A transport loss during Abort retains the deadline as primary evidence,
  // abandons the transcript, and leaves surviving ranks for explicit cleanup.
  try {
    let _ : Unit = supervisor.terminal_transport_lost(1)
  } catch {
    Failed(DeadlineExpired) => ()
    _ => fail("unexpected deadline fallback error")
  } noraise {
    _ => fail("deadline fallback did not fault the process group")
  }
  assert_eq(supervisor.lifecycle(), Faulted)
  assert_eq(supervisor.failure(), Some(DeadlineExpired))
  assert_eq(supervisor.protocol.lifecycle(), Faulted)
  assert_true(supervisor.cleanup_lost[1])
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "active host fault seals one exact inspect-then-take report" {
  let plan_owner = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let plan_b = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let PreparedRankGroupProcess(supervisor) = ownership_prepared_for(
    plan_owner, plan_b,
  )
  let plan = ownership_plan(plan_owner)
  ignore(ownership_prime_active(supervisor, plan, 20UL))
  supervisor.lifecycle_value = Polling
  supervisor.record_failure(Clock)
  assert_eq(supervisor.lifecycle(), Faulted)
  let inspected = supervisor.inspect_group_failure()
  assert_eq(inspected.reason(), HostFailure)
  assert_eq(inspected.plan_sequence_value(), 1UL)
  let taken = supervisor.take_group_failure()
  assert_eq(taken.reason(), inspected.reason())
  assert_eq(taken.plan_sequence_value(), inspected.plan_sequence_value())
  try supervisor.inspect_group_failure() catch {
    Invalid(Lifecycle) => ()
    _ => fail("unexpected post-take inspection error")
  } noraise {
    _ => fail("one-shot report remained inspectable after take")
  }
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "warmed public exchange and post-commit symbols retain exact owners" {
  let begin_owner = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let begin_b = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let PreparedRankGroupProcess(begin_supervisor) = ownership_prepared_for(
    begin_owner, begin_b,
  )
  ownership_prime_wire_ready(begin_supervisor)
  begin_supervisor.lifecycle_value = Ready
  let begin_plan = ownership_plan(begin_owner)
  try begin_supervisor.begin(begin_plan) catch {
    Failed(TransportLost(rank=0)) => ()
    _ => fail("unexpected prepared-child begin failure")
  } noraise {
    _ => fail("prepared child accepted a live group write")
  }

  let progress_owner = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let progress_b = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let PreparedRankGroupProcess(progress_supervisor) = ownership_prepared_for(
    progress_owner, progress_b,
  )
  ownership_prime_wire_ready(progress_supervisor)
  ownership_prime_wire_submitted(progress_supervisor)
  let progress_plan = ownership_plan(progress_owner)
  ignore(
    ownership_prime_active(
      progress_supervisor, progress_plan, 0xfffffffffffffff0UL,
    ),
  )
  progress_supervisor.lifecycle_value = Polling
  try progress_supervisor.progress(progress_plan) catch {
    Failed(TransportLost(rank=0)) => ()
    _ => fail("unexpected prepared-child progress failure")
  } noraise {
    _ => fail("prepared child accepted progress I/O")
  }

  let completion_owner = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let completion_b = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let PreparedRankGroupProcess(completion_supervisor) = ownership_prepared_for(
    completion_owner, completion_b,
  )
  let completion_plan = ownership_plan(completion_owner)
  let completion_submitted = ownership_prime_active(
    completion_supervisor, completion_plan, 10UL,
  )
  completion_supervisor.protocol.accept_follower_ack(
    completion_submitted,
    completion_submitted.lease(completion_plan, 1).follower_ack(),
  )
  completion_supervisor.lifecycle_value = CompletionReady
  let received = completion_supervisor.received_completion()
  let accepted = completion_supervisor.accept_completion(
    received,
    completion_plan,
    ownership_completion(completion_plan),
  )
  assert_eq(completion_supervisor.lifecycle(), CompletionAccepted)
  let retired_plan = completion_owner.retire(completion_plan)
  completion_owner.reset(retired_plan)
  completion_supervisor.retire_after_scheduler_commit(accepted)
  assert_eq(completion_supervisor.lifecycle(), Ready)
}
