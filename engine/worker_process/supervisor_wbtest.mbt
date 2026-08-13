///|
fn supervisor_limits() -> @worker_protocol.WorkerProtocolLimits raise {
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
fn supervisor_inference_limits() -> @inference.InferenceLimits raise {
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
fn supervisor_process_limits() -> WorkerProcessLimits raise {
  WorkerProcessLimits::new(
    max_frame_bytes=4096,
    io_timeout_millis=1000,
    shutdown_timeout_millis=1000,
  )
}

///|
fn supervisor_startup() -> @worker_wire.WorkerStartupContract raise {
  @worker_wire.WorkerStartupContract::new(
    identity=@spec.ModelIdentity::new(
      @spec.ContentDigest::from_sha256(
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      ),
      @spec.PlanDigest::from_sha256(
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
      ),
    ),
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
    ),
    device_ordinal=0,
    model_generation_value=1UL,
    worker_limits=supervisor_limits(),
    inference_limits=supervisor_inference_limits(),
    has_predecessor=false,
    predecessor_value=0UL,
  )
}

///|
fn test_supervisor() -> WorkerProcessSupervisor raise {
  let worker_limits = supervisor_limits()
  let inference_limits = supervisor_inference_limits()
  let plan_a = @worker_wire.PlanFrameBuffer::new(
    worker_limits, inference_limits,
  )
  let plan_b = @worker_wire.PlanFrameBuffer::new(
    worker_limits, inference_limits,
  )
  let completion_a = @worker_wire.CompletionFrameBuffer::new(
    worker_limits, inference_limits,
  )
  let completion_b = @worker_wire.CompletionFrameBuffer::new(
    worker_limits, inference_limits,
  )
  let limits = supervisor_process_limits()
  {
    child: @process.spawn(
      b"/bin/cat",
      @process.ProcessLimits::new(
        max_frame_bytes=limits.max_frame_bytes(),
        io_timeout_millis=limits.io_timeout_millis(),
        shutdown_timeout_millis=limits.shutdown_timeout_millis(),
      ),
    ),
    startup: supervisor_startup(),
    plan_a,
    plan_b,
    completion_a,
    completion_b,
    tx_a: FixedArray::make(plan_a.capacity(), b'\x00'),
    tx_b: FixedArray::make(plan_b.capacity(), b'\x00'),
    rx: FixedArray::make(completion_a.capacity(), b'\x00'),
    state_a: 0,
    state_b: 0,
    sequence_a: 0UL,
    sequence_b: 0UL,
    last_submitted: 0UL,
    last_retired: 0UL,
    lifecycle: Ready,
  }
}

///|
fn supervisor_page() -> @page_allocator.PageId raise {
  @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=2,
      max_run_pages=2,
      max_active_references_per_page=2,
      max_cached_references_per_page=2,
    ),
  ).allocate_one()
}

///|
fn supervisor_plan(
  sequence : UInt64,
  generation? : UInt64 = 1UL,
) -> @worker_protocol.SubmittedSchedulePlan raise {
  let limits = supervisor_limits()
  let owner = @worker_protocol.SchedulePlanBuffer::new(limits)
  owner.append_prefill(
    request_id=@inference.RequestId::new(sequence),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[1, 2],
    pages=[supervisor_page()],
    capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
    completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
  )
  let generation = @worker_protocol.ModelPlanGeneration::from_wire(generation)
  owner.submit(
    sequence=@worker_protocol.PlanSequence::from_wire(sequence),
    model_plan_generation=generation,
    token_budget=2,
    if sequence == 1UL {
      None
    } else {
      Some(@worker_protocol.PlanSequence::from_wire(sequence - 1UL))
    },
    generation,
  )
}

///|
test "startup handshake rejects an incompatible child before readiness" {
  try
    prepare(b"/bin/cat", supervisor_startup(), supervisor_process_limits())
  catch {
    Invalid(Submission) => ()
    _ => fail("unexpected incompatible-child handshake error")
  } noraise {
    _ => fail("unhandshaken child became ready")
  }
}

///|
test "submission rejects a foreign loaded model generation before write" {
  let owner = test_supervisor()
  try owner.submit(supervisor_plan(1UL, generation=2UL)) catch {
    Invalid(Submission) => ()
    _ => fail("unexpected foreign-generation error")
  } noraise {
    _ => fail("foreign model generation reached the child")
  }
  assert_eq(owner.outstanding_count(), 0)
  owner.close()
}

///|
fn encoded_completion(
  plan : @worker_protocol.SubmittedSchedulePlan,
  destination : FixedArray[Byte],
) -> Int raise {
  let limits = supervisor_limits()
  let inference = supervisor_inference_limits()
  let plan_owner = @worker_wire.PlanFrameBuffer::new(limits, inference)
  let frame = plan_owner.encode_plan(plan)
  let completion_owner = @worker_wire.CompletionFrameBuffer::new(
    limits, inference,
  )
  let writer = completion_owner.writer_for(frame)
  writer.append_prefill_completed(frame.prefill_row(0))
  let completion = writer.submit()
  completion.copy_to(destination, destination_offset=0)
  completion.length()
}

///|
fn inject_completion(
  owner : WorkerProcessSupervisor,
  side : Int,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> ReceivedWorkerCompletion raise {
  let length = encoded_completion(plan, owner.rx)
  owner.accept_completion_bytes(side, plan, length)
}

///|
test "received side remains pinned until explicit retirement" {
  let owner = test_supervisor()
  let plan = supervisor_plan(1UL)
  let submitted = owner.submit(plan)
  assert_eq(submitted.sequence_value(), 1UL)
  let received = inject_completion(owner, 0, plan)
  let length = received.frame(owner).length()
  assert_eq(owner.outstanding_count(), 1)
  assert_eq(received.frame(owner).length(), length)
  assert_eq(received.frame(owner).length(), length)
  owner.retire_received(received)
  assert_eq(owner.outstanding_count(), 0)
  try received.frame(owner) catch {
    Invalid(Completion) => ()
    _ => fail("unexpected retired-frame error")
  } noraise {
    _ => fail("retired completion remained live")
  }
  owner.close()
}

///|
test "A side reuses only after ordered A/B retirement" {
  let owner = test_supervisor()
  let plan_a1 = supervisor_plan(1UL)
  let plan_b2 = supervisor_plan(2UL)
  let submission_a1 = owner.submit(plan_a1)
  let submission_b2 = owner.submit(plan_b2)
  let received_a1 = inject_completion(owner, 0, plan_a1)
  try owner.receive(submission_b2, plan_b2) catch {
    Backpressured => ()
    _ => fail("unexpected pinned-A backpressure error")
  } noraise {
    _ => fail("B response was read while A remained pinned")
  }
  owner.retire_received(received_a1)
  let plan_a3 = supervisor_plan(3UL)
  let submission_a3 = owner.submit(plan_a3)
  assert_eq(submission_a1.sequence_value(), 1UL)
  assert_eq(submission_a3.sequence_value(), 3UL)
  let received_b2 = inject_completion(owner, 1, plan_b2)
  owner.retire_received(received_b2)
  let received_a3 = inject_completion(owner, 0, plan_a3)
  owner.retire_received(received_a3)
  assert_eq(owner.outstanding_count(), 0)
  owner.close()
}

///|
test "A/B pressure and out-of-order receive fail before channel read" {
  let owner = test_supervisor()
  let plan_a = supervisor_plan(1UL)
  let plan_b = supervisor_plan(2UL)
  let a = owner.submit(plan_a)
  let b = owner.submit(plan_b)
  assert_false(owner.has_capacity())
  try owner.receive(b, plan_b) catch {
    Invalid(Sequence) => ()
    _ => fail("unexpected out-of-order response error")
  } noraise {
    _ => fail("out-of-order response reached the channel")
  }
  try owner.submit(supervisor_plan(3UL)) catch {
    Backpressured => ()
    _ => fail("unexpected A/B pressure error")
  } noraise {
    _ => fail("third outstanding plan was admitted")
  }
  assert_eq(a.sequence_value(), 1UL)
  owner.close()
}

///|
test "malformed child response permanently fails the session" {
  let owner = test_supervisor()
  let plan = supervisor_plan(1UL)
  let submission = owner.submit(plan)
  try owner.receive(submission, plan) catch {
    Transport(InvalidConfiguration) => ()
    error => fail("unexpected echoed-plan response error: \{Repr(error)}")
  } noraise {
    _ => fail("plan frame was accepted as a completion")
  }
  assert_true(owner.lifecycle() == Failed)
  owner.close()
}

///|
test "closed submissions are abandoned only in exact sequence order" {
  let owner = test_supervisor()
  let plan_a = supervisor_plan(1UL)
  let plan_b = supervisor_plan(2UL)
  let submission_a = owner.submit(plan_a)
  let submission_b = owner.submit(plan_b)
  try owner.recovery_startup_contract() catch {
    Invalid(Recovery) => ()
    _ => fail("unexpected live recovery-contract error")
  } noraise {
    _ => fail("live child published a recovery contract")
  }
  owner.close()
  try owner.abandon_submission(submission_b) catch {
    Invalid(Sequence) => ()
    _ => fail("unexpected out-of-order abandonment error")
  } noraise {
    _ => fail("later submission was abandoned first")
  }
  owner.abandon_submission(submission_a)
  try owner.recovery_startup_contract() catch {
    Invalid(Recovery) => ()
    _ => fail("unexpected outstanding recovery-contract error")
  } noraise {
    _ => fail("outstanding submission was omitted from recovery")
  }
  owner.abandon_submission(submission_b)
  let recovery = owner.recovery_startup_contract()
  assert_eq(recovery.identity(), supervisor_startup().identity())
  assert_eq(
    recovery.bootstrap_digest(),
    supervisor_startup().bootstrap_digest(),
  )
  assert_eq(recovery.model_generation_value(), 1UL)
  assert_true(recovery.has_predecessor())
  assert_eq(recovery.predecessor_value(), 2UL)
  assert_eq(owner.outstanding_count(), 0)
}

///|
test "validated completion remains retireable after child close" {
  let owner = test_supervisor()
  let plan = supervisor_plan(1UL)
  ignore(owner.submit(plan))
  let received = inject_completion(owner, 0, plan)
  owner.close()
  assert_eq(received.frame(owner).entry_count(), 1)
  owner.retire_received(received)
  let recovery = owner.recovery_startup_contract()
  assert_true(recovery.has_predecessor())
  assert_eq(recovery.predecessor_value(), 1UL)
  assert_eq(owner.outstanding_count(), 0)
}
