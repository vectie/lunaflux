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
  let generation = @worker_protocol.ModelPlanGeneration::from_wire(1UL)
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
  let owner = start(
    b"/bin/cat",
    supervisor_limits(),
    supervisor_inference_limits(),
    supervisor_process_limits(),
    has_predecessor=false,
    predecessor_value=0UL,
  )
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
  let owner = start(
    b"/bin/cat",
    supervisor_limits(),
    supervisor_inference_limits(),
    supervisor_process_limits(),
    has_predecessor=false,
    predecessor_value=0UL,
  )
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
  let owner = start(
    b"/bin/cat",
    supervisor_limits(),
    supervisor_inference_limits(),
    supervisor_process_limits(),
    has_predecessor=false,
    predecessor_value=0UL,
  )
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
  let owner = start(
    b"/bin/cat",
    supervisor_limits(),
    supervisor_inference_limits(),
    supervisor_process_limits(),
    has_predecessor=false,
    predecessor_value=0UL,
  )
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
