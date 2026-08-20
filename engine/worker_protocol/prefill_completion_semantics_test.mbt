///|
fn final_prefill_plan_buffer(
  request : UInt64,
) -> (
  @worker_protocol.SchedulePlanBuffer,
  @worker_protocol.SubmittedSchedulePlan,
) raise {
  let owner = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  owner.append_final_prefill(
    request_id=request_id(request),
    request_generation=@inference.RequestGeneration::from_wire(2UL),
    sequence_token_start=0,
    tokens=[10, 11, 12],
    pages=[allocator().allocate_one()],
    capabilities=[@plan.KernelCapabilityId::causal_attention()],
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=sampling_seed(),
    sample_index=5,
    completion_slot=slot(0),
  )
  let plan = owner.submit(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=model_generation(1UL),
    token_budget=3,
    None,
    model_generation(1UL),
  )
  (owner, plan)
}

///|
test "final prefill requires sampled-token completion" {
  let (_, plan) = final_prefill_plan_buffer(21UL)
  let row = plan.prefill_row(0)
  assert_true(row.produces_token())
  assert_eq(row.sampling(), @inference.SamplingParameters::greedy())
  assert_eq(row.sampling_seed_value(), sampling_seed().value())
  assert_eq(row.sample_index(), 5)

  let valid = @worker_protocol.CompletionBuffer::new(protocol_limits())
  valid.append_final_prefill_completed(
    slot=slot(0),
    request_id=request_id(21UL),
    request_generation=@inference.RequestGeneration::from_wire(2UL),
    processed_tokens=3,
    token_id=77,
  )
  let submitted = valid.submit_for(plan)
  assert_eq(submitted.entry(0).kind(), FinalPrefillCompletion)
  assert_eq(submitted.entry(0).processed_tokens(), 3)
  assert_eq(submitted.entry(0).token_id(), 77)

  let wrong_kind = @worker_protocol.CompletionBuffer::new(protocol_limits())
  wrong_kind.append_prefill_completed(
    slot=slot(0),
    request_id=request_id(21UL),
    request_generation=@inference.RequestGeneration::from_wire(2UL),
    processed_tokens=3,
  )
  try wrong_kind.submit_for(plan) catch {
    CompletionKindMismatch(index=0) => ()
    _ => fail("expected final-prefill kind mismatch")
  } noraise {
    _ => fail("intermediate completion satisfied final prefill")
  }
}

///|
test "intermediate prefill rejects sampled-token completion" {
  let owner = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  owner.append_prefill(
    request_id=request_id(22UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[10],
    pages=[allocator().allocate_one()],
    capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
    completion_slot=slot(0),
  )
  let plan = owner.submit(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=model_generation(1UL),
    token_budget=1,
    None,
    model_generation(1UL),
  )
  assert_false(plan.prefill_row(0).produces_token())
  let completions = @worker_protocol.CompletionBuffer::new(protocol_limits())
  completions.append_final_prefill_completed(
    slot=slot(0),
    request_id=request_id(22UL),
    request_generation=@inference.RequestGeneration::initial(),
    processed_tokens=1,
    token_id=5,
  )
  try completions.submit_for(plan) catch {
    CompletionKindMismatch(index=0) => ()
    _ => fail("expected intermediate-prefill kind mismatch")
  } noraise {
    _ => fail("sampled-token completion satisfied intermediate prefill")
  }
}

///|
test "reusable final prefill preserves flag sampling count and token" {
  let pages = allocator()
  let plan_owner = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  plan_owner.append_final_prefill(
    request_id=request_id(23UL),
    request_generation=@inference.RequestGeneration::from_wire(4UL),
    sequence_token_start=2,
    tokens=[10, 11],
    pages=[pages.allocate_one()],
    capabilities=[@plan.KernelCapabilityId::causal_attention()],
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=sampling_seed(),
    sample_index=3,
    completion_slot=slot(0),
  )
  let submitted_plan = plan_owner.submit(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=model_generation(1UL),
    token_budget=2,
    None,
    model_generation(1UL),
  )
  let row = submitted_plan.prefill_row(0)
  assert_true(row.produces_token())
  assert_eq(row.sampling(), @inference.SamplingParameters::greedy())
  assert_eq(row.sampling_seed_value(), sampling_seed().value())
  assert_eq(row.sample_index(), 3)
  let completions = @worker_protocol.CompletionBuffer::new(protocol_limits())
  completions.append_final_prefill_completed(
    slot=slot(0),
    request_id=request_id(23UL),
    request_generation=@inference.RequestGeneration::from_wire(4UL),
    processed_tokens=2,
    token_id=91,
  )
  let submitted = completions.submit_for(submitted_plan)
  let entry = submitted.entry(0)
  assert_eq(entry.kind(), FinalPrefillCompletion)
  assert_eq(entry.processed_tokens(), 2)
  assert_eq(entry.token_id(), 91)
}

///|
test "reusable intermediate prefill has no token accessor" {
  let plan_owner = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  plan_owner.append_prefill(
    request_id=request_id(26UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[10],
    pages=[allocator().allocate_one()],
    capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
    completion_slot=slot(0),
  )
  let plan = plan_owner.submit(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=model_generation(1UL),
    token_budget=1,
    None,
    model_generation(1UL),
  )
  let completions = @worker_protocol.CompletionBuffer::new(protocol_limits())
  completions.append_prefill_completed(
    slot=slot(0),
    request_id=request_id(26UL),
    request_generation=@inference.RequestGeneration::initial(),
    processed_tokens=1,
  )
  let entry = completions.submit_for(plan).entry(0)
  try entry.token_id() catch {
    CompletionKindMismatch(index=0) => ()
    _ => fail("expected intermediate prefill to have no token")
  } noraise {
    _ => fail("expected token accessor to fail")
  }
}

///|
test "scalar final-prefill draft matches reusable append semantics" {
  let pages = allocator()
  let owner = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let draft = owner.begin_final_prefill(
    request_id=request_id(24UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=sampling_seed(),
    sample_index=4,
    completion_slot=slot(0),
  )
  draft.push_token(10)
  draft.push_page(pages.allocate_one())
  draft.push_capability(@plan.KernelCapabilityId::causal_attention())
  draft.commit()
  let plan = owner.submit(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=model_generation(1UL),
    token_budget=1,
    None,
    model_generation(1UL),
  )
  assert_true(plan.prefill_row(0).produces_token())
  assert_eq(
    plan.prefill_row(0).sampling(),
    @inference.SamplingParameters::greedy(),
  )
  assert_eq(plan.prefill_row(0).sampling_seed_value(), sampling_seed().value())
  assert_eq(plan.prefill_row(0).sample_index(), 4)
}

///|
test "sampling cursor rejects out-of-range rows before mutation" {
  let owner = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  try
    owner.begin_decode(
      request_id=request_id(27UL),
      request_generation=@inference.RequestGeneration::initial(),
      sequence_token_position=0,
      sampling=@inference.SamplingParameters::greedy(),
      sampling_seed=sampling_seed(),
      sample_index=protocol_limits().max_sequence_tokens(),
      completion_slot=slot(0),
    )
  catch {
    InvalidRowDescriptor(kind=Decode, rule=InvalidSampleIndex) => ()
    _ => fail("unexpected sampling cursor error")
  } noraise {
    _ => fail("out-of-range sampling cursor was admitted")
  }
  assert_eq(owner.state(), Writable)
}
