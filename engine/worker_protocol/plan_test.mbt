///|
test "limits and monotonic identities reject illegal values" {
  try
    @worker_protocol.WorkerProtocolLimits::new(
      max_prefill_rows=2,
      max_decode_rows=2,
      max_plan_rows=1,
      max_plan_tokens=4,
      max_plan_pages=4,
      max_capabilities=4,
      max_completion_slots=2,
      max_sequence_tokens=4,
      max_token_id=8,
    )
  catch {
    InvalidLimit(PlanRows) => ()
    _ => fail("expected invalid row limit")
  } noraise {
    _ => fail("invalid row envelope was accepted")
  }
  try @worker_protocol.PlanSequence::from_wire(0UL) catch {
    InvalidIdentity(PlanSequenceIdentity) => ()
    _ => fail("expected invalid sequence")
  } noraise {
    _ => fail("zero plan sequence was accepted")
  }
  let maximum = plan_sequence(0xffffffffffffffffUL)
  try maximum.next() catch {
    IdentityOverflow(PlanSequenceIdentity) => ()
    _ => fail("expected sequence overflow")
  } noraise {
    _ => fail("maximum plan sequence advanced")
  }
  assert_true(plan_sequence(2UL).immediately_follows(plan_sequence(1UL)))
  assert_false(plan_sequence(3UL).immediately_follows(plan_sequence(1UL)))
  assert_false(
    plan_sequence(1UL).immediately_follows(plan_sequence(0xffffffffffffffffUL)),
  )
}

///|
test "view and completion slot bounds remain distinct" {
  try
    @worker_protocol.TokenRange::new(start=0, length=0, protocol_limits())
  catch {
    InvalidView(table=Tokens, rule=EmptyRange) => ()
    _ => fail("expected empty token range rejection")
  } noraise {
    _ => fail("empty token range was accepted")
  }
  try @worker_protocol.CompletionSlot::new(8, protocol_limits()) catch {
    InvalidCompletionSlot => ()
    _ => fail("expected completion-slot rejection")
  } noraise {
    _ => fail("out-of-range completion slot was accepted")
  }
}

///|
test "fixed plan submission validates totals sequence and model generation" {
  let owner = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  append_mixed_buffer(owner, allocator())
  try
    owner.submit(
      sequence=@worker_protocol.PlanSequence::first(),
      model_plan_generation=model_generation(1UL),
      token_budget=2,
      None,
      model_generation(1UL),
    )
  catch {
    TokenBudgetMismatch => ()
    _ => fail("expected exact token-budget rejection")
  } noraise {
    _ => fail("wrong token budget was accepted")
  }
  assert_eq(owner.state(), Writable)
  try
    owner.submit(
      sequence=plan_sequence(2UL),
      model_plan_generation=model_generation(1UL),
      token_budget=3,
      None,
      model_generation(1UL),
    )
  catch {
    StalePlanSequence => ()
    _ => fail("expected skipped-sequence rejection")
  } noraise {
    _ => fail("skipped plan sequence was accepted")
  }
  try
    owner.submit(
      sequence=@worker_protocol.PlanSequence::first(),
      model_plan_generation=model_generation(2UL),
      token_budget=3,
      None,
      model_generation(1UL),
    )
  catch {
    StaleModelPlanGeneration => ()
    _ => fail("expected stale-model rejection")
  } noraise {
    _ => fail("stale model generation was accepted")
  }
  let submitted = owner.submit(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=model_generation(1UL),
    token_budget=3,
    None,
    model_generation(1UL),
  )
  assert_eq(submitted.row_count(), 2)
}

///|
test "fixed plan duplicate rejection leaves canonical suffix reusable" {
  let owner = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let pages = allocator()
  owner.append_prefill(
    request_id=request_id(4UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[1],
    pages=[pages.allocate_one()],
    capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
    completion_slot=slot(0),
  )
  try
    owner.append_decode(
      request_id=request_id(4UL),
      request_generation=@inference.RequestGeneration::initial(),
      sequence_token_position=1,
      input_token=2,
      pages=[pages.allocate_one()],
      capabilities=[@plan.KernelCapabilityId::causal_attention()],
      sampling=@inference.SamplingParameters::greedy(),
      sampling_seed=sampling_seed(),
      sample_index=0,
      completion_slot=slot(1),
    )
  catch {
    @worker_protocol.InvalidRowDescriptor(kind=Decode, rule=DuplicateRequest) =>
      ()
    _ => fail("expected duplicate-request rejection")
  } noraise {
    _ => fail("duplicate request was accepted")
  }
  try
    owner.append_decode(
      request_id=request_id(5UL),
      request_generation=@inference.RequestGeneration::initial(),
      sequence_token_position=1,
      input_token=2,
      pages=[pages.allocate_one()],
      capabilities=[@plan.KernelCapabilityId::causal_attention()],
      sampling=@inference.SamplingParameters::greedy(),
      sampling_seed=sampling_seed(),
      sample_index=0,
      completion_slot=slot(0),
    )
  catch {
    @worker_protocol.InvalidRowDescriptor(
      kind=Decode,
      rule=DuplicateCompletionSlot
    ) => ()
    _ => fail("expected duplicate-slot rejection")
  } noraise {
    _ => fail("duplicate completion slot was accepted")
  }
  owner.append_decode(
    request_id=request_id(5UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_position=1,
    input_token=2,
    pages=[pages.allocate_one()],
    capabilities=[@plan.KernelCapabilityId::causal_attention()],
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=sampling_seed(),
    sample_index=0,
    completion_slot=slot(1),
  )
  let submitted = owner.submit(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=model_generation(1UL),
    token_budget=2,
    None,
    model_generation(1UL),
  )
  assert_eq(submitted.row_count(), 2)
  assert_eq(submitted.token_count(), 2)
  assert_eq(submitted.page_count(), 2)
  assert_eq(submitted.capability_count(), 2)
}
