///|
fn final_prefill_row() -> @worker_protocol.PrefillRow {
  try! @worker_protocol.PrefillRow::final_prompt(
    request_id=request_id(21UL),
    request_generation=@inference.RequestGeneration::from_wire(2UL),
    sequence_token_start=0,
    tokens=token_range(0, 3),
    page_table=page_view(0, 1),
    capabilities=capability_view(0, 1),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=sampling_seed(),
    sample_index=5,
    completion_slot=slot(0),
    limits=protocol_limits(),
  )
}

///|
fn final_prefill_plan() -> @worker_protocol.SchedulePlan {
  let pages = allocator()
  try! @worker_protocol.SchedulePlan::new(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=model_generation(1UL),
    token_budget=3,
    tokens=[10, 11, 12],
    pages=[pages.allocate_one()],
    capabilities=[@plan.KernelCapabilityId::causal_attention()],
    prefill_rows=[final_prefill_row()],
    decode_rows=[],
    limits=protocol_limits(),
  )
}

///|
test "immutable final prefill requires sampled-token completion" {
  let plan = final_prefill_plan()
  let row = plan.prefill_rows()[0]
  assert_true(row.produces_token())
  assert_eq(row.sampling(), @inference.SamplingParameters::greedy())
  assert_true(row.sampling_seed() == sampling_seed())
  assert_eq(row.sample_index(), 5)
  let final_entry = try! @worker_protocol.CompletionEntry::final_prefill_completed(
    slot=slot(0),
    request_id=request_id(21UL),
    request_generation=@inference.RequestGeneration::from_wire(2UL),
    processed_tokens=3,
    token_id=77,
    limits=protocol_limits(),
  )
  let record = try! @worker_protocol.CompletionRecord::new(
    sequence=plan.sequence(),
    model_plan_generation=plan.model_plan_generation(),
    entries=[final_entry],
    limits=protocol_limits(),
  )
  record.validate_for(plan)
  assert_true(
    record.entries()[0].outcome()
    is FinalPrefillCompleted(processed_tokens=3, token_id=77),
  )
  let wrong_kind = try! @worker_protocol.CompletionRecord::new(
    sequence=plan.sequence(),
    model_plan_generation=plan.model_plan_generation(),
    entries=[
      @worker_protocol.CompletionEntry::prefill_completed(
        slot=slot(0),
        request_id=request_id(21UL),
        request_generation=@inference.RequestGeneration::from_wire(2UL),
        processed_tokens=3,
        limits=protocol_limits(),
      ),
    ],
    limits=protocol_limits(),
  )
  try wrong_kind.validate_for(plan) catch {
    CompletionKindMismatch(index=0) => ()
    _ => fail("expected final-prefill kind mismatch")
  } noraise {
    _ => fail("expected validation to fail")
  }
}

///|
test "immutable intermediate prefill rejects sampled-token completion" {
  let plan = try! @worker_protocol.SchedulePlan::new(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=model_generation(1UL),
    token_budget=1,
    tokens=[10],
    pages=[allocator().allocate_one()],
    capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
    prefill_rows=[prefill(22UL, 0UL, 0, 1, 0, 0, 0)],
    decode_rows=[],
    limits=protocol_limits(),
  )
  assert_false(plan.prefill_rows()[0].produces_token())
  let record = try! @worker_protocol.CompletionRecord::new(
    sequence=plan.sequence(),
    model_plan_generation=plan.model_plan_generation(),
    entries=[
      @worker_protocol.CompletionEntry::final_prefill_completed(
        slot=slot(0),
        request_id=request_id(22UL),
        request_generation=@inference.RequestGeneration::initial(),
        processed_tokens=1,
        token_id=5,
        limits=protocol_limits(),
      ),
    ],
    limits=protocol_limits(),
  )
  try record.validate_for(plan) catch {
    CompletionKindMismatch(index=0) => ()
    _ => fail("expected intermediate-prefill kind mismatch")
  } noraise {
    _ => fail("expected validation to fail")
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

///|
test "final-prefill completion validates sampled token range" {
  try
    @worker_protocol.CompletionEntry::final_prefill_completed(
      slot=slot(0),
      request_id=request_id(25UL),
      request_generation=@inference.RequestGeneration::initial(),
      processed_tokens=1,
      token_id=1025,
      limits=protocol_limits(),
    )
  catch {
    InvalidCompletionToken(slot=0) => ()
    _ => fail("expected token range failure")
  } noraise {
    _ => fail("expected construction to fail")
  }
}
