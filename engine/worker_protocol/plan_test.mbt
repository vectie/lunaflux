///|
test "complete mixed plan is exact and immutable" {
  let page_owner = allocator()
  let tokens = [10, 11, 12]
  let pages = [page_owner.allocate_one(), page_owner.allocate_one()]
  let capabilities = [
    @plan.KernelCapabilityId::embedding_lookup(),
    @plan.KernelCapabilityId::causal_attention(),
  ]
  let prefills = [prefill(7UL, 0UL, 0, 2, 0, 0, 0)]
  let decodes = [decode(8UL, 3UL, 2, 1, 1, 1)]
  let plan = @worker_protocol.SchedulePlan::new(
    sequence=plan_sequence(2UL),
    model_plan_generation=model_generation(1UL),
    token_budget=3,
    tokens~,
    pages~,
    capabilities~,
    prefill_rows=prefills,
    decode_rows=decodes,
    limits=protocol_limits(),
  )
  tokens[0] = 99
  pages[0] = page_owner.allocate_one()
  capabilities[0] = @plan.KernelCapabilityId::gated_mlp()
  prefills[0] = prefill(9UL, 0UL, 0, 2, 0, 0, 0)
  decodes[0] = decode(10UL, 0UL, 2, 1, 1, 1)
  assert_eq(plan.sequence().value(), 2UL)
  assert_eq(plan.model_plan_generation().value(), 1UL)
  assert_eq(plan.token_budget(), 3)
  assert_eq(plan.row_count(), 2)
  assert_eq(plan.tokens()[0], 10)
  assert_eq(plan.pages()[0].index(), 0)
  assert_eq(
    plan.capabilities()[0],
    @plan.KernelCapabilityId::embedding_lookup(),
  )
  assert_eq(plan.prefill_rows()[0].request_id(), request_id(7UL))
  assert_eq(plan.decode_rows()[0].request_id(), request_id(8UL))
  plan.validate_submission(Some(plan_sequence(1UL)), model_generation(1UL))
}

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
    error => debug_inspect(error, content="InvalidLimit(PlanRows)")
  } noraise {
    _ => fail("expected invalid row limit")
  }
  try @worker_protocol.PlanSequence::from_wire(0UL) catch {
    error =>
      debug_inspect(error, content="InvalidIdentity(PlanSequenceIdentity)")
  } noraise {
    _ => fail("expected invalid sequence")
  }
  let maximum = plan_sequence(0xffffffffffffffffUL)
  try maximum.next() catch {
    error =>
      debug_inspect(error, content="IdentityOverflow(PlanSequenceIdentity)")
  } noraise {
    _ => fail("expected sequence overflow")
  }
  assert_true(plan_sequence(2UL).immediately_follows(plan_sequence(1UL)))
  assert_false(plan_sequence(3UL).immediately_follows(plan_sequence(1UL)))
  assert_false(
    plan_sequence(1UL).immediately_follows(plan_sequence(0xffffffffffffffffUL)),
  )
}

///|
test "submission validation admits first and rejects duplicate and skipped plans" {
  let first = mixed_plan()
  let first_plan = @worker_protocol.SchedulePlan::new(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=first.model_plan_generation(),
    token_budget=first.token_budget(),
    tokens=first.tokens(),
    pages=first.pages(),
    capabilities=first.capabilities(),
    prefill_rows=first.prefill_rows(),
    decode_rows=first.decode_rows(),
    limits=protocol_limits(),
  )
  first_plan.validate_submission(None, model_generation(1UL))
  try
    first_plan.validate_submission(
      Some(plan_sequence(1UL)),
      model_generation(1UL),
    )
  catch {
    error => debug_inspect(error, content="StalePlanSequence")
  } noraise {
    _ => fail("expected duplicate sequence rejection")
  }
  try first.validate_submission(None, model_generation(1UL)) catch {
    error => debug_inspect(error, content="StalePlanSequence")
  } noraise {
    _ => fail("expected skipped initial sequence rejection")
  }
  let maximum = @worker_protocol.SchedulePlan::new(
    sequence=plan_sequence(0xffffffffffffffffUL),
    model_plan_generation=first.model_plan_generation(),
    token_budget=first.token_budget(),
    tokens=first.tokens(),
    pages=first.pages(),
    capabilities=first.capabilities(),
    prefill_rows=first.prefill_rows(),
    decode_rows=first.decode_rows(),
    limits=protocol_limits(),
  )
  try
    maximum.validate_submission(
      Some(plan_sequence(0xfffffffffffffffeUL)),
      model_generation(1UL),
    )
  catch {
    _ => fail("maximum sequence is a valid immediate successor")
  } noraise {
    _ => ()
  }
}

///|
test "completion slot and completion token bounds are distinct" {
  try @worker_protocol.CompletionSlot::new(8, protocol_limits()) catch {
    error => debug_inspect(error, content="InvalidCompletionSlot")
  } noraise {
    _ => fail("expected invalid completion slot")
  }
  try
    @worker_protocol.CompletionEntry::decode_completed(
      slot=slot(1),
      request_id=request_id(1UL),
      request_generation=@inference.RequestGeneration::initial(),
      token_id=1025,
      limits=protocol_limits(),
    )
  catch {
    error => debug_inspect(error, content="InvalidCompletionToken(slot=1)")
  } noraise {
    _ => fail("expected invalid completion token")
  }
  try
    @worker_protocol.CompletionEntry::prefill_completed(
      slot=slot(0),
      request_id=request_id(1UL),
      request_generation=@inference.RequestGeneration::initial(),
      processed_tokens=33,
      limits=protocol_limits(),
    )
  catch {
    error =>
      debug_inspect(error, content="CompletionTokenCountMismatch(index=0)")
  } noraise {
    _ => fail("expected bounded prefill completion")
  }
}

///|
test "empty ranges and illegal decode rows fail before planning" {
  try
    @worker_protocol.TokenRange::new(start=0, length=0, protocol_limits())
  catch {
    error =>
      debug_inspect(error, content="InvalidView(table=Tokens, rule=EmptyRange)")
  } noraise {
    _ => fail("expected empty range rejection")
  }
  try
    @worker_protocol.DecodeRow::new(
      request_id=request_id(1UL),
      request_generation=@inference.RequestGeneration::initial(),
      sequence_token_position=0,
      input_token=token_range(0, 2),
      page_table=page_view(0, 1),
      capabilities=capability_view(0, 1),
      sampling=@inference.SamplingParameters::greedy(),
      completion_slot=slot(0),
      limits=protocol_limits(),
    )
  catch {
    error =>
      debug_inspect(
        error,
        content="InvalidRowDescriptor(kind=Decode, rule=EmptyRange)",
      )
  } noraise {
    _ => fail("expected multi-token decode rejection")
  }
}

///|
test "duplicate request and completion slot are rejected" {
  let page_owner = allocator()
  try
    @worker_protocol.SchedulePlan::new(
      sequence=plan_sequence(1UL),
      model_plan_generation=model_generation(1UL),
      token_budget=2,
      tokens=[1, 2],
      pages=[page_owner.allocate_one(), page_owner.allocate_one()],
      capabilities=[
        @plan.KernelCapabilityId::embedding_lookup(),
        @plan.KernelCapabilityId::causal_attention(),
      ],
      prefill_rows=[prefill(4UL, 0UL, 0, 1, 0, 0, 0)],
      decode_rows=[decode(4UL, 1UL, 1, 1, 1, 1)],
      limits=protocol_limits(),
    )
  catch {
    error =>
      debug_inspect(
        error,
        content="InvalidRow(kind=Decode, index=0, rule=DuplicateRequest)",
      )
  } noraise {
    _ => fail("expected duplicate request rejection")
  }
  let another_owner = allocator()
  try
    @worker_protocol.SchedulePlan::new(
      sequence=plan_sequence(1UL),
      model_plan_generation=model_generation(1UL),
      token_budget=2,
      tokens=[1, 2],
      pages=[another_owner.allocate_one(), another_owner.allocate_one()],
      capabilities=[
        @plan.KernelCapabilityId::embedding_lookup(),
        @plan.KernelCapabilityId::causal_attention(),
      ],
      prefill_rows=[prefill(4UL, 0UL, 0, 1, 0, 0, 0)],
      decode_rows=[decode(5UL, 0UL, 1, 1, 1, 0)],
      limits=protocol_limits(),
    )
  catch {
    error =>
      debug_inspect(
        error,
        content="InvalidRow(kind=Decode, index=0, rule=DuplicateCompletionSlot)",
      )
  } noraise {
    _ => fail("expected duplicate slot rejection")
  }
}

///|
test "actual table bounds and exact totals are enforced" {
  let page_owner = allocator()
  try
    @worker_protocol.SchedulePlan::new(
      sequence=plan_sequence(1UL),
      model_plan_generation=model_generation(1UL),
      token_budget=1,
      tokens=[1],
      pages=[page_owner.allocate_one()],
      capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
      prefill_rows=[prefill(1UL, 0UL, 3, 1, 0, 0, 0)],
      decode_rows=[],
      limits=protocol_limits(),
    )
  catch {
    error =>
      debug_inspect(
        error,
        content="InvalidRow(kind=Prefill, index=0, rule=RangeOutsideTable)",
      )
  } noraise {
    _ => fail("expected actual range rejection")
  }
  let another_owner = allocator()
  try
    @worker_protocol.SchedulePlan::new(
      sequence=plan_sequence(1UL),
      model_plan_generation=model_generation(1UL),
      token_budget=2,
      tokens=[1],
      pages=[another_owner.allocate_one()],
      capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
      prefill_rows=[prefill(1UL, 0UL, 0, 1, 0, 0, 0)],
      decode_rows=[],
      limits=protocol_limits(),
    )
  catch {
    error => debug_inspect(error, content="TokenBudgetMismatch")
  } noraise {
    _ => fail("expected exact token budget rejection")
  }
}
