///|
test "mixed completion record validates independent result ordering" {
  let plan = mixed_plan()
  let entries = [
    try! @worker_protocol.CompletionEntry::decode_completed(
      slot=slot(1),
      request_id=request_id(8UL),
      request_generation=@inference.RequestGeneration::from_wire(3UL),
      token_id=42,
      limits=protocol_limits(),
    ),
    try! @worker_protocol.CompletionEntry::prefill_completed(
      slot=slot(0),
      request_id=request_id(7UL),
      request_generation=@inference.RequestGeneration::initial(),
      processed_tokens=2,
      limits=protocol_limits(),
    ),
  ]
  let record = @worker_protocol.CompletionRecord::new(
    sequence=plan.sequence(),
    model_plan_generation=plan.model_plan_generation(),
    entries~,
    limits=protocol_limits(),
  )
  entries[0] = @worker_protocol.CompletionEntry::failed(
    slot=slot(1),
    request_id=request_id(8UL),
    request_generation=@inference.RequestGeneration::from_wire(3UL),
    failure=ExecutionFailed,
  )
  assert_true(record.entries()[0].outcome() is DecodeCompleted(token_id=42))
  record.validate_for(plan)
}

///|
test "stale plan and model generations are rejected" {
  let plan = mixed_plan()
  let valid_entries = [
    try! @worker_protocol.CompletionEntry::prefill_completed(
      slot=slot(0),
      request_id=request_id(7UL),
      request_generation=@inference.RequestGeneration::initial(),
      processed_tokens=2,
      limits=protocol_limits(),
    ),
    try! @worker_protocol.CompletionEntry::decode_completed(
      slot=slot(1),
      request_id=request_id(8UL),
      request_generation=@inference.RequestGeneration::from_wire(3UL),
      token_id=42,
      limits=protocol_limits(),
    ),
  ]
  let stale_plan = try! @worker_protocol.CompletionRecord::new(
    sequence=plan_sequence(3UL),
    model_plan_generation=plan.model_plan_generation(),
    entries=valid_entries,
    limits=protocol_limits(),
  )
  try stale_plan.validate_for(plan) catch {
    error => debug_inspect(error, content="StalePlanSequence")
  } noraise {
    _ => fail("expected stale plan sequence")
  }
  let stale_model = try! @worker_protocol.CompletionRecord::new(
    sequence=plan.sequence(),
    model_plan_generation=model_generation(2UL),
    entries=valid_entries,
    limits=protocol_limits(),
  )
  try stale_model.validate_for(plan) catch {
    error => debug_inspect(error, content="StaleModelPlanGeneration")
  } noraise {
    _ => fail("expected stale model generation")
  }
}

///|
test "post-submit cancellation permits retirement but blocks publication" {
  let plan = mixed_plan()
  let entries = [
    try! @worker_protocol.CompletionEntry::prefill_completed(
      slot=slot(0),
      request_id=request_id(7UL),
      request_generation=@inference.RequestGeneration::initial(),
      processed_tokens=2,
      limits=protocol_limits(),
    ),
    try! @worker_protocol.CompletionEntry::decode_completed(
      slot=slot(1),
      request_id=request_id(8UL),
      request_generation=@inference.RequestGeneration::from_wire(3UL),
      token_id=42,
      limits=protocol_limits(),
    ),
  ]
  let record = try! @worker_protocol.CompletionRecord::new(
    sequence=plan.sequence(),
    model_plan_generation=plan.model_plan_generation(),
    entries~,
    limits=protocol_limits(),
  )
  record.validate_for(plan)
  assert_true(
    record.entries()[0].is_current(
      request_id(7UL),
      @inference.RequestGeneration::initial(),
    ),
  )
  assert_false(
    record.entries()[0].is_current(
      request_id(7UL),
      @inference.RequestGeneration::from_wire(1UL),
    ),
  )
}

///|
test "completion duplicates, kind, and token totals fail closed" {
  let plan = mixed_plan()
  let duplicate = try! @worker_protocol.CompletionEntry::prefill_completed(
    slot=slot(0),
    request_id=request_id(7UL),
    request_generation=@inference.RequestGeneration::initial(),
    processed_tokens=2,
    limits=protocol_limits(),
  )
  try
    @worker_protocol.CompletionRecord::new(
      sequence=plan.sequence(),
      model_plan_generation=plan.model_plan_generation(),
      entries=[duplicate, duplicate],
      limits=protocol_limits(),
    )
  catch {
    error => debug_inspect(error, content="DuplicateCompletionSlot(index=1)")
  } noraise {
    _ => fail("expected duplicate completion slot")
  }
  let wrong = [
    try! @worker_protocol.CompletionEntry::decode_completed(
      slot=slot(0),
      request_id=request_id(7UL),
      request_generation=@inference.RequestGeneration::initial(),
      token_id=3,
      limits=protocol_limits(),
    ),
    try! @worker_protocol.CompletionEntry::decode_completed(
      slot=slot(1),
      request_id=request_id(8UL),
      request_generation=@inference.RequestGeneration::from_wire(3UL),
      token_id=4,
      limits=protocol_limits(),
    ),
  ]
  let record = try! @worker_protocol.CompletionRecord::new(
    sequence=plan.sequence(),
    model_plan_generation=plan.model_plan_generation(),
    entries=wrong,
    limits=protocol_limits(),
  )
  try record.validate_for(plan) catch {
    error => debug_inspect(error, content="CompletionKindMismatch(index=0)")
  } noraise {
    _ => fail("expected completion kind mismatch")
  }
  let wrong_count = [
    try! @worker_protocol.CompletionEntry::prefill_completed(
      slot=slot(0),
      request_id=request_id(7UL),
      request_generation=@inference.RequestGeneration::initial(),
      processed_tokens=1,
      limits=protocol_limits(),
    ),
    wrong[1],
  ]
  let count_record = try! @worker_protocol.CompletionRecord::new(
    sequence=plan.sequence(),
    model_plan_generation=plan.model_plan_generation(),
    entries=wrong_count,
    limits=protocol_limits(),
  )
  try count_record.validate_for(plan) catch {
    error =>
      debug_inspect(error, content="CompletionTokenCountMismatch(index=0)")
  } noraise {
    _ => fail("expected completion token count mismatch")
  }
}
