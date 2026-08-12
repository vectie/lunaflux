///|
test "build_next activates FIFO requests and serializes final prefills" {
  let owner = scheduler()
  let first = owner.admit(request(41UL), 0UL)
  let second = owner.admit(request(42UL), 0UL)
  let plan = next_plan(owner)
  assert_eq(try! plan.sequence_wire(), 1UL)
  assert_eq(try! plan.token_budget(), 4)
  assert_eq(try! plan.prefill_count(), 2)
  assert_eq(try! plan.decode_count(), 0)
  let first_row = try! plan.prefill_row(0)
  let second_row = try! plan.prefill_row(1)
  assert_eq(try! first_row.request_id(), first.request_id())
  assert_eq(try! second_row.request_id(), second.request_id())
  assert_true(try! first_row.produces_token())
  assert_true(try! second_row.produces_token())
  assert_eq(try! first_row.tokens().length(), 2)
  assert_eq(try! second_row.tokens().length(), 2)
  assert_eq(owner.waiting_request_count(), 0)
  assert_eq(owner.status(first.request_id()).unwrap().state(), Prefill)
  assert_eq(owner.status(second.request_id()).unwrap().state(), Prefill)
  assert_eq(owner.free_page_count(), 6)
  assert_eq(owner.allocated_block_table_count(), 2)
  assert_true(owner.status(first.request_id()).unwrap().in_flight())
  assert_true(owner.status(second.request_id()).unwrap().in_flight())
}

///|
test "build_next reports typed no-work after all requests are in flight" {
  let owner = scheduler()
  ignore(owner.admit(request(51UL), 0UL))
  ignore(owner.admit(request(52UL), 0UL))
  ignore(next_plan(owner))
  assert_true(try! owner.build_next().is_idle())
}

///|
test "chunked prefill stays intermediate and preserves decode reserve" {
  let owner = scheduler()
  ignore(owner.admit(request_with_tokens(61UL, [1, 2, 3, 4, 5], 1, 6), 0UL))
  let plan = next_plan(owner)
  assert_eq(try! plan.token_budget(), 2)
  assert_eq(try! plan.prefill_count(), 1)
  let row = try! plan.prefill_row(0)
  assert_false(try! row.produces_token())
  assert_eq(try! row.tokens().length(), 2)
  assert_eq(try! row.page_table().length(), 1)
  assert_eq(owner.free_page_count(), 7)
}

///|
test "prefill may consume exactly to the emergency reserve boundary" {
  let owner = tight_scheduler()
  ignore(owner.admit(request(62UL), 0UL))
  let plan = next_plan(owner)
  assert_eq(try! plan.prefill_count(), 1)
  assert_eq(owner.free_page_count(), 2)
}

///|
test "intermediate and final prefills use their exact capability recipes" {
  let limits = worker_limits()
  let generation = @worker_protocol.ModelPlanGeneration::first()
  let intermediate = try! @worker_protocol.RowCapabilityRecipe::new(
    identity=selected_identity(),
    model_plan_generation=generation,
    [@plan.KernelCapabilityId::embedding_lookup()],
    limits,
  )
  let final_recipe = try! @worker_protocol.RowCapabilityRecipe::new(
    identity=selected_identity(),
    model_plan_generation=generation,
    [@plan.KernelCapabilityId::rms_norm()],
    limits,
  )
  let decode = try! @worker_protocol.RowCapabilityRecipe::new(
    identity=selected_identity(),
    model_plan_generation=generation,
    [@plan.KernelCapabilityId::causal_attention()],
    limits,
  )
  let owner = try! Scheduler::new(
    resolved_capacity(),
    selected_identity(),
    generation,
    intermediate,
    final_recipe,
    decode,
  )
  ignore(owner.admit(request_with_tokens(71UL, [1, 2, 3, 4, 5], 1, 6), 0UL))
  ignore(owner.admit(request(72UL), 0UL))
  let plan = next_plan(owner)
  assert_false(try! plan.prefill_row(0).produces_token())
  assert_true(try! plan.prefill_row(1).produces_token())
  assert_eq(
    try! plan.capability_at(0),
    @plan.KernelCapabilityId::embedding_lookup(),
  )
  assert_eq(try! plan.capability_at(1), @plan.KernelCapabilityId::rms_norm())
}
