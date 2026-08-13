///|
fn recipe_limits(
  max_capabilities : Int,
) -> @worker_protocol.WorkerProtocolLimits {
  try! @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=3,
    max_decode_rows=3,
    max_plan_rows=3,
    max_plan_tokens=8,
    max_plan_pages=8,
    max_capabilities~,
    max_completion_slots=3,
    max_sequence_tokens=16,
    max_token_id=1024,
  )
}

///|
fn recipe_identity() -> @spec.ModelIdentity {
  let content = try! @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let plan = try! @spec.PlanDigest::from_sha256(
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  )
  @spec.ModelIdentity::new(content, plan)
}

///|
fn capability_recipe(
  capabilities : ArrayView[@plan.KernelCapabilityId],
  limits : @worker_protocol.WorkerProtocolLimits,
) -> @worker_protocol.RowCapabilityRecipe raise @worker_protocol.WorkerProtocolError {
  @worker_protocol.RowCapabilityRecipe::new(
    identity=recipe_identity(),
    model_plan_generation=model_generation(1UL),
    capabilities,
    limits,
  )
}

///|
test "capability recipe defensively owns exact ordered duplicates" {
  let source = [
    @plan.KernelCapabilityId::embedding_lookup(),
    @plan.KernelCapabilityId::rms_norm(),
    @plan.KernelCapabilityId::rms_norm(),
  ]
  let recipe = capability_recipe(source, recipe_limits(8))
  source[0] = @plan.KernelCapabilityId::causal_attention()
  assert_eq(recipe.length(), 3)
  assert_eq(recipe.identity(), recipe_identity())
  assert_eq(recipe.model_plan_generation(), model_generation(1UL))

  let pages = allocator()
  let buffer = @worker_protocol.SchedulePlanBuffer::new(recipe_limits(8))
  let draft = buffer.begin_prefill(
    request_id=request_id(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    completion_slot=@worker_protocol.CompletionSlot::new(0, recipe_limits(8)),
  )
  draft.push_token(7)
  draft.push_page(pages.allocate_one())
  draft.push_recipe(recipe)
  draft.commit()
  let submitted = buffer.submit(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=model_generation(1UL),
    token_budget=1,
    None,
    model_generation(1UL),
  )
  assert_eq(
    submitted.capability_at(0),
    @plan.KernelCapabilityId::embedding_lookup(),
  )
  assert_eq(submitted.capability_at(1), @plan.KernelCapabilityId::rms_norm())
  assert_eq(submitted.capability_at(2), @plan.KernelCapabilityId::rms_norm())
}

///|
test "capability recipe rejects empty and oversized startup lists" {
  try capability_recipe([], recipe_limits(2)) catch {
    InvalidCapabilityRecipe(EmptyCapabilityRecipe) => ()
    _ => fail("expected empty recipe rejection")
  } noraise {
    _ => fail("expected construction to fail")
  }
  try
    capability_recipe(
      [
        @plan.KernelCapabilityId::embedding_lookup(),
        @plan.KernelCapabilityId::rms_norm(),
        @plan.KernelCapabilityId::causal_attention(),
      ],
      recipe_limits(2),
    )
  catch {
    BoundExceeded(Capabilities) => ()
    _ => fail("expected oversized recipe rejection")
  } noraise {
    _ => fail("expected construction to fail")
  }
}

///|
test "capability recipe computes bounded flattened cell requirements" {
  let limits = recipe_limits(6)
  let recipe = capability_recipe(
    [
      @plan.KernelCapabilityId::rms_norm(),
      @plan.KernelCapabilityId::causal_attention(),
    ],
    limits,
  )
  assert_eq(recipe.required_cells(0, limits), 0)
  assert_eq(recipe.required_cells(3, limits), 6)
  try recipe.required_cells(-1, limits) catch {
    InvalidLimit(PlanRows) => ()
    _ => fail("expected negative row rejection")
  } noraise {
    _ => fail("expected requirement to fail")
  }
  try recipe.required_cells(4, limits) catch {
    InvalidLimit(PlanRows) => ()
    _ => fail("expected row limit rejection")
  } noraise {
    _ => fail("expected requirement to fail")
  }
  let tight = recipe_limits(5)
  let tight_recipe = capability_recipe(
    [
      @plan.KernelCapabilityId::rms_norm(),
      @plan.KernelCapabilityId::causal_attention(),
    ],
    tight,
  )
  try tight_recipe.required_cells(3, tight) catch {
    BoundExceeded(Capabilities) => ()
    _ => fail("expected capability cell rejection")
  } noraise {
    _ => fail("expected requirement to fail")
  }
}

///|
test "recipe push preflights capacity without partial draft mutation" {
  let limits = recipe_limits(3)
  let recipe = capability_recipe(
    [
      @plan.KernelCapabilityId::rms_norm(),
      @plan.KernelCapabilityId::causal_attention(),
      @plan.KernelCapabilityId::residual_add(),
    ],
    limits,
  )
  let pages = allocator()
  let buffer = @worker_protocol.SchedulePlanBuffer::new(limits)
  let draft = buffer.begin_decode(
    request_id=request_id(2UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_position=0,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=sampling_seed(),
    sample_index=0,
    completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
  )
  draft.push_token(8)
  draft.push_page(pages.allocate_one())
  draft.push_capability(@plan.KernelCapabilityId::embedding_lookup())
  try draft.push_recipe(recipe) catch {
    BoundExceeded(Capabilities) => ()
    _ => fail("expected transactional recipe capacity failure")
  } noraise {
    _ => fail("expected recipe push to fail")
  }
  draft.push_capability(@plan.KernelCapabilityId::gated_mlp())
  draft.push_capability(@plan.KernelCapabilityId::dense_projection())
  draft.commit()
  let submitted = buffer.submit(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=model_generation(1UL),
    token_budget=1,
    None,
    model_generation(1UL),
  )
  assert_eq(submitted.capability_count(), 3)
  assert_eq(
    submitted.capability_at(0),
    @plan.KernelCapabilityId::embedding_lookup(),
  )
  assert_eq(submitted.capability_at(1), @plan.KernelCapabilityId::gated_mlp())
  assert_eq(
    submitted.capability_at(2),
    @plan.KernelCapabilityId::dense_projection(),
  )
}

///|
test "recipe push is semantically identical to ordered scalar pushes" {
  let limits = recipe_limits(4)
  let expected = [
    @plan.KernelCapabilityId::embedding_lookup(),
    @plan.KernelCapabilityId::rms_norm(),
    @plan.KernelCapabilityId::rms_norm(),
    @plan.KernelCapabilityId::causal_attention(),
  ]
  let recipe = capability_recipe(expected, limits)
  let pages = allocator()
  let scalar_buffer = @worker_protocol.SchedulePlanBuffer::new(limits)
  let scalar = scalar_buffer.begin_decode(
    request_id=request_id(6UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_position=0,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=sampling_seed(),
    sample_index=0,
    completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
  )
  scalar.push_token(4)
  scalar.push_page(pages.allocate_one())
  for capability in expected {
    scalar.push_capability(capability)
  }
  scalar.commit()
  let scalar_plan = scalar_buffer.submit(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=model_generation(1UL),
    token_budget=1,
    None,
    model_generation(1UL),
  )

  let recipe_buffer = @worker_protocol.SchedulePlanBuffer::new(limits)
  let copied = recipe_buffer.begin_decode(
    request_id=request_id(6UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_position=0,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=sampling_seed(),
    sample_index=0,
    completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
  )
  copied.push_token(4)
  copied.push_page(pages.allocate_one())
  copied.push_recipe(recipe)
  copied.commit()
  let recipe_plan = recipe_buffer.submit(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=model_generation(1UL),
    token_budget=1,
    None,
    model_generation(1UL),
  )
  assert_eq(recipe_plan.capability_count(), scalar_plan.capability_count())
  for index in 0..<recipe_plan.capability_count() {
    assert_eq(
      recipe_plan.capability_at(index),
      scalar_plan.capability_at(index),
    )
  }
}

///|
test "one recipe applies to intermediate final and decode rows" {
  let limits = recipe_limits(6)
  let recipe = capability_recipe(
    [
      @plan.KernelCapabilityId::rms_norm(),
      @plan.KernelCapabilityId::causal_attention(),
    ],
    limits,
  )
  let pages = allocator()
  let buffer = @worker_protocol.SchedulePlanBuffer::new(limits)
  let intermediate = buffer.begin_prefill(
    request_id=request_id(3UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
  )
  intermediate.push_token(1)
  intermediate.push_page(pages.allocate_one())
  intermediate.push_recipe(recipe)
  intermediate.commit()
  let final_prompt = buffer.begin_final_prefill(
    request_id=request_id(4UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=sampling_seed(),
    sample_index=0,
    completion_slot=@worker_protocol.CompletionSlot::new(1, limits),
  )
  final_prompt.push_token(2)
  final_prompt.push_page(pages.allocate_one())
  final_prompt.push_recipe(recipe)
  final_prompt.commit()
  let decode = buffer.begin_decode(
    request_id=request_id(5UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_position=1,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=sampling_seed(),
    sample_index=0,
    completion_slot=@worker_protocol.CompletionSlot::new(2, limits),
  )
  decode.push_token(3)
  decode.push_page(pages.allocate_one())
  decode.push_recipe(recipe)
  decode.commit()
  let submitted = buffer.submit(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=model_generation(1UL),
    token_budget=3,
    None,
    model_generation(1UL),
  )
  assert_eq(submitted.capability_count(), 6)
  assert_eq(submitted.prefill_row(0).capabilities(), capability_view(0, 2))
  assert_eq(submitted.prefill_row(1).capabilities(), capability_view(2, 2))
  assert_eq(submitted.decode_row(0).capabilities(), capability_view(4, 2))
}
