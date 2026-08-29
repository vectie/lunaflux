///|
fn approved_worker_limits() -> @worker_protocol.WorkerProtocolLimits raise {
  @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=1,
    max_decode_rows=1,
    max_plan_rows=1,
    max_plan_tokens=1,
    max_plan_pages=APPROVED_CONTEXT_PAGE_COUNT,
    max_capabilities=64,
    max_completion_slots=1,
    max_sequence_tokens=APPROVED_CONTEXT_TOKENS,
    max_token_id=2999,
  )
}

///|
fn approved_step_limits() -> @device_step.DeviceStepLimits raise {
  @device_step.DeviceStepLimits::new(
    max_rows=1,
    max_tokens=1,
    max_pages=APPROVED_CONTEXT_PAGE_COUNT,
    max_sequence_tokens=APPROVED_CONTEXT_TOKENS,
    max_token_id=2999,
  )
}

///|
fn approved_schedule_plan(
  model_plan : @plan.ModelPlan,
  page : @page_allocator.PageId,
  sequence_value : UInt64,
  input_token : Int,
) -> (
  @worker_protocol.SchedulePlanBuffer,
  @worker_protocol.SubmittedSchedulePlan,
) raise {
  let limits = approved_worker_limits()
  let owner = @worker_protocol.SchedulePlanBuffer::new(limits)
  let request_id = @inference.RequestId::new(1UL)
  let request_generation = @inference.RequestGeneration::initial()
  let slot = @worker_protocol.CompletionSlot::new(0, limits)
  if sequence_value == 1UL {
    owner.append_final_prefill(
      request_id~,
      request_generation~,
      sequence_token_start=0,
      tokens=[input_token],
      pages=[page],
      capabilities=model_plan.required_capabilities(),
      sampling=@inference.SamplingParameters::greedy(),
      sampling_seed=@inference.SamplingSeed::new(7UL),
      sample_index=0,
      completion_slot=slot,
    )
  } else {
    owner.append_decode(
      request_id~,
      request_generation~,
      sequence_token_position=sequence_value.to_int() - 1,
      input_token~,
      pages=[page],
      capabilities=model_plan.required_capabilities(),
      sampling=@inference.SamplingParameters::greedy(),
      sampling_seed=@inference.SamplingSeed::new(7UL),
      sample_index=sequence_value.to_int() - 1,
      completion_slot=slot,
    )
  }
  let generation = @worker_protocol.ModelPlanGeneration::first()
  let sequence = @worker_protocol.PlanSequence::from_wire(sequence_value)
  let previous = if sequence_value == 1UL {
    None
  } else {
    Some(@worker_protocol.PlanSequence::from_wire(sequence_value - 1UL))
  }
  let plan = owner.submit(
    sequence~,
    model_plan_generation=generation,
    token_budget=1,
    previous,
    generation,
  )
  (owner, plan)
}

///|
fn execute_approved_step(
  executor : @device_step.PagedGraphExecutor,
  model_plan : @plan.ModelPlan,
  page : @page_allocator.PageId,
  sequence_value : UInt64,
  input_token : Int,
  selected_ids : FixedArray[Int]?,
  selected_logits : FixedArray[Double]?,
) -> Int raise {
  let (plan_owner, plan) = approved_schedule_plan(
    model_plan, page, sequence_value, input_token,
  )
  let completion_owner = @worker_protocol.CompletionBuffer::new(
    approved_worker_limits(),
  )
  let staged = executor.stage(plan)
  let executed = executor.execute(staged)
  match (selected_ids, selected_logits) {
    (Some(ids), Some(destination)) =>
      executor.observe_offline_selected_logits(executed, 0, ids, destination)
    (None, None) => ()
    _ => raise InvalidInput("selected-logit observation arguments disagree")
  }
  let submitted = executor.sample_completion(
    executed,
    plan,
    completion_owner.writer_for(plan),
  )
  submitted.validate_for(plan)
  let entry = submitted.entry(0)
  let token = entry.token_id()
  executor.finish(executed)
  let consumed = completion_owner.consume(submitted, plan)
  completion_owner.reset(consumed)
  let retired = plan_owner.retire(plan)
  plan_owner.reset(retired)
  token
}

///|
fn execute_approved_continuation(
  executor : @device_step.PagedGraphExecutor,
  model_plan : @plan.ModelPlan,
  corpus : ApprovedCorpusCase,
) -> Double raise {
  let allocator = @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=1,
      max_run_pages=1,
      max_active_references_per_page=1,
      max_cached_references_per_page=1,
    ),
  )
  let page = allocator.allocate_one()
  defer {
    try! allocator.release_active(page)
    if allocator.allocated_page_count() != 0 || allocator.free_page_count() != 1 {
      abort("approved BF16 page owner did not balance")
    }
  }
  let observed = FixedArray::make(corpus.selected_ids.length(), 0.0)
  let mut input_token = 1
  for index, expected in corpus.continuation {
    let actual = execute_approved_step(
      executor,
      model_plan,
      page,
      (index + 1).to_uint64(),
      input_token,
      if index == 0 {
        Some(corpus.selected_ids)
      } else {
        None
      },
      if index == 0 {
        Some(observed)
      } else {
        None
      },
    )
    if actual != expected {
      raise NumericalMismatch("greedy continuation mismatch at output \{index}")
    }
    input_token = actual
  }
  compare_approved_selected_logits(corpus.selected_logits, observed)
}
