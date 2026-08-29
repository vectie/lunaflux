///|
fn hostile_request_id(value : UInt64) -> @inference.RequestId raise {
  @inference.RequestId::new(value)
}

///|
fn run_batch_protocol_hostiles(
  model_plan : @plan.ModelPlan,
  page : @page_allocator.PageId,
) -> Unit raise {
  let limits = fixture_worker_limits()
  let generation = @inference.RequestGeneration::initial()
  let capability = @plan.KernelCapabilityId::embedding_lookup()
  let rows = @worker_protocol.SchedulePlanBuffer::new(limits)
  for index in 0..<FIXTURE_MAX_ROWS {
    rows.append_prefill(
      request_id=hostile_request_id((index + 1).to_uint64()),
      request_generation=generation,
      sequence_token_start=0,
      tokens=[index],
      pages=[page],
      capabilities=[capability],
      completion_slot=@worker_protocol.CompletionSlot::new(index, limits),
    )
  }
  try
    rows.append_prefill(
      request_id=hostile_request_id(100UL),
      request_generation=generation,
      sequence_token_start=0,
      tokens=[0],
      pages=[page],
      capabilities=[capability],
      completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
    )
  catch {
    @worker_protocol.WorkerProtocolError::BoundExceeded(PlanRows) => ()
    _ => alloc_probe_abort()
  } noraise {
    _ => alloc_probe_abort()
  }

  let tokens = @worker_protocol.SchedulePlanBuffer::new(limits)
  let token_draft = tokens.begin_prefill(
    request_id=hostile_request_id(101UL),
    request_generation=generation,
    sequence_token_start=0,
    completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
  )
  for token in 0..<FIXTURE_MAX_TOKENS {
    token_draft.push_token(token % 8)
  }
  try token_draft.push_token(0) catch {
    BoundExceeded(PlanTokens) => ()
    _ => alloc_probe_abort()
  } noraise {
    _ => alloc_probe_abort()
  }
  token_draft.rollback()

  let pages = @worker_protocol.SchedulePlanBuffer::new(limits)
  let page_draft = pages.begin_prefill(
    request_id=hostile_request_id(102UL),
    request_generation=generation,
    sequence_token_start=0,
    completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
  )
  for _ in 0..<FIXTURE_MAX_PAGES {
    page_draft.push_page(page)
  }
  try page_draft.push_page(page) catch {
    BoundExceeded(PlanPages) => ()
    _ => alloc_probe_abort()
  } noraise {
    _ => alloc_probe_abort()
  }
  page_draft.rollback()

  let capabilities = @worker_protocol.SchedulePlanBuffer::new(limits)
  let capability_draft = capabilities.begin_prefill(
    request_id=hostile_request_id(103UL),
    request_generation=generation,
    sequence_token_start=0,
    completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
  )
  for _ in 0..<FIXTURE_MAX_CAPABILITIES {
    capability_draft.push_capability(capability)
  }
  try capability_draft.push_capability(capability) catch {
    BoundExceeded(Capabilities) => ()
    _ => alloc_probe_abort()
  } noraise {
    _ => alloc_probe_abort()
  }
  capability_draft.rollback()
  if model_plan.required_capabilities().length() != FIXTURE_CAPABILITIES_PER_ROW {
    alloc_probe_abort()
  }
}

///|
fn run_invalid_csr_evidence(
  directory : String,
  manifest : String,
  model : @spec.LlamaModelMetadata,
  model_plan : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
) -> Unit raise {
  let owner = prepare_fixture_owner(
    directory, manifest, model, model_plan, target,
  )
  let allocator = @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=2,
      max_run_pages=1,
      max_active_references_per_page=1,
      max_cached_references_per_page=1,
    ),
  )
  let first = allocator.allocate_one()
  let second = allocator.allocate_one()
  let limits = fixture_worker_limits()
  let schedule = @worker_protocol.SchedulePlanBuffer::new(limits)
  schedule.append_prefill(
    request_id=hostile_request_id(201UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[0],
    pages=[first, second],
    capabilities=model_plan.required_capabilities(),
    completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
  )
  let generation = @worker_protocol.ModelPlanGeneration::first()
  let plan = schedule.submit(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=generation,
    token_budget=1,
    None,
    generation,
  )
  let frame_owner = @worker_wire.PlanFrameBuffer::new(
    limits,
    fixture_inference_limits(),
  )
  let frame = frame_owner.encode_plan(plan)
  let completions = @worker_wire.CompletionFrameBuffer::new(
    limits,
    fixture_inference_limits(),
  )
  let writer = completions.writer_for(frame)
  try owner.execute_frame(frame, writer) catch {
    Executor(InvalidPlan(PageCount)) => ()
    _ => alloc_probe_abort()
  } noraise {
    _ => alloc_probe_abort()
  }
  if owner.lifecycle() != WorkerFaulted {
    alloc_probe_abort()
  }
  try completions.current() catch {
    StaleFrame => ()
    _ => alloc_probe_abort()
  } noraise {
    _ => alloc_probe_abort()
  }
  let reuse = completions.writer_for(frame)
  reuse.abort()
  owner.close()
  allocator.release_active(first)
  allocator.release_active(second)
  if fake_live_children() != 0 || allocator.allocated_page_count() != 0 {
    alloc_probe_abort()
  }
}
