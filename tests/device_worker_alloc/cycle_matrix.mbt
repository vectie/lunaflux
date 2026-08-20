///|
const CYCLE_ORDINARY_PREFILL : Int = 0

///|
const CYCLE_GREEDY_FINAL_PREFILL : Int = 1

///|
const CYCLE_STOCHASTIC_FINAL_PREFILL : Int = 2

///|
const CYCLE_GREEDY_DECODE : Int = 3

///|
const CYCLE_STOCHASTIC_DECODE : Int = 4

///|
struct PreparedCycle {
  plan_owner : @worker_protocol.SchedulePlanBuffer
  plan : @worker_protocol.SubmittedSchedulePlan
  frame : @worker_wire.ValidatedPlanFrame
  branch : Int
  request_id : UInt64
  request_generation : UInt64
  row_position : Int
  sample_index : Int
  sampling_seed : UInt64
  page_index : Int
  page_generation : Int
  capability_count : Int
  expected_input_token : Int
  expected_kind : @worker_protocol.CompletionEntryKind
  expected_processed_tokens : Int
  expected_token : Int
  expected_readbacks : UInt64
}

///|
struct PreparedMatrix {
  cycles : FixedArray[PreparedCycle]
  allocator : @page_allocator.PageAllocator
  pages : FixedArray[@page_allocator.PageId]
}

///|
fn stochastic_parameters() -> @inference.SamplingParameters raise {
  @inference.SamplingParameters::sample(
    temperature=0.75,
    top_k=4,
    top_p=0.9,
    fixture_inference_limits(),
  )
}

///|
fn expected_stochastic_token(
  seed : @inference.SamplingSeed,
  sample_index : Int,
) -> Int raise {
  let limits = @sampling.SamplingLimits::new(1, 8)
  let scratch = @sampling.SamplerScratch::new(8, limits)
  @sampling.stochastic_sample_at(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    stochastic_parameters(),
    limits,
    seed,
    sample_index,
    scratch,
  ).token_id()
}

///|
fn prepare_cycle_for_branch(
  sequence : UInt64,
  branch : Int,
  model_plan : @plan.ModelPlan,
  page : @page_allocator.PageId,
) -> PreparedCycle raise {
  let limits = fixture_worker_limits()
  if branch < CYCLE_ORDINARY_PREFILL || branch > CYCLE_STOCHASTIC_DECODE {
    abort("invalid allocation evidence branch")
  }
  let request_group = (sequence - 1UL) / 3UL
  let request_id = request_group + 1UL
  let request_generation = request_group
  let generation = @inference.RequestGeneration::from_wire(request_generation)
  let seed = @inference.SamplingSeed::new(request_id + 1000UL)
  let owner = @worker_protocol.SchedulePlanBuffer::new(limits)
  let mut row_position = 0
  let mut sample_index = 0
  let mut sampling_seed = 1UL
  let mut expected_kind : @worker_protocol.CompletionEntryKind = PrefillCompletion
  let mut expected_processed_tokens = 1
  let mut expected_token = 0
  let mut expected_input_token = 0
  let mut expected_readbacks = 0UL
  if branch == CYCLE_ORDINARY_PREFILL {
    owner.append_prefill(
      request_id=@inference.RequestId::new(request_id),
      request_generation=generation,
      sequence_token_start=0,
      tokens=[0],
      pages=[page],
      capabilities=model_plan.required_capabilities(),
      completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
    )
  } else if branch == CYCLE_GREEDY_FINAL_PREFILL ||
    branch == CYCLE_STOCHASTIC_FINAL_PREFILL {
    row_position = 1
    sample_index = 0
    sampling_seed = seed.value()
    expected_kind = FinalPrefillCompletion
    expected_input_token = 1
    expected_readbacks = 1UL
    let sampling = if branch == CYCLE_GREEDY_FINAL_PREFILL {
      @inference.SamplingParameters::greedy()
    } else {
      stochastic_parameters()
    }
    if branch == CYCLE_STOCHASTIC_FINAL_PREFILL {
      expected_token = expected_stochastic_token(seed, sample_index)
    }
    owner.append_final_prefill(
      request_id=@inference.RequestId::new(request_id),
      request_generation=generation,
      sequence_token_start=row_position,
      tokens=[1],
      pages=[page],
      capabilities=model_plan.required_capabilities(),
      sampling~,
      sampling_seed=seed,
      sample_index~,
      completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
    )
  } else {
    row_position = 2
    sample_index = 1
    sampling_seed = seed.value()
    expected_kind = DecodeCompletion
    expected_processed_tokens = 0
    expected_readbacks = 1UL
    let sampling = if branch == CYCLE_GREEDY_DECODE {
      @inference.SamplingParameters::greedy()
    } else {
      stochastic_parameters()
    }
    if branch == CYCLE_GREEDY_DECODE {
      expected_input_token = expected_stochastic_token(seed, 0)
    } else {
      expected_input_token = 0
      expected_token = expected_stochastic_token(seed, sample_index)
    }
    owner.append_decode(
      request_id=@inference.RequestId::new(request_id),
      request_generation=generation,
      sequence_token_position=row_position,
      input_token=expected_input_token,
      pages=[page],
      capabilities=model_plan.required_capabilities(),
      sampling~,
      sampling_seed=seed,
      sample_index~,
      completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
    )
  }
  let model_generation = @worker_protocol.ModelPlanGeneration::first()
  let plan = owner.submit(
    sequence=@worker_protocol.PlanSequence::from_wire(sequence),
    model_plan_generation=model_generation,
    token_budget=1,
    if sequence == 1UL {
      None
    } else {
      Some(@worker_protocol.PlanSequence::from_wire(sequence - 1UL))
    },
    model_generation,
  )
  let frame_owner = @worker_wire.PlanFrameBuffer::new(
    limits,
    fixture_inference_limits(),
  )
  {
    plan_owner: owner,
    plan,
    frame: frame_owner.encode_plan(plan),
    branch,
    request_id,
    request_generation,
    row_position,
    sample_index,
    sampling_seed,
    page_index: page.index(),
    page_generation: page.generation(),
    capability_count: model_plan.required_capabilities().length(),
    expected_input_token,
    expected_kind,
    expected_processed_tokens,
    expected_token,
    expected_readbacks,
  }
}

///|
fn prepare_cycle(
  sequence : UInt64,
  model_plan : @plan.ModelPlan,
  page : @page_allocator.PageId,
) -> PreparedCycle raise {
  let phase = ((sequence - 1UL) % 6UL).to_int()
  let branch = if phase == 0 || phase == 3 {
    CYCLE_ORDINARY_PREFILL
  } else if phase == 1 {
    CYCLE_GREEDY_FINAL_PREFILL
  } else if phase == 2 {
    CYCLE_STOCHASTIC_DECODE
  } else if phase == 4 {
    CYCLE_STOCHASTIC_FINAL_PREFILL
  } else {
    CYCLE_GREEDY_DECODE
  }
  prepare_cycle_for_branch(sequence, branch, model_plan, page)
}

///|
fn prepare_cycles(
  count : Int,
  model_plan : @plan.ModelPlan,
) -> PreparedMatrix raise {
  if count <= 0 || count % 3 != 0 {
    abort("allocation evidence matrix must contain complete request groups")
  }
  let group_count = count / 3
  let allocator = @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=group_count,
      max_run_pages=1,
      max_active_references_per_page=1,
      max_cached_references_per_page=1,
    ),
  )
  let first_page = allocator.allocate_one()
  let pages = FixedArray::make(group_count, first_page)
  for index in 1..<group_count {
    pages[index] = allocator.allocate_one()
  }
  let first = prepare_cycle(1UL, model_plan, pages[0])
  let cycles = FixedArray::make(count, first)
  for index in 1..<count {
    cycles[index] = prepare_cycle(
      (index + 1).to_uint64(),
      model_plan,
      pages[index / 3],
    )
  }
  { cycles, allocator, pages }
}

///|
fn PreparedMatrix::release_pages(self : PreparedMatrix) -> Unit raise {
  for page in self.pages {
    self.allocator.release_active(page)
  }
  if self.allocator.allocated_page_count() != 0 ||
    self.allocator.free_page_count() != self.pages.length() {
    alloc_probe_abort()
  }
}

///|
fn authenticate_cycle(cycle : PreparedCycle) -> Unit raise {
  if cycle.frame.sequence_value() != cycle.plan.sequence_wire() ||
    cycle.frame.model_generation_value() != 1UL ||
    cycle.frame.token_budget() != 1 ||
    cycle.frame.token_count() != 1 ||
    cycle.frame.page_count() != 1 ||
    cycle.frame.capability_count() != cycle.capability_count ||
    cycle.frame.token_at(0) != cycle.expected_input_token ||
    cycle.frame.page_index_at(0) != cycle.page_index ||
    cycle.frame.page_generation_at(0) != cycle.page_generation {
    alloc_probe_abort()
  }
  if cycle.branch == CYCLE_ORDINARY_PREFILL ||
    cycle.branch == CYCLE_GREEDY_FINAL_PREFILL ||
    cycle.branch == CYCLE_STOCHASTIC_FINAL_PREFILL {
    let row = cycle.frame.prefill_row(0)
    let stochastic = cycle.branch == CYCLE_STOCHASTIC_FINAL_PREFILL
    if cycle.frame.prefill_count() != 1 ||
      cycle.frame.decode_count() != 0 ||
      row.request_id_value() != cycle.request_id ||
      row.request_generation_value() != cycle.request_generation ||
      row.sequence_token_start() != cycle.row_position ||
      row.token_start() != 0 ||
      row.token_length() != 1 ||
      row.page_start() != 0 ||
      row.page_length() != 1 ||
      row.capability_start() != 0 ||
      row.capability_length() != cycle.capability_count ||
      row.sample_index() != cycle.sample_index ||
      row.completion_slot() != 0 ||
      row.sampling_seed_value() != cycle.sampling_seed ||
      row.produces_token() != (cycle.branch != CYCLE_ORDINARY_PREFILL) ||
      row.sampling_mode_value() != (if stochastic { 1 } else { 0 }) ||
      row.sampling_temperature() != (if stochastic { 0.75 } else { 1.0 }) ||
      row.sampling_has_top_k() != stochastic ||
      row.sampling_has_top_p() != stochastic ||
      (
        stochastic &&
        (row.sampling_top_k_value() != 4 || row.sampling_top_p_value() != 0.9)
      ) {
      alloc_probe_abort()
    }
  } else {
    let row = cycle.frame.decode_row(0)
    let stochastic = cycle.branch == CYCLE_STOCHASTIC_DECODE
    if cycle.frame.prefill_count() != 0 ||
      cycle.frame.decode_count() != 1 ||
      row.request_id_value() != cycle.request_id ||
      row.request_generation_value() != cycle.request_generation ||
      row.sequence_token_position() != cycle.row_position ||
      row.token_start() != 0 ||
      row.page_start() != 0 ||
      row.page_length() != 1 ||
      row.capability_start() != 0 ||
      row.capability_length() != cycle.capability_count ||
      row.sample_index() != cycle.sample_index ||
      row.completion_slot() != 0 ||
      row.sampling_seed_value() != cycle.sampling_seed ||
      row.sampling_mode_value() != (if stochastic { 1 } else { 0 }) ||
      row.sampling_temperature() != (if stochastic { 0.75 } else { 1.0 }) ||
      row.sampling_has_top_k() != stochastic ||
      row.sampling_has_top_p() != stochastic ||
      (
        stochastic &&
        (row.sampling_top_k_value() != 4 || row.sampling_top_p_value() != 0.9)
      ) {
      alloc_probe_abort()
    }
  }
}

///|
fn execute_cycle(
  owner : @device_worker.DeviceWorkerOwner,
  completion_owner : @worker_wire.CompletionFrameBuffer,
  cycle : PreparedCycle,
  operation_count : UInt64,
) -> Unit raise {
  authenticate_cycle(cycle)
  let copies = fake_copy_count()
  let launches = fake_launch_count()
  let syncs = fake_sync_count()
  let reads = fake_readback_count()
  let creates = fake_resource_create_count()
  let closes = fake_resource_close_count()
  let writer = completion_owner.writer_for(cycle.frame)
  let completion = owner.execute_frame(cycle.frame, writer)
  completion.validate_for(cycle.plan)
  if completion.sequence_value() != cycle.plan.sequence_wire() ||
    completion.model_generation_value() != 1UL ||
    completion.entry_count() != 1 {
    alloc_probe_abort()
  }
  let entry = completion.entry(0)
  if entry.kind() != cycle.expected_kind ||
    entry.request_id_value() != cycle.request_id ||
    entry.request_generation_value() != cycle.request_generation ||
    entry.processed_tokens() != cycle.expected_processed_tokens ||
    entry.token_id() != cycle.expected_token ||
    fake_copy_count() - copies != 8UL ||
    fake_launch_count() - launches != operation_count ||
    fake_sync_count() - syncs != operation_count ||
    fake_readback_count() - reads != cycle.expected_readbacks ||
    fake_resource_create_count() != creates ||
    fake_resource_close_count() != closes {
    alloc_probe_abort()
  }
  let retired = cycle.plan_owner.retire(cycle.plan)
  cycle.plan_owner.reset(retired)
}
