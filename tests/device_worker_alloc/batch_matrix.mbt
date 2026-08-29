///|
struct PreparedBatchCycle {
  plan_owner : @worker_protocol.SchedulePlanBuffer
  plan : @worker_protocol.SubmittedSchedulePlan
  frame : @worker_wire.ValidatedPlanFrame
  expected_ids : FixedArray[UInt64]
  expected_generations : FixedArray[UInt64]
  expected_slots : FixedArray[Int]
  expected_kinds : FixedArray[@worker_protocol.CompletionEntryKind]
  expected_processed : FixedArray[Int]
  expected_tokens : FixedArray[Int]
  pages : FixedArray[@page_allocator.PageId]
}

///|
struct PreparedBatchMatrix {
  cycles : FixedArray[PreparedBatchCycle]
  allocator : @page_allocator.PageAllocator
  pages : FixedArray[@page_allocator.PageId]
}

///|
fn oracle_counter_state(seed : UInt64, sample_index : Int) -> UInt64 {
  let mut value = seed + (sample_index + 1).to_uint64() * 0x9e3779b97f4a7c15UL
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9UL
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebUL
  value = value ^ (value >> 31)
  if value == 0UL {
    0x9e3779b97f4a7c15UL
  } else {
    value
  }
}

///|
fn oracle_logit(token_row : Int, token_id : Int) -> Double {
  let first = token_row % 5
  if token_id >= first && token_id < first + 4 {
    (4 - (token_id - first)).to_double()
  } else {
    0.0
  }
}

///|
fn oracle_preferred_before(token_row : Int, left : Int, right : Int) -> Bool {
  let left_logit = oracle_logit(token_row, left)
  let right_logit = oracle_logit(token_row, right)
  left_logit > right_logit || (left_logit == right_logit && left < right)
}

///|
fn oracle_sorted_tokens(token_row : Int) -> FixedArray[Int] {
  let tokens = FixedArray::make(8, 0)
  for token in 0..<8 {
    tokens[token] = token
  }
  for index in 1..<8 {
    let candidate = tokens[index]
    let mut cursor = index
    while cursor > 0 &&
          oracle_preferred_before(token_row, candidate, tokens[cursor - 1]) {
      tokens[cursor] = tokens[cursor - 1]
      cursor -= 1
    }
    tokens[cursor] = candidate
  }
  tokens
}

///|
fn oracle_stochastic_token(
  seed : UInt64,
  sample_index : Int,
  token_row : Int,
) -> Int {
  let tokens = oracle_sorted_tokens(token_row)
  let weights = FixedArray::make(4, 0.0)
  let maximum = oracle_logit(token_row, tokens[0])
  let mut top_k_total = 0.0
  for index in 0..<4 {
    let weight = @math.exp(
      (oracle_logit(token_row, tokens[index]) - maximum) / 0.75,
    )
    weights[index] = weight
    top_k_total += weight
  }
  let nucleus_threshold = top_k_total * 0.9
  let mut nucleus_count = 0
  let mut cumulative = 0.0
  while nucleus_count < 4 && cumulative < nucleus_threshold {
    cumulative += weights[nucleus_count]
    nucleus_count += 1
  }
  let mut retained_total = 0.0
  for index in 0..<nucleus_count {
    retained_total += weights[index]
  }
  let unit = (oracle_counter_state(seed, sample_index) >> 11).to_double() *
    1.1102230246251565e-16
  let threshold = unit * retained_total
  let mut selected = nucleus_count - 1
  cumulative = 0.0
  for index in 0..<nucleus_count {
    cumulative += weights[index]
    if threshold < cumulative {
      selected = index
      break
    }
  }
  tokens[selected]
}

///|
fn oracle_greedy_token(token_row : Int) -> Int {
  oracle_sorted_tokens(token_row)[0]
}

///|
fn authenticate_oracle_fixture() -> Unit {
  let token_row = 6
  let tokens = oracle_sorted_tokens(token_row)
  let first_weight = @math.exp(
    (oracle_logit(token_row, tokens[0]) - oracle_logit(token_row, tokens[0])) /
    0.75,
  )
  let second_weight = @math.exp(
    (oracle_logit(token_row, tokens[1]) - oracle_logit(token_row, tokens[0])) /
    0.75,
  )
  if oracle_greedy_token(token_row) != 1 ||
    tokens[0] != 1 ||
    tokens[1] != 2 ||
    first_weight == second_weight {
    abort("nonuniform batch oracle fixture drifted")
  }
}

///|
fn batch_sampling(stochastic : Bool) -> @inference.SamplingParameters raise {
  if stochastic {
    stochastic_parameters()
  } else {
    @inference.SamplingParameters::greedy()
  }
}

///|
fn prepare_batch_cycle(
  sequence : UInt64,
  model_plan : @plan.ModelPlan,
  pages : FixedArray[@page_allocator.PageId],
) -> PreparedBatchCycle raise {
  let limits = fixture_worker_limits()
  let capabilities = model_plan.required_capabilities()
  if capabilities.length() != FIXTURE_CAPABILITIES_PER_ROW ||
    pages.length() != FIXTURE_MAX_PAGES {
    abort("batch allocation fixture geometry drifted")
  }
  let owner = @worker_protocol.SchedulePlanBuffer::new(limits)
  let base = sequence * 16UL
  let generations = FixedArray::from_array([
    sequence * 4UL,
    sequence * 4UL + 1UL,
    sequence * 4UL + 2UL,
    sequence * 4UL + 3UL,
  ])
  let ids = FixedArray::from_array([
    base + 1UL,
    base + 2UL,
    base + 3UL,
    base + 4UL,
  ])
  let slots = FixedArray::from_array([2, 0, 3, 1])
  let alternate = sequence % 2UL == 0UL
  let row1_stochastic = alternate
  let row2_stochastic = !alternate
  let row3_stochastic = alternate
  let seed1 = base + 101UL
  let seed2 = base + 102UL
  let seed3 = base + 103UL
  let sample1 = (sequence % 8UL).to_int()
  let sample2 = ((sequence + 1UL) % 8UL).to_int()
  let sample3 = ((sequence + 2UL) % 8UL).to_int()
  owner.append_prefill(
    request_id=@inference.RequestId::new(ids[0]),
    request_generation=@inference.RequestGeneration::from_wire(generations[0]),
    sequence_token_start=2,
    tokens=[0, 1, 2],
    pages=[pages[0], pages[1]],
    capabilities~,
    completion_slot=@worker_protocol.CompletionSlot::new(slots[0], limits),
  )
  owner.append_final_prefill(
    request_id=@inference.RequestId::new(ids[1]),
    request_generation=@inference.RequestGeneration::from_wire(generations[1]),
    sequence_token_start=3,
    tokens=[3, 4, 5],
    pages=[pages[2], pages[3]],
    capabilities~,
    sampling=batch_sampling(row1_stochastic),
    sampling_seed=@inference.SamplingSeed::new(seed1),
    sample_index=sample1,
    completion_slot=@worker_protocol.CompletionSlot::new(slots[1], limits),
  )
  owner.append_final_prefill(
    request_id=@inference.RequestId::new(ids[2]),
    request_generation=@inference.RequestGeneration::from_wire(generations[2]),
    sequence_token_start=7,
    tokens=[6],
    pages=[pages[4], pages[5]],
    capabilities~,
    sampling=batch_sampling(row2_stochastic),
    sampling_seed=@inference.SamplingSeed::new(seed2),
    sample_index=sample2,
    completion_slot=@worker_protocol.CompletionSlot::new(slots[2], limits),
  )
  owner.append_decode(
    request_id=@inference.RequestId::new(ids[3]),
    request_generation=@inference.RequestGeneration::from_wire(generations[3]),
    sequence_token_position=7,
    input_token=7,
    pages=[pages[6], pages[7]],
    capabilities~,
    sampling=batch_sampling(row3_stochastic),
    sampling_seed=@inference.SamplingSeed::new(seed3),
    sample_index=sample3,
    completion_slot=@worker_protocol.CompletionSlot::new(slots[3], limits),
  )
  let generation = @worker_protocol.ModelPlanGeneration::first()
  let plan = owner.submit(
    sequence=@worker_protocol.PlanSequence::from_wire(sequence),
    model_plan_generation=generation,
    token_budget=FIXTURE_MAX_TOKENS,
    if sequence == 1UL {
      None
    } else {
      Some(@worker_protocol.PlanSequence::from_wire(sequence - 1UL))
    },
    generation,
  )
  let frame_owner = @worker_wire.PlanFrameBuffer::new(
    limits,
    fixture_inference_limits(),
  )
  {
    plan_owner: owner,
    plan,
    frame: frame_owner.encode_plan(plan),
    expected_ids: ids,
    expected_generations: generations,
    expected_slots: slots,
    expected_kinds: FixedArray::from_array([
      PrefillCompletion,
      FinalPrefillCompletion,
      FinalPrefillCompletion,
      DecodeCompletion,
    ]),
    expected_processed: FixedArray::from_array([3, 3, 1, 0]),
    expected_tokens: FixedArray::from_array([
      0,
      if row1_stochastic {
        oracle_stochastic_token(seed1, sample1, 5)
      } else {
        oracle_greedy_token(5)
      },
      if row2_stochastic {
        oracle_stochastic_token(seed2, sample2, 6)
      } else {
        oracle_greedy_token(6)
      },
      if row3_stochastic {
        oracle_stochastic_token(seed3, sample3, 7)
      } else {
        oracle_greedy_token(7)
      },
    ]),
    pages,
  }
}

///|
fn prepare_batch_matrix(
  count : Int,
  model_plan : @plan.ModelPlan,
) -> PreparedBatchMatrix raise {
  if count <= 0 {
    abort("batch allocation evidence requires a positive cycle count")
  }
  authenticate_oracle_fixture()
  let allocator = @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=FIXTURE_MAX_PAGES,
      max_run_pages=1,
      max_active_references_per_page=1,
      max_cached_references_per_page=1,
    ),
  )
  let first = allocator.allocate_one()
  let pages = FixedArray::make(FIXTURE_MAX_PAGES, first)
  for index in 1..<FIXTURE_MAX_PAGES {
    pages[index] = allocator.allocate_one()
  }
  for index in 0..<(FIXTURE_MAX_PAGES / 2) {
    allocator.release_active(pages[index * 2])
    pages[index * 2] = allocator.allocate_one()
  }
  let initial = prepare_batch_cycle(1UL, model_plan, pages)
  let cycles = FixedArray::make(count, initial)
  for index in 1..<count {
    cycles[index] = prepare_batch_cycle(
      (index + 1).to_uint64(),
      model_plan,
      pages,
    )
  }
  { cycles, allocator, pages }
}

///|
fn PreparedBatchMatrix::release_pages(self : PreparedBatchMatrix) -> Unit raise {
  for page in self.pages {
    self.allocator.release_active(page)
  }
  if self.allocator.allocated_page_count() != 0 ||
    self.allocator.free_page_count() != FIXTURE_MAX_PAGES {
    alloc_probe_abort()
  }
}
