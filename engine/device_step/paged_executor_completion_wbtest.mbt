///|
fn completion_test_scratch(size : Int) -> PagedCompletionScratch {
  let limits = try! @sampling.SamplingLimits::new(2, size)
  {
    final_output_offset: 0L,
    row_bytes: size.to_int64() * 2L,
    vocabulary_size: size,
    limits,
    sampler: try! @sampling.SamplerScratch::new(size, limits),
    encoded_logits: FixedArray::make(size * 2, b'\x00'),
    logits: FixedArray::make(size, 0.0),
    selected_tokens: FixedArray::make(2, 0),
  }
}

///|
test "BF16 terminal row decodes exactly and greedy ties choose lowest token" {
  let scratch = completion_test_scratch(3)
  scratch.encoded_logits[0] = b'\x80'
  scratch.encoded_logits[1] = b'\x3f'
  scratch.encoded_logits[2] = b'\x00'
  scratch.encoded_logits[3] = b'\x40'
  scratch.encoded_logits[4] = b'\x00'
  scratch.encoded_logits[5] = b'\x40'
  decode_paged_bf16_logits(scratch)
  assert_eq(scratch.logits[0], 1.0)
  assert_eq(scratch.logits[1], 2.0)
  assert_eq(scratch.logits[2], 2.0)
  assert_eq(
    try! select_paged_logit(
      scratch,
      @inference.SamplingParameters::greedy(),
      7UL,
      0,
    ),
    1,
  )
}

///|
test "non-finite BF16 terminal output fails closed" {
  let scratch = completion_test_scratch(1)
  scratch.encoded_logits[0] = b'\x80'
  scratch.encoded_logits[1] = b'\x7f'
  decode_paged_bf16_logits(scratch)
  try
    select_paged_logit(scratch, @inference.SamplingParameters::greedy(), 7UL, 0)
  catch {
    ExecutorSampling(NonFiniteLogit(row=0, token_id=0)) => ()
    _ => fail("unexpected non-finite terminal-output error")
  } noraise {
    _ => fail("non-finite terminal output was accepted")
  }
}

///|
test "completion scratch rejects a graph whose retained output is not logits" {
  let metadata = blueprint_metadata()
  let plan = blueprint_model_plan(metadata)
  let layout = blueprint_layout(plan)
  let memory = blueprint_memory_plan(metadata, plan, layout, blueprint_target())
  try prepare_paged_completion_scratch(plan, memory, blueprint_limits()) catch {
    InvalidExecutor(SamplingOutput) => ()
    _ => fail("unexpected non-logit final-output error")
  } noraise {
    _ => fail("hidden-state output was admitted as vocabulary logits")
  }
}

///|
test "completion append preserves intermediate and sampled row semantics" {
  let allocator = step_allocator()
  let plan_owner = @worker_protocol.SchedulePlanBuffer::new(
    step_protocol_limits(),
  )
  plan_owner.append_prefill(
    request_id=step_request(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[10, 11, 12],
    pages=[allocator.allocate_one()],
    capabilities=step_required_capability_array(),
    completion_slot=step_slot(0),
  )
  plan_owner.append_decode(
    request_id=step_request(2UL),
    request_generation=@inference.RequestGeneration::from_wire(2UL),
    sequence_token_position=4,
    input_token=13,
    pages=[allocator.allocate_one(), allocator.allocate_one()],
    capabilities=step_required_capability_array(),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=step_sampling_seed(),
    sample_index=1,
    completion_slot=step_slot(1),
  )
  let plan = submit_step_buffer(plan_owner, token_budget=4)
  let completion_owner = @worker_protocol.CompletionBuffer::new(
    step_protocol_limits(),
  )
  let selected = FixedArray::make(2, 0)
  selected[1] = 23
  let completion = try! append_paged_completion(
    plan,
    completion_owner.writer_for(plan),
    {
      sequence_value: 1UL,
      prefill_count: 1,
      decode_count: 1,
      row_count: 2,
      token_count: 4,
      page_count: 3,
    },
    selected,
  )
  let prefill = completion.entry_for_slot(step_slot(0))
  let decode = completion.entry_for_slot(step_slot(1))
  assert_eq(prefill.kind(), PrefillCompletion)
  assert_eq(prefill.processed_tokens(), 3)
  assert_eq(decode.kind(), DecodeCompletion)
  assert_eq(decode.token_id(), 23)
}

///|
test "completion append emits final-prefill sampled token" {
  let allocator = step_allocator()
  let plan_owner = @worker_protocol.SchedulePlanBuffer::new(
    step_protocol_limits(),
  )
  plan_owner.append_final_prefill(
    request_id=step_request(3UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[10, 11],
    pages=[allocator.allocate_one()],
    capabilities=step_required_capability_array(),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=step_sampling_seed(),
    sample_index=0,
    completion_slot=step_slot(0),
  )
  let plan = submit_step_buffer(plan_owner, token_budget=2)
  let completion_owner = @worker_protocol.CompletionBuffer::new(
    step_protocol_limits(),
  )
  let selected = FixedArray::make(1, 29)
  let completion = try! append_paged_completion(
    plan,
    completion_owner.writer_for(plan),
    {
      sequence_value: 1UL,
      prefill_count: 1,
      decode_count: 0,
      row_count: 1,
      token_count: 2,
      page_count: 1,
    },
    selected,
  )
  let entry = completion.entry_for_slot(step_slot(0))
  assert_eq(entry.kind(), FinalPrefillCompletion)
  assert_eq(entry.processed_tokens(), 2)
  assert_eq(entry.token_id(), 29)
}
