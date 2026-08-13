///|
test "mixed rows produce exact CSR and SoA values without page generations" {
  let allocator = step_allocator()
  let buffer = @worker_protocol.SchedulePlanBuffer::new(step_protocol_limits())
  buffer.append_prefill(
    request_id=step_request(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[10, 11, 12],
    pages=[allocator.allocate_one()],
    capabilities=step_required_capability_array(),
    completion_slot=step_slot(0),
  )
  let decode_pages = [allocator.allocate_one(), allocator.allocate_one()]
  buffer.append_decode(
    request_id=step_request(2UL),
    request_generation=@inference.RequestGeneration::from_wire(2UL),
    sequence_token_position=4,
    input_token=13,
    pages=decode_pages,
    capabilities=step_required_capability_array(),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=step_sampling_seed(),
    sample_index=0,
    completion_slot=step_slot(1),
  )
  let submitted = submit_step_buffer(buffer, token_budget=4)
  let host = make_host_buffers(step_device_limits())
  let summary = preflight_then_write(
    submitted,
    1UL,
    step_test_identity(),
    step_test_layout(),
    step_device_limits(),
    host,
  )
  assert_eq(summary.prefill_count, 1)
  assert_eq(summary.decode_count, 1)
  assert_eq(summary.row_count, 2)
  assert_eq(summary.token_count, 4)
  assert_eq(summary.page_count, 3)
  assert_eq(read_host_i32(host.token_ids, 0), 10)
  assert_eq(read_host_i32(host.token_ids, 3), 13)
  assert_eq(read_host_i32(host.token_positions, 0), 0)
  assert_eq(read_host_i32(host.token_positions, 2), 2)
  assert_eq(read_host_i32(host.token_positions, 3), 4)
  assert_eq(read_host_i32(host.query_row_offsets, 0), 0)
  assert_eq(read_host_i32(host.query_row_offsets, 1), 3)
  assert_eq(read_host_i32(host.query_row_offsets, 2), 4)
  assert_eq(read_host_i32(host.sequence_lengths, 0), 3)
  assert_eq(read_host_i32(host.sequence_lengths, 1), 5)
  assert_eq(read_host_i32(host.page_offsets, 0), 0)
  assert_eq(read_host_i32(host.page_offsets, 1), 1)
  assert_eq(read_host_i32(host.page_offsets, 2), 3)
  assert_eq(read_host_i32(host.physical_page_indices, 0), 0)
  assert_eq(read_host_i32(host.physical_page_indices, 1), 1)
  assert_eq(read_host_i32(host.physical_page_indices, 2), 2)
  assert_eq(allocator.debug_snapshot().page_at(0).id().generation(), 1)
  assert_eq(read_host_i32(host.counts, 4), 3)
}

///|
test "page boundary requires exact ceil sequence length over page size" {
  let allocator = step_allocator()
  let exact = @worker_protocol.SchedulePlanBuffer::new(step_protocol_limits())
  exact.append_prefill(
    request_id=step_request(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[1, 2, 3, 4],
    pages=[allocator.allocate_one()],
    capabilities=step_required_capability_array(),
    completion_slot=step_slot(0),
  )
  let host = make_host_buffers(step_device_limits())
  let exact_summary = preflight_then_write(
    submit_step_buffer(exact, token_budget=4),
    1UL,
    step_test_identity(),
    step_test_layout(),
    step_device_limits(),
    host,
  )
  assert_eq(exact_summary.page_count, 1)
  let crossed = @worker_protocol.SchedulePlanBuffer::new(step_protocol_limits())
  crossed.append_prefill(
    request_id=step_request(2UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[1, 2, 3, 4, 5],
    pages=[allocator.allocate_one(), allocator.allocate_one()],
    capabilities=step_required_capability_array(),
    completion_slot=step_slot(1),
  )
  let crossed_summary = preflight_then_write(
    submit_step_buffer(crossed, token_budget=5),
    1UL,
    step_test_identity(),
    step_test_layout(),
    step_device_limits(),
    host,
  )
  assert_eq(crossed_summary.page_count, 2)
}

///|
test "incorrect page count and model generation fail before host writes" {
  let allocator = step_allocator()
  let buffer = @worker_protocol.SchedulePlanBuffer::new(step_protocol_limits())
  buffer.append_prefill(
    request_id=step_request(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[1, 2, 3, 4, 5],
    pages=[allocator.allocate_one()],
    capabilities=step_required_capability_array(),
    completion_slot=step_slot(0),
  )
  let submitted = submit_step_buffer(buffer, token_budget=5)
  let limits = step_device_limits()
  let host = make_host_buffers(limits)
  write_i32(host.counts, 0, 99)
  try
    preflight_then_write(
      submitted,
      1UL,
      step_test_identity(),
      step_test_layout(),
      limits,
      host,
    )
  catch {
    InvalidPlan(PageCount) => ()
    _ => fail("unexpected page-count failure")
  } noraise {
    _ => fail("expected page-count failure")
  }
  assert_eq(read_host_i32(host.counts, 0), 99)
  try
    preflight_then_write(
      submitted,
      2UL,
      step_test_identity(),
      step_test_layout(),
      limits,
      host,
    )
  catch {
    InvalidPlan(ModelGeneration) => ()
    _ => fail("unexpected generation failure")
  } noraise {
    _ => fail("expected generation failure")
  }
  assert_eq(read_host_i32(host.counts, 0), 99)
}

///|
test "retired submitted handle is rejected without host publication" {
  let allocator = step_allocator()
  let buffer = @worker_protocol.SchedulePlanBuffer::new(step_protocol_limits())
  buffer.append_decode(
    request_id=step_request(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_position=0,
    input_token=7,
    pages=[allocator.allocate_one()],
    capabilities=step_required_capability_array(),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=step_sampling_seed(),
    sample_index=0,
    completion_slot=step_slot(0),
  )
  let submitted = submit_step_buffer(buffer, token_budget=1)
  ignore(buffer.retire(submitted))
  let limits = step_device_limits()
  let host = make_host_buffers(limits)
  write_i32(host.counts, 0, 77)
  try
    preflight_then_write(
      submitted,
      1UL,
      step_test_identity(),
      step_test_layout(),
      limits,
      host,
    )
  catch {
    Protocol(InvalidPlanBufferState(expected=Submitted, actual=Retired)) => ()
    _ => fail("unexpected stale-plan failure")
  } noraise {
    _ => fail("expected stale plan failure")
  }
  assert_eq(read_host_i32(host.counts, 0), 77)
}

///|
test "physical page indices must fit the committed device layout" {
  let allocator = step_allocator(capacity=2)
  ignore(allocator.allocate_one())
  let outside = allocator.allocate_one()
  let buffer = @worker_protocol.SchedulePlanBuffer::new(step_protocol_limits())
  buffer.append_decode(
    request_id=step_request(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_position=0,
    input_token=7,
    pages=[outside],
    capabilities=step_required_capability_array(),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=step_sampling_seed(),
    sample_index=0,
    completion_slot=step_slot(0),
  )
  let submitted = submit_step_buffer(buffer, token_budget=1)
  let limits = step_device_limits(max_rows=1, max_tokens=1, max_pages=1)
  try
    preflight_then_write(
      submitted,
      1UL,
      step_test_identity(),
      step_test_layout(pages=1),
      limits,
      make_host_buffers(limits),
    )
  catch {
    InvalidPlan(PageIndex) => ()
    _ => fail("unexpected physical-page failure")
  } noraise {
    _ => fail("expected physical-page rejection")
  }
}

///|
test "loaded token and sequence envelopes are enforced before host writes" {
  let allocator = step_allocator()
  let buffer = @worker_protocol.SchedulePlanBuffer::new(step_protocol_limits())
  buffer.append_prefill(
    request_id=step_request(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=4,
    tokens=[11],
    pages=[allocator.allocate_one(), allocator.allocate_one()],
    capabilities=step_required_capability_array(),
    completion_slot=step_slot(0),
  )
  let submitted = submit_step_buffer(buffer, token_budget=1)
  let token_limits = step_device_limits(max_token_id=10)
  let token_host = make_host_buffers(token_limits)
  write_i32(token_host.counts, 0, 71)
  try
    preflight_then_write(
      submitted,
      1UL,
      step_test_identity(),
      step_test_layout(),
      token_limits,
      token_host,
    )
  catch {
    InvalidPlan(TokenId) => ()
    _ => fail("unexpected token-envelope failure")
  } noraise {
    _ => fail("expected token-envelope rejection")
  }
  assert_eq(read_host_i32(token_host.counts, 0), 71)
  let sequence_limits = step_device_limits(
    max_sequence_tokens=4,
    max_token_id=11,
  )
  let sequence_host = make_host_buffers(sequence_limits)
  write_i32(sequence_host.counts, 0, 73)
  try
    preflight_then_write(
      submitted,
      1UL,
      step_test_identity(),
      step_test_layout(),
      sequence_limits,
      sequence_host,
    )
  catch {
    InvalidPlan(SequenceLength) => ()
    _ => fail("unexpected sequence-envelope failure")
  } noraise {
    _ => fail("expected sequence-envelope rejection")
  }
  assert_eq(read_host_i32(sequence_host.counts, 0), 73)
}

///|
test "sequence and epoch preflight failures preserve prior host publication" {
  let allocator = step_allocator()
  let buffer = @worker_protocol.SchedulePlanBuffer::new(step_protocol_limits())
  buffer.append_decode(
    request_id=step_request(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_position=0,
    input_token=7,
    pages=[allocator.allocate_one()],
    capabilities=step_required_capability_array(),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=step_sampling_seed(),
    sample_index=0,
    completion_slot=step_slot(0),
  )
  let limits = step_device_limits()
  let host = make_host_buffers(limits)
  write_i32(host.counts, 0, 79)
  write_i32(host.token_ids, 0, 83)
  let wrong_sequence = submit_step_buffer(buffer, sequence=2UL, token_budget=1)
  try
    preflight_stage_then_write(
      wrong_sequence,
      1UL,
      step_test_identity(),
      step_test_layout(),
      limits,
      host,
      step_required_capabilities(),
      false,
      0UL,
      0UL,
    )
  catch {
    InvalidPlan(PlanSequence) => ()
    _ => fail("unexpected sequence failure")
  } noraise {
    _ => fail("expected sequence rejection")
  }
  assert_eq(read_host_i32(host.counts, 0), 79)
  assert_eq(read_host_i32(host.token_ids, 0), 83)
  try
    preflight_stage_then_write(
      wrong_sequence,
      1UL,
      step_test_identity(),
      step_test_layout(),
      limits,
      host,
      step_required_capabilities(),
      true,
      1UL,
      18446744073709551615UL,
    )
  catch {
    StageEpochOverflow => ()
    _ => fail("unexpected epoch failure")
  } noraise {
    _ => fail("expected epoch rejection")
  }
  assert_eq(read_host_i32(host.counts, 0), 79)
  assert_eq(read_host_i32(host.token_ids, 0), 83)
}

///|
test "row capabilities must equal the bound paged plan recipe" {
  let allocator = step_allocator()
  let buffer = @worker_protocol.SchedulePlanBuffer::new(step_protocol_limits())
  buffer.append_decode(
    request_id=step_request(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_position=0,
    input_token=7,
    pages=[allocator.allocate_one()],
    capabilities=[@plan.KernelCapabilityId::causal_attention()],
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=step_sampling_seed(),
    sample_index=0,
    completion_slot=step_slot(0),
  )
  let limits = step_device_limits()
  let host = make_host_buffers(limits)
  write_i32(host.counts, 0, 89)
  try
    preflight_then_write(
      submit_step_buffer(buffer, token_budget=1),
      1UL,
      step_test_identity(),
      step_test_layout(),
      limits,
      host,
    )
  catch {
    InvalidPlan(CapabilityRecipe) => ()
    _ => fail("unexpected capability-recipe failure")
  } noraise {
    _ => fail("expected capability-recipe rejection")
  }
  assert_eq(read_host_i32(host.counts, 0), 89)
}
