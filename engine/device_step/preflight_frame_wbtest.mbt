///|
fn encoded_step_frame(
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> @worker_wire.ValidatedPlanFrame raise {
  let owner = @worker_wire.PlanFrameBuffer::new(
    step_protocol_limits(),
    @inference.InferenceLimits::new(
      max_text_bytes=32,
      max_input_tokens=32,
      max_new_tokens=32,
      max_context_tokens=32,
      max_token_id=64,
      max_stop_token_ids=2,
      max_stop_strings=1,
      max_stop_string_bytes=1,
      max_trace_bytes=1,
      max_cache_scope_bytes=1,
      max_decoded_delta_bytes=1,
      max_deadline_millis=100UL,
      max_top_k=64,
      max_temperature=4.0,
    ),
  )
  owner.encode_plan(plan)
}

///|
test "wire plan preflight produces exact descriptor bytes" {
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
  buffer.append_decode(
    request_id=step_request(2UL),
    request_generation=@inference.RequestGeneration::from_wire(2UL),
    sequence_token_position=4,
    input_token=13,
    pages=[allocator.allocate_one(), allocator.allocate_one()],
    capabilities=step_required_capability_array(),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=step_sampling_seed(),
    sample_index=0,
    completion_slot=step_slot(1),
  )
  let frame = encoded_step_frame(submit_step_buffer(buffer, token_budget=4))
  let limits = step_device_limits()
  let host = make_host_buffers(limits)
  let summary = preflight_frame_and_write(
    frame,
    1UL,
    step_test_identity(),
    step_test_layout(),
    limits,
    host,
    step_required_capabilities(),
    true,
  )
  assert_eq(summary.row_count, 2)
  assert_eq(summary.token_count, 4)
  assert_eq(summary.page_count, 3)
  assert_eq(read_host_i32(host.token_ids, 0), 10)
  assert_eq(read_host_i32(host.token_ids, 3), 13)
  assert_eq(read_host_i32(host.token_positions, 3), 4)
  assert_eq(read_host_i32(host.query_row_offsets, 2), 4)
  assert_eq(read_host_i32(host.page_offsets, 2), 3)
  assert_eq(read_host_i32(host.physical_page_indices, 2), 2)
}

///|
test "wire plan preflight failures preserve prior host publication" {
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
  let frame = encoded_step_frame(submit_step_buffer(buffer, token_budget=1))
  let limits = step_device_limits(max_token_id=6)
  let host = make_host_buffers(limits)
  write_i32(host.counts, 0, 91)
  write_i32(host.token_ids, 0, 92)
  try
    preflight_frame_and_write(
      frame,
      1UL,
      step_test_identity(),
      step_test_layout(),
      limits,
      host,
      step_required_capabilities(),
      false,
    )
  catch {
    InvalidPlan(TokenId) => ()
    _ => fail("expected token rejection")
  } noraise {
    _ => fail("out-of-envelope token accepted")
  }
  assert_eq(read_host_i32(host.counts, 0), 91)
  assert_eq(read_host_i32(host.token_ids, 0), 92)
}
