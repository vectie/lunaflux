///|
test "effect variants do not silently expand a declared specialization byte bound" {
  assert_true(
    effect_capture_budget(GraphMemoryAbsent).reserve(32768) is Some(_),
  )
  assert_eq(effect_capture_budget(GraphMemoryAbsent).reserve(32769), None)
  assert_eq(effect_capture_budget(GraphMemoryBounded).reserve(1), None)
  assert_eq(effect_capture_budget(GraphMemoryUnsupported).reserve(1), None)
}

///|
test "effect-only lowering retains split effects and omits the pure suffix" {
  let operations = ReadOnlyArray::from_array(
    [0, 1, 2, 2, 3, 4, 5, 5].map(value => try! @plan.OperationId::new(value)),
  )
  assert_eq(effect_only_launch_count(operations, 3), 4)
  assert_eq(effect_only_launch_count(operations, 6), 8)
  assert_eq(effect_only_launch_count(operations, 0), 8)
  let reordered = ReadOnlyArray::from_array(
    [0, 1, 5, 2].map(value => try! @plan.OperationId::new(value)),
  )
  assert_eq(effect_only_launch_count(reordered, 3), 4)
}

///|
test "model output demand retains KV effects without requiring terminal logits" {
  let model = step_test_model_plan()
  let operations = model.operations()
  assert_eq(
    model.observed_operation_prefix(output_demand=true),
    operations.length(),
  )
  let prefix = model.observed_operation_prefix(output_demand=false)
  assert_true(prefix > 0 && prefix < operations.length())
  assert_true(operations[prefix - 1].persistent_state() != NoPersistentState)
  for index in prefix..<operations.length() {
    assert_eq(operations[index].persistent_state(), NoPersistentState)
  }
  for index in 0..<operations.length() {
    assert_eq(
      pure_suffix_counts_offset(index, prefix),
      if index < prefix {
        -1L
      } else {
        byte_capacity(PURE_SUFFIX_COUNTS_CELL)
      },
    )
  }
  // No effect boundary means no proven prefix; retain the original view.
  assert_eq(pure_suffix_counts_offset(10, 0), -1L)
}

///|
test "reused descriptor demand follows producing rows in both plan paths" {
  let host = make_host_buffers(step_device_limits())
  let wire_host = make_host_buffers(step_device_limits())
  for produces in [true, false, true, false] {
    let allocator = step_allocator()
    let buffer = @worker_protocol.SchedulePlanBuffer::new(
      step_protocol_limits(),
    )
    let pages = [allocator.allocate_one()]
    if produces {
      buffer.append_final_prefill(
        request_id=step_request(1UL),
        request_generation=@inference.RequestGeneration::initial(),
        sequence_token_start=0,
        tokens=[1, 2],
        pages~,
        capabilities=step_required_capability_array(),
        sampling=@inference.SamplingParameters::greedy(),
        sampling_seed=step_sampling_seed(),
        sample_index=0,
        completion_slot=step_slot(0),
      )
    } else {
      buffer.append_prefill(
        request_id=step_request(1UL),
        request_generation=@inference.RequestGeneration::initial(),
        sequence_token_start=0,
        tokens=[1, 2],
        pages~,
        capabilities=step_required_capability_array(),
        completion_slot=step_slot(0),
      )
    }
    let plan = submit_step_buffer(buffer, token_budget=2)
    preflight_then_write(
      plan,
      1UL,
      step_test_identity(),
      step_test_layout(),
      step_device_limits(),
      host,
    )
    |> ignore
    preflight_frame_and_write(
      encoded_step_frame(plan),
      1UL,
      step_test_identity(),
      step_test_layout(),
      step_device_limits(),
      wire_host,
      step_required_capabilities(),
      true,
    )
    |> ignore
    assert_eq(host.output_rows, if produces { 1 } else { 0 })
    assert_eq(wire_host.output_rows, host.output_rows)
    assert_eq(read_host_i32(host.counts, 3), 2)
    assert_eq(read_host_i32(wire_host.counts, 3), 2)
    assert_eq(read_host_i32(host.sequence_lengths, 0), 2)
    assert_eq(read_host_i32(wire_host.sequence_lengths, 0), 2)
    assert_eq(
      host.counts.length(),
      output_descriptor_cells(step_device_limits()) * 4,
    )
  }
}
