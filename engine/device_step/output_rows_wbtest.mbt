///|
fn output_rows_fixture(
  produces : ArrayView[Bool],
  sequence? : UInt64 = 1UL,
) -> @worker_protocol.SubmittedSchedulePlan raise {
  let allocator = step_allocator()
  let buffer = @worker_protocol.SchedulePlanBuffer::new(step_protocol_limits())
  for index, produces in produces {
    let tokens = Array::make(index + 1, index + 1)
    let pages = [allocator.allocate_one()]
    if produces {
      buffer.append_final_prefill(
        request_id=step_request(index.to_uint64() + 1UL),
        request_generation=@inference.RequestGeneration::initial(),
        sequence_token_start=0,
        tokens~,
        pages~,
        capabilities=step_required_capability_array(),
        sampling=@inference.SamplingParameters::greedy(),
        sampling_seed=step_sampling_seed(),
        sample_index=0,
        completion_slot=step_slot(index),
      )
    } else {
      buffer.append_prefill(
        request_id=step_request(index.to_uint64() + 1UL),
        request_generation=@inference.RequestGeneration::initial(),
        sequence_token_start=0,
        tokens~,
        pages~,
        capabilities=step_required_capability_array(),
        completion_slot=step_slot(index),
      )
    }
  }
  submit_step_buffer(
    buffer,
    sequence~,
    token_budget=produces.length() * (produces.length() + 1) / 2,
  )
}

///|
test "terminal projection preserves ragged row identity and original descriptors" {
  let limits = step_device_limits(max_tokens=16)
  let host = make_host_buffers(limits)
  let wire = make_host_buffers(limits)
  for
    produces in [
      [true, false, true, false],
      [false, true, false, false],
      [false, false, false, false],
      [true, true, true, true],
    ] {
    let plan = output_rows_fixture(produces)
    let summary = preflight_and_write(
      plan,
      1UL,
      step_test_identity(),
      step_test_layout(),
      limits,
      host,
      step_required_capabilities(),
      true,
    )
    let frame = encoded_step_frame(plan)
    ignore(
      preflight_frame_and_write(
        frame,
        1UL,
        step_test_identity(),
        step_test_layout(),
        limits,
        wire,
        step_required_capabilities(),
        true,
      ),
    )
    assert_eq(host.counts, wire.counts)
    assert_eq(host.output_indices, wire.output_indices)
    assert_eq(read_descriptor_i32(host.counts, 2), 4)
    assert_eq(read_descriptor_i32(host.counts, 3), 10)
    assert_eq(read_descriptor_i32(host.query_row_offsets, 4), 10)
    let mut output = 0
    let mut end = 0
    for index, producing in produces {
      end += index + 1
      if producing {
        assert_eq(host.output_indices[index], output)
        output += 1
        assert_eq(
          read_descriptor_i32(host.counts, OUTPUT_ROW_ENDS_CELL + output),
          end,
        )
      } else {
        assert_eq(host.output_indices[index], -1)
      }
    }
    assert_eq(host.output_rows, output)
    assert_eq(
      read_descriptor_i32(host.counts, OUTPUT_COUNTS_CELL + 2),
      if output == 1 {
        2
      } else {
        output
      },
    )
    if output == 1 {
      assert_eq(read_descriptor_i32(host.counts, OUTPUT_ROW_ENDS_CELL + 2), 0)
    }
    assert_eq(
      output_physical_rows(host, summary),
      if output == 1 {
        2
      } else {
        output
      },
    )
  }
}

///|
test "compacted device sampler preserves seed index and active parameters" {
  let plan = output_rows_fixture([false, true, false, true])
  let frame = encoded_step_frame(plan)
  let limits = step_device_limits(max_tokens=16)
  let host = make_host_buffers(limits)
  let summary = preflight_and_write(
    plan,
    1UL,
    step_test_identity(),
    step_test_layout(),
    limits,
    host,
    step_required_capabilities(),
    true,
  )
  let sampling = @luna_sampling_strategy.plan_device_sampling(
    maximum_rows=4,
    vocabulary_size=128,
    maximum_top_k=4,
    maximum_workspace_bytes=1048576L,
  )
  let original : PagedDeviceSamplingScratch = {
    plan: sampling,
    parameters: FixedArray::make(160, b'\x00'),
    results: FixedArray::make(32, b'\x00'),
  }
  let compact : PagedDeviceSamplingScratch = {
    plan: sampling,
    parameters: FixedArray::make(160, b'\x7f'),
    results: FixedArray::make(32, b'\x00'),
  }
  let wire : PagedDeviceSamplingScratch = {
    plan: sampling,
    parameters: FixedArray::make(160, b'\x7f'),
    results: FixedArray::make(32, b'\x00'),
  }
  stage_submitted_device_sampling_parameters(original, plan, summary)
  stage_submitted_device_sampling_parameters(
    compact,
    plan,
    summary,
    compact=true,
  )
  stage_wire_device_sampling_parameters(wire, frame, summary, compact=true)
  assert_eq(compact.parameters, wire.parameters)
  for row in 0..<2 {
    for byte in 0..<40 {
      assert_eq(
        compact.parameters[row * 40 + byte],
        original.parameters[(row * 2 + 1) * 40 + byte],
      )
    }
  }
}
