///|
test "attention phase row domains partition ragged mixed and empty batches" {
  for case in [(0, 0, 0), (0, 4, 4), (3, 0, 17), (3, 2, 19)] {
    let (prefill, decode, tokens) = case
    let phase = attention_phase_rows(prefill, decode, tokens)
    assert_eq(phase.prefill_rows, prefill)
    assert_eq(phase.prefill_tokens, tokens - decode)
    let seen = FixedArray::make(tokens, 0)
    for token in 0..<phase.prefill_tokens {
      seen[token] += 1
    }
    for token in phase.prefill_tokens..<tokens {
      seen[token] += 1
    }
    for count in seen {
      assert_eq(count, 1)
    }
  }
}

///|
test "prefill counts share one descriptor upload without modifying original views" {
  let host = make_host_buffers(step_device_limits(max_tokens=32))
  let original_cells = host.counts.length() / INT32_BYTES
  for cell in 0..<original_cells {
    write_i32(host.counts, cell, 41 + cell)
  }
  let summary : DeviceStepSummary = {
    sequence_value: 1UL,
    prefill_count: 2,
    decode_count: 2,
    row_count: 4,
    token_count: 15,
    page_count: 9,
  }
  publish_prefill_attention_counts(host, summary)
  for cell in 0..<original_cells {
    if cell < PREFILL_ATTENTION_COUNTS_CELL || cell >= OUTPUT_ROW_ENDS_CELL {
      assert_eq(read_descriptor_i32(host.counts, cell), 41 + cell)
    }
  }
  for cell, value in [2, 0, 2, 13, 9] {
    assert_eq(
      read_descriptor_i32(host.counts, PREFILL_ATTENTION_COUNTS_CELL + cell),
      value,
    )
  }
  publish_prefill_attention_counts(host, empty_device_step_summary())
  for cell in 0..<5 {
    assert_eq(
      read_descriptor_i32(host.counts, PREFILL_ATTENTION_COUNTS_CELL + cell),
      0,
    )
  }
  for projected in [false, true] {
    host.project_output_rows = projected
    for outputs in [0, 1, 4] {
      host.output_rows = outputs
      let uploaded = uploaded_count_cells(host, summary)
      assert_true(PREFILL_ATTENTION_COUNTS_CELL + 5 <= uploaded)
      assert_true(uploaded <= original_cells)
    }
  }
}

///|
test "attention phase rewrite preserves effect order and inserts decode after merge" {
  let source : ReadOnlyArray[String] = [
    "qkv", "kv_write", "attention_partial", "attention_merge", "output_projection",
  ]
  let result = specialize_attention_phase_list(source, step => {
    match step {
      "attention_partial" => Some(("prefill_partial", None))
      "attention_merge" => Some(("prefill_merge", Some("decode")))
      _ => None
    }
  })
  assert_eq(
    result,
    Some(
      ReadOnlyArray::from_array([
        "qkv", "kv_write", "prefill_partial", "prefill_merge", "decode", "output_projection",
      ]),
    ),
  )
  assert_eq(specialize_attention_phase_list(source, _ => None), None)
  let empty : ReadOnlyArray[String] = []
  assert_eq(specialize_attention_phase_list(empty, _ => None), None)
}

///|
fn attention_phase_ragged_fixture() -> @worker_protocol.SubmittedSchedulePlan raise {
  let allocator = step_allocator()
  let buffer = @worker_protocol.SchedulePlanBuffer::new(step_protocol_limits())
  for row in 0..<2 {
    buffer.append_prefill(
      request_id=step_request(row.to_uint64() + 1UL),
      request_generation=@inference.RequestGeneration::initial(),
      sequence_token_start=0,
      tokens=Array::make(row + 1, row + 10),
      pages=[allocator.allocate_one()],
      capabilities=step_required_capability_array(),
      completion_slot=step_slot(row),
    )
  }
  buffer.append_decode(
    request_id=step_request(3UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_position=4,
    input_token=13,
    pages=[allocator.allocate_one(), allocator.allocate_one()],
    capabilities=step_required_capability_array(),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=step_sampling_seed(),
    sample_index=0,
    completion_slot=step_slot(2),
  )
  submit_step_buffer(buffer, token_budget=4)
}

///|
test "phase counts preserve ragged submitted and wire rows in both projection modes" {
  let submitted = attention_phase_ragged_fixture()
  let frame = encoded_step_frame(submitted)
  let limits = step_device_limits()
  for projected in [false, true] {
    let host = make_host_buffers(limits)
    let wire = make_host_buffers(limits)
    host.project_output_rows = projected
    wire.project_output_rows = projected
    ignore(
      preflight_and_write(
        submitted,
        1UL,
        step_test_identity(),
        step_test_layout(),
        limits,
        host,
        step_required_capabilities(),
        true,
      ),
    )
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
    let phase_start = PREFILL_ATTENTION_COUNTS_CELL
    for cell, value in [2, 0, 2, 3, 4] {
      assert_eq(read_descriptor_i32(host.counts, phase_start + cell), value)
    }
    for cell, value in [2, 1, 3, 4, 4] {
      assert_eq(read_descriptor_i32(host.counts, cell), value)
    }
    for row, offset in [0, 1, 3, 4] {
      assert_eq(read_descriptor_i32(host.query_row_offsets, row), offset)
      assert_eq(read_descriptor_i32(wire.query_row_offsets, row), offset)
    }
    assert_eq(host.output_indices[2], 0)
    assert_eq(read_descriptor_i32(host.token_positions, 3), 4)
    assert_eq(read_descriptor_i32(host.page_offsets, 2), 2)
    assert_eq(read_descriptor_i32(host.page_offsets, 3), 4)
  }
}
