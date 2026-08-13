///|
test "deterministic randomized staging matches scalar CSR reference" {
  for case_index in 0..<128 {
    let tokens_per_page = 1 + (case_index * 17 + 3) % 8
    let sequence_start = (case_index * 29 + 5) % 31
    let token_length = 1 + (case_index * 37 + 7) % 19
    let sequence_length = sequence_start + token_length
    let page_count = 1 + (sequence_length - 1) / tokens_per_page
    let allocator = step_allocator(capacity=page_count)
    let tokens : Array[Int] = []
    for index in 0..<token_length {
      tokens.push((case_index * 41 + index) % 4097)
    }
    let pages : Array[@page_allocator.PageId] = []
    for _ in 0..<page_count {
      pages.push(allocator.allocate_one())
    }
    let buffer = @worker_protocol.SchedulePlanBuffer::new(
      step_protocol_limits(),
    )
    buffer.append_prefill(
      request_id=step_request(
        Int64::from_int(case_index + 1).reinterpret_as_uint64(),
      ),
      request_generation=@inference.RequestGeneration::initial(),
      sequence_token_start=sequence_start,
      tokens~,
      pages~,
      capabilities=step_required_capability_array(),
      completion_slot=step_slot(0),
    )
    let submitted = submit_step_buffer(buffer, token_budget=token_length)
    let limits = step_device_limits(max_rows=1, max_tokens=32, max_pages=64)
    let host = make_host_buffers(limits)
    let summary = preflight_then_write(
      submitted,
      1UL,
      step_test_identity(),
      step_test_layout(tokens_per_page~, pages=64),
      limits,
      host,
    )
    assert_eq(summary.row_count, 1)
    assert_eq(summary.token_count, token_length)
    assert_eq(summary.page_count, page_count)
    assert_eq(read_host_i32(host.query_row_offsets, 0), 0)
    assert_eq(read_host_i32(host.query_row_offsets, 1), token_length)
    assert_eq(read_host_i32(host.sequence_lengths, 0), sequence_length)
    assert_eq(read_host_i32(host.page_offsets, 0), 0)
    assert_eq(read_host_i32(host.page_offsets, 1), page_count)
    for index in 0..<token_length {
      assert_eq(read_host_i32(host.token_ids, index), tokens[index])
      assert_eq(
        read_host_i32(host.token_positions, index),
        sequence_start + index,
      )
    }
    for index in 0..<page_count {
      assert_eq(read_host_i32(host.physical_page_indices, index), index)
      assert_true(pages[index].generation() > 0)
    }
  }
}
