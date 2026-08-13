///|
test "repeated release-shaped packing reuses fixed storage across boundaries" {
  let allocator = step_allocator()
  let buffer = @worker_protocol.SchedulePlanBuffer::new(step_protocol_limits())
  buffer.append_prefill(
    request_id=step_request(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=3,
    tokens=[10, 11],
    pages=[allocator.allocate_one(), allocator.allocate_one()],
    capabilities=step_required_capability_array(),
    completion_slot=step_slot(0),
  )
  let submitted = submit_step_buffer(buffer, token_budget=2)
  let limits = step_device_limits()
  let host = make_host_buffers(limits)
  let token_storage = host.token_ids
  let page_storage = host.physical_page_indices
  let required = step_required_capabilities()
  let identity = step_test_identity()
  let layout = step_test_layout()
  for _ in 0..<512 {
    ignore(
      preflight_and_write(
        submitted, 1UL, identity, layout, limits, host, required, false,
      ),
    )
    ignore(
      preflight_and_write(
        submitted, 1UL, identity, layout, limits, host, required, true,
      ),
    )
    assert_true(physical_equal(host.token_ids, token_storage))
    assert_true(physical_equal(host.physical_page_indices, page_storage))
  }
}
