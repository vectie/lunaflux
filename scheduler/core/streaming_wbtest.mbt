///|
fn wb_streaming_seed() -> Scheduler raise {
  let owner = wb_prefix_scheduler()
  try! owner.configure_streaming(
    try! @streaming_ir.Capability::new(
      slots=4,
      host_bytes=256L,
      page_bytes=64L,
      page_count=16,
      reserve_pages=16,
      restore_ns=100L,
      recompute_ns=1000L,
      separate_host_memory=true,
    ),
    1UL,
  )
  let (seed, lease) = wb_prefix_request(
    1001UL,
    [1, 2, 3, 4, 5],
    ReadWrite,
    b"scope",
  )
  ignore(try! owner.admit(seed, 0UL))
  wb_complete_one_token_request(owner, 7)
  ignore(try! owner.take_generated_token())
  ignore(try! owner.take_terminal_notice())
  try! lease.release()
  for slot in 0..<2 {
    let command = (try! owner.next_streaming_command(0UL)).unwrap()
    assert_true(command.opcode is Spill)
    assert_true((try! owner.build_next()).is_backpressured())
    try! owner.accept_streaming_reply({
      command,
      status: HostReady,
      host: { slot, generation: 1, },
    })
  }
  assert_eq(owner.page_allocator.free_page_count(), 16)
  owner
}

///|
test "prefill selects admitted shapes within one shared step workspace budget" {
  let owner = wb_prefix_scheduler()
  owner.configure_streaming(
    @streaming_ir.Capability::new(
      slots=4,
      host_bytes=256L,
      page_bytes=64L,
      page_count=16,
      reserve_pages=4,
      restore_ns=100L,
      recompute_ns=1000L,
      separate_host_memory=true,
      workspace_budget=256L,
      shapes=[
        @streaming_ir.Shape::new(tokens=1, workspace_bytes=64L),
        @streaming_ir.Shape::new(tokens=2, workspace_bytes=128L),
        @streaming_ir.Shape::new(tokens=4, workspace_bytes=512L),
      ],
    ),
    1UL,
  )
  let (first, lease1) = wb_prefix_request(
    1010UL,
    [1, 2, 3, 4, 5],
    Disabled,
    b"scope",
  )
  let (second, lease2) = wb_prefix_request(
    1011UL,
    [6, 7, 8, 9, 10],
    Disabled,
    b"scope",
  )
  let h1 = owner.admit(first, 0UL)
  let h2 = owner.admit(second, 0UL)
  let plan = wb_next_plan(owner)
  assert_eq(plan.prefill_count(), 2)
  assert_eq(plan.prefill_row(0).tokens().length(), 2)
  assert_eq(plan.prefill_row(1).tokens().length(), 2)
  let writer = owner.completion_writer(plan)
  for i in 0..<2 {
    let row = plan.prefill_row(i)
    writer.append_prefill_completed(
      slot=row.completion_slot(),
      request_id=row.request_id(),
      request_generation=row.request_generation(),
      processed_tokens=2,
    )
  }
  ignore(owner.complete(plan, writer.submit()))
  ignore(owner.cancel(h1))
  ignore(owner.cancel(h2))
  ignore(owner.take_terminal_notice())
  ignore(owner.take_terminal_notice())
  lease1.release()
  lease2.release()
}

///|
test "shared prefix ancestors spill once and cold eviction balances host slots" {
  let owner = wb_prefix_scheduler()
  let (seed, lease1) = wb_prefix_request(
    1020UL,
    [1, 2, 3, 4, 5],
    ReadWrite,
    b"scope",
  )
  ignore(owner.admit(seed, 0UL))
  wb_complete_one_token_request(owner, 7)
  ignore(owner.take_generated_token())
  ignore(owner.take_terminal_notice())
  lease1.release()
  let (child, lease2) = wb_prefix_request(
    1021UL,
    [1, 2, 3, 4, 6, 7, 8],
    ReadWrite,
    b"scope",
  )
  ignore(owner.admit(child, 0UL))
  let plan = wb_next_plan(owner)
  ignore(owner.complete(plan, wb_final_completion(owner, plan, 7)))
  ignore(owner.take_generated_token())
  ignore(owner.take_terminal_notice())
  lease2.release()
  assert_eq(owner.luna_telemetry().prefix_pages(), 3)
  owner.configure_streaming(
    @streaming_ir.Capability::new(
      slots=4,
      host_bytes=256L,
      page_bytes=64L,
      page_count=16,
      reserve_pages=16,
      restore_ns=100L,
      recompute_ns=1000L,
      separate_host_memory=true,
    ),
    1UL,
  )
  for slot in 0..<3 {
    let command = owner.next_streaming_command(0UL).unwrap()
    owner.accept_streaming_reply({
      command,
      status: HostReady,
      host: { slot, generation: 1, },
    })
  }
  assert_eq(owner.page_allocator.free_page_count(), 16)
  assert_eq(owner.streaming.unwrap().host_used, 3)
  while owner.prefix_index.unwrap().entry_count() > 0 {
    ignore(owner.evict_one_prefix())
  }
  for _ in 0..<3 {
    let command = owner.next_streaming_command(0UL).unwrap()
    assert_true(command.opcode is Evict)
    owner.accept_streaming_reply({
      command,
      status: Evicted,
      host: command.host,
    })
  }
  assert_eq(owner.streaming.unwrap().host_used, 0)
}

///|
test "deadline and transfer timeout cannot publish or reclaim live destinations" {
  let owner = wb_streaming_seed()
  let (request, lease) = wb_prefix_request(
    1030UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
  )
  let handle = owner.admit(request, 0UL)
  let command = owner.next_streaming_command(0UL).unwrap()
  owner.observe_streaming_time(100UL)
  owner.accept_streaming_reply({
    command,
    status: DeviceReady,
    host: command.host,
  })
  assert_eq(owner.free_page_count(), 16)
  assert_eq(owner.streaming.unwrap().host_used, 2)
  ignore(owner.cancel(handle))
  ignore(owner.take_terminal_notice())
  lease.release()
  let other = wb_streaming_seed()
  let (request2, lease2) = wb_prefix_request(
    1031UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
  )
  let h2 = other.admit(request2, 0UL)
  ignore(other.next_streaming_command(0UL))
  try other.observe_streaming_time(10001UL) catch {
    Invalid(..) => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("missing timeout")
  }
  assert_eq(other.free_page_count(), 14)
  ignore(other.invalidate_device_state())
  assert_eq(other.free_page_count(), 16)
  ignore(other.cancel(h2))
  ignore(other.take_terminal_notice())
  lease2.release()
}

///|
test "cold prefix restores atomically before activation and preserves output" {
  let owner = wb_streaming_seed()
  let (request, lease) = wb_prefix_request(
    1002UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
  )
  ignore(owner.admit(request, 0UL))
  let first = owner.next_streaming_command(0UL).unwrap()
  assert_true(first.opcode is Restore)
  assert_eq(owner.page_allocator.free_page_count(), 14)
  assert_true(owner.build_next().is_backpressured())
  owner.accept_streaming_reply({
    command: first,
    status: DeviceReady,
    host: first.host,
  })
  assert_eq(owner.luna_telemetry().prefix_hits(), 0UL)
  assert_false(owner.prefix_lookup.unwrap().fully_resident())
  let second = owner.next_streaming_command(0UL).unwrap()
  assert_true(second.opcode is Restore)
  owner.accept_streaming_reply({
    command: second,
    status: DeviceReady,
    host: second.host,
  })
  for _ in 0..<2 {
    let command = owner.next_streaming_command(0UL).unwrap()
    assert_true(command.opcode is Evict)
    owner.accept_streaming_reply({
      command,
      status: Evicted,
      host: command.host,
    })
  }
  assert_true(owner.next_streaming_command(0UL) is None)
  let plan = wb_next_plan(owner)
  assert_eq(plan.prefill_row(0).sequence_token_start(), 4)
  ignore(owner.complete(plan, wb_final_completion(owner, plan, 7)))
  assert_eq(owner.take_generated_token().token_id(), 7)
  assert_eq(owner.take_terminal_notice().cached_input_tokens(), 4)
  lease.release()
  assert_eq(owner.streaming.unwrap().host_used, 0)
  assert_eq(owner.page_allocator.free_page_count(), 14)
}

///|
test "cancel retains all restore destinations until acknowledged and stale replies release nothing" {
  let owner = wb_streaming_seed()
  let (request, lease) = wb_prefix_request(
    1003UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
  )
  let handle = owner.admit(request, 0UL)
  let command = owner.next_streaming_command(0UL).unwrap()
  ignore(owner.cancel(handle))
  assert_eq(owner.page_allocator.free_page_count(), 14)
  try
    owner.accept_streaming_reply({
      command: { ..command, worker: 2UL, },
      status: DeviceReady,
      host: command.host,
    })
  catch {
    Invalid(..) => ()
    _ => fail("unexpected rejection")
  } noraise {
    _ => fail("accepted old worker reply")
  }
  assert_eq(owner.page_allocator.free_page_count(), 14)
  let cancel = owner.next_streaming_command(0UL).unwrap()
  assert_true(cancel.opcode is Cancel)
  owner.accept_streaming_reply({
    command: cancel,
    status: Cancelled,
    host: command.host,
  })
  assert_eq(owner.page_allocator.free_page_count(), 16)
  assert_false(owner.streaming_busy())
  assert_eq(owner.take_terminal_notice().cached_input_tokens(), 0)
  lease.release()
}

///|
test "worker loss retires private restore pages and cold identities before replacement" {
  let owner = wb_streaming_seed()
  let (request, lease) = wb_prefix_request(
    1004UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
  )
  let handle = owner.admit(request, 0UL)
  let command = owner.next_streaming_command(0UL).unwrap()
  ignore(owner.invalidate_device_state())
  assert_eq(owner.page_allocator.free_page_count(), 16)
  assert_eq(owner.luna_telemetry().prefix_entries(), 0)
  assert_true(owner.next_streaming_command(0UL) is None)
  try owner.bind_streaming_replacement(1UL) catch {
    Invalid(..) => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("reused epoch")
  }
  owner.bind_streaming_replacement(2UL)
  try
    owner.accept_streaming_reply({
      command,
      status: DeviceReady,
      host: command.host,
    })
  catch {
    Invalid(..) => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("accepted stale reply")
  }
  ignore(owner.cancel(handle))
  ignore(owner.take_terminal_notice())
  lease.release()
}

///|
test "cold scope mismatch recomputes and memory pressure evicts host payloads" {
  let owner = wb_streaming_seed()
  let (request, lease) = wb_prefix_request(
    1005UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"other",
  )
  let handle = owner.admit(request, 0UL)
  assert_true(owner.next_streaming_command(0UL) is None)
  let plan = wb_next_plan(owner)
  assert_eq(plan.prefill_row(0).sequence_token_start(), 0)
  ignore(owner.complete(plan, wb_prefill_completion(owner, plan)))
  ignore(owner.cancel(handle))
  ignore(owner.take_terminal_notice())
  lease.release()
  ignore(owner.evict_one_prefix())
  for _ in 0..<2 {
    let command = owner.next_streaming_command(0UL).unwrap()
    assert_true(command.opcode is Evict)
    owner.accept_streaming_reply({
      command,
      status: Evicted,
      host: command.host,
    })
  }
  assert_eq(owner.streaming.unwrap().host_used, 0)
}
