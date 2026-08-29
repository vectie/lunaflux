///|
test "device loss clears waiting shared pages before replacement reuse" {
  let owner = wb_prefix_scheduler()
  let (seed, seed_lease) = wb_prefix_request(
    981UL,
    [1, 2, 3, 4, 5],
    ReadWrite,
    b"scope",
  )
  ignore(try! owner.admit(seed, 0UL))
  wb_complete_one_token_request(owner, 7)
  ignore(try! owner.take_generated_token())
  ignore(try! owner.take_terminal_notice())
  try! seed_lease.release()

  let (waiting, waiting_lease) = wb_prefix_request(
    982UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
  )
  let handle = try! owner.admit(waiting, 0UL)
  let slot = handle.slot().index()
  assert_eq(owner.prefix_matched_tokens[slot], 0)
  assert_eq(owner.prefix_candidate_tokens[slot], 4)
  assert_eq(owner.allocated_block_table_count(), 0)
  assert_eq(owner.luna_telemetry().kv_pages_used(), 2)

  assert_eq(
    try! owner.invalidate_device_state_exact(handle),
    ExactDeviceNoChange,
  )
  assert_eq(owner.prefix_matched_tokens[slot], 0)
  assert_eq(owner.prefix_candidate_tokens[slot], 0)
  assert_eq(owner.prefill_cursors[slot], 0)
  assert_eq(owner.allocated_block_table_count(), 0)
  assert_eq(owner.luna_telemetry().prefix_entries(), 0)
  assert_eq(owner.luna_telemetry().prefix_pages(), 0)
  assert_eq(owner.luna_telemetry().kv_pages_used(), 0)

  let replacement_plan = wb_next_plan(owner)
  let row = try! replacement_plan.prefill_row(0)
  assert_eq(try! row.sequence_token_start(), 0)
  assert_eq(try! row.tokens().length(), 4)
  ignore(
    try! owner.complete(
      replacement_plan,
      wb_prefill_completion(owner, replacement_plan),
    ),
  )
  ignore(try! owner.cancel(handle))
  ignore(try! owner.take_terminal_notice())
  try! waiting_lease.release()
}

///|
test "terminal instance-loss drain clears cached pages with no live request" {
  let owner = wb_prefix_scheduler()
  let (seed, lease) = wb_prefix_request(
    983UL,
    [1, 2, 3, 4, 5],
    ReadWrite,
    b"scope",
  )
  ignore(try! owner.admit(seed, 0UL))
  wb_complete_one_token_request(owner, 7)
  ignore(try! owner.take_generated_token())
  ignore(try! owner.take_terminal_notice())
  try! lease.release()
  assert_eq(owner.luna_telemetry().prefix_entries(), 1)
  assert_eq(owner.luna_telemetry().kv_pages_used(), 2)
  assert_eq(try! owner.drain_instance_loss(), Complete)
  assert_eq(owner.luna_telemetry().prefix_entries(), 0)
  assert_eq(owner.luna_telemetry().prefix_pages(), 0)
  assert_eq(owner.luna_telemetry().kv_pages_used(), 0)
}

///|
test "device loss fails active work and preserves reset waiting request" {
  let owner = wb_prefix_scheduler()
  let (seed, seed_lease) = wb_prefix_request(
    984UL,
    [1, 2, 3, 4, 5],
    ReadWrite,
    b"scope",
  )
  ignore(try! owner.admit(seed, 0UL))
  wb_complete_one_token_request(owner, 7)
  ignore(try! owner.take_generated_token())
  ignore(try! owner.take_terminal_notice())
  try! seed_lease.release()

  let (active, active_lease) = wb_prefix_request(
    985UL,
    [8, 9, 10, 11, 12],
    ReadOnly,
    b"scope",
  )
  ignore(try! owner.admit(active, 0UL))
  let active_plan = wb_next_plan(owner)
  ignore(
    try! owner.complete(active_plan, wb_prefill_completion(owner, active_plan)),
  )
  let (waiting, waiting_lease) = wb_prefix_request(
    986UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
  )
  let waiting_handle = try! owner.admit(waiting, 0UL)
  let waiting_slot = waiting_handle.slot().index()
  assert_eq(owner.prefix_matched_tokens[waiting_slot], 0)
  assert_eq(owner.prefix_candidate_tokens[waiting_slot], 4)

  assert_eq(try! owner.invalidate_device_state(), 1)
  assert_eq((try! owner.take_terminal_notice()).reason(), WorkerFailed)
  try! active_lease.release()
  assert_eq(owner.prefix_matched_tokens[waiting_slot], 0)
  assert_eq(owner.prefix_candidate_tokens[waiting_slot], 0)
  assert_eq(owner.prefill_cursors[waiting_slot], 0)
  assert_eq(owner.luna_telemetry().prefix_entries(), 0)
  assert_eq(owner.luna_telemetry().kv_pages_used(), 0)
  ignore(try! owner.cancel(waiting_handle))
  ignore(try! owner.take_terminal_notice())
  try! waiting_lease.release()
}
