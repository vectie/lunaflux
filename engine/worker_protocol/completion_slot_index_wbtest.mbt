///|
test "completion slot index clears every touched mapping" {
  let limits = try! WorkerProtocolLimits::new(
    max_prefill_rows=2,
    max_decode_rows=2,
    max_plan_rows=2,
    max_plan_tokens=4,
    max_plan_pages=4,
    max_capabilities=4,
    max_completion_slots=4,
    max_sequence_tokens=8,
    max_token_id=31,
  )
  let owner = try! CompletionBuffer::new(limits)
  let first = try! CompletionSlot::new(3, limits)
  try! owner.append_failed(
    slot=first,
    request_id=try! @inference.RequestId::new(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    failure=ExecutionFailed,
  )
  assert_eq(owner.slot_to_entry[3], 0)
  owner.clear_slot_index()
  assert_eq(owner.slot_to_entry[3], -1)
  assert_eq(owner.slot_to_entry[0], -1)
  assert_eq(owner.slot_to_entry[1], -1)
  assert_eq(owner.slot_to_entry[2], -1)
}
