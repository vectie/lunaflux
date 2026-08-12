///|
test "submitted completion resolves reversed entries by slot" {
  let plan_owner = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let plan = submitted_plan_for_completion(plan_owner)
  let owner = @worker_protocol.CompletionBuffer::new(protocol_limits())
  owner.append_decode_completed(
    slot=slot(1),
    request_id=request_id(8UL),
    request_generation=@inference.RequestGeneration::from_wire(3UL),
    token_id=42,
  )
  owner.append_prefill_completed(
    slot=slot(0),
    request_id=request_id(7UL),
    request_generation=@inference.RequestGeneration::initial(),
    processed_tokens=2,
  )
  let submitted = owner.submit_for(plan)
  assert_eq(submitted.entry(0).kind(), DecodeCompletion)
  assert_eq(submitted.entry_for_slot(slot(0)).kind(), PrefillCompletion)
  assert_eq(submitted.entry_for_slot(slot(0)).processed_tokens(), 2)
  assert_eq(submitted.entry_for_slot(slot(1)).kind(), DecodeCompletion)
  assert_eq(submitted.entry_for_slot(slot(1)).token_id(), 42)
  try submitted.entry_for_slot(slot(7)) catch {
    UnknownCompletionSlot(index=7) => ()
    _ => fail("expected unknown indexed completion slot")
  } noraise {
    _ => fail("expected absent slot lookup to fail")
  }
}

///|
test "slot index is cleared across abort and lifecycle reset" {
  let plan_owner = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let plan = submitted_plan_for_completion(plan_owner)
  let owner = @worker_protocol.CompletionBuffer::new(protocol_limits())
  let partial = owner.writer_for(plan)
  partial.append_prefill_completed(
    slot=slot(0),
    request_id=request_id(7UL),
    request_generation=@inference.RequestGeneration::initial(),
    processed_tokens=2,
  )
  partial.abort()
  let retry = owner.writer_for(plan)
  append_mixed_writer(retry)
  let submitted = retry.submit()
  assert_eq(submitted.entry_for_slot(slot(0)).processed_tokens(), 2)
  assert_eq(submitted.entry_for_slot(slot(1)).token_id(), 42)
  let consumed = owner.consume(submitted, plan)
  let retired = plan_owner.retire(plan)
  owner.reset(consumed)
  plan_owner.reset(retired)
  let next_plan = submitted_plan_for_completion(plan_owner)
  let next_writer = owner.writer_for(next_plan)
  append_mixed_writer(next_writer)
  let next_submitted = next_writer.submit()
  assert_eq(next_submitted.entry_for_slot(slot(0)).processed_tokens(), 2)
  assert_eq(next_submitted.entry_for_slot(slot(1)).token_id(), 42)
}
