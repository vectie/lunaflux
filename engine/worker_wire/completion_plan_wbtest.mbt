///|
fn plan_validation_fixture() -> @worker_protocol.SubmittedSchedulePlan raise {
  let limits = completion_test_owner().worker_limits
  let owner = @worker_protocol.SchedulePlanBuffer::new(limits)
  let allocator = @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=2,
      max_run_pages=2,
      max_active_references_per_page=2,
      max_cached_references_per_page=2,
    ),
  )
  owner.append_final_prefill(
    request_id=@inference.RequestId::new(7UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[1, 2],
    pages=[allocator.allocate_one()],
    capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(7UL),
    sample_index=0,
    completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
  )
  owner.submit_wire(
    sequence_value=1UL,
    model_plan_generation_value=2UL,
    token_budget=2,
    has_previous=false,
    previous_sequence_value=0UL,
    loaded_generation_value=2UL,
  )
}

///|
fn load_plan_completion(
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> (CompletionFrameBuffer, FixedArray[Byte], ValidatedCompletionFrame) raise {
  let frame_owner = completion_test_owner()
  let completion_owner = @worker_protocol.CompletionBuffer::new(
    frame_owner.worker_limits,
  )
  let row = plan.prefill_row(0)
  completion_owner.append_final_prefill_completed(
    slot=row.completion_slot(),
    request_id=row.request_id(),
    request_generation=row.request_generation(),
    processed_tokens=row.tokens().length(),
    token_id=3,
  )
  let encoded = frame_owner.encode_completion(completion_owner.submit_for(plan))
  let bytes = FixedArray::make(encoded.length(), b'\x00')
  encoded.copy_to(bytes, destination_offset=0)
  (frame_owner, bytes, encoded)
}

///|
test "plan validation rejects checksum-valid foreign completion semantics" {
  let plan = plan_validation_fixture()
  let (owner, valid, prior) = load_plan_completion(plan)
  prior.validate_for(plan)
  let entry = completion_entry_offset(0)
  let attacks : Array[(Int, Int, UInt64, WorkerWireFrameRule, String)] = [
    (16, 8, 2UL, CompletionEntry, "sequence"),
    (24, 8, 3UL, CompletionEntry, "model generation"),
    (entry, 4, 1UL, CompletionSlot, "completion slot"),
    (entry + 8, 8, 8UL, CompletionEntry, "request id"),
    (entry + 16, 8, 1UL, CompletionEntry, "request generation"),
    (entry + 24, 4, 1UL, CompletionEntry, "processed count"),
  ]
  for attack in attacks {
    let (offset, width, value, expected_rule, label) = attack
    let bytes = copy_completion_fixture(valid)
    if width == 8 {
      write_u64(bytes, offset, value)
    } else {
      write_i32(bytes, offset, value.to_int())
    }
    refresh_completion_checksum(bytes)
    let hostile = owner.load(bytes, bytes.length())
    try hostile.validate_for(plan) catch {
      InvalidFrame(rule) if rule == expected_rule => ()
      _ => fail("unexpected " + label + " rejection")
    } noraise {
      _ => fail(label + " attack was accepted")
    }
    assert_eq(plan.sequence_wire(), 1UL)
    let restored = owner.load(valid, valid.length())
    restored.validate_for(plan)
    assert_eq(restored.entry_for_slot(0).request_id_value(), 7UL)
  }

  let wrong_kind = copy_completion_fixture(valid)
  write_u32(wrong_kind, entry + 4, 0)
  write_i32(wrong_kind, entry + 28, 0)
  refresh_completion_checksum(wrong_kind)
  let hostile_kind = owner.load(wrong_kind, wrong_kind.length())
  try hostile_kind.validate_for(plan) catch {
    InvalidFrame(CompletionEntry) => ()
    _ => fail("unexpected completion-kind rejection")
  } noraise {
    _ => fail("completion-kind attack was accepted")
  }
  let restored = owner.load(valid, valid.length())
  restored.validate_for(plan)
  assert_eq(restored.entry_for_slot(0).processed_tokens(), 2)
}
