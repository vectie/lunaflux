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
  let (owner, bytes, _) = load_plan_completion(plan)
  let entry = completion_entry_offset(0)
  write_u64(bytes, entry + 8, 8UL)
  refresh_completion_checksum(bytes)
  let foreign_request = owner.load(bytes, bytes.length())
  try foreign_request.validate_for(plan) catch {
    InvalidFrame(CompletionEntry) => ()
    _ => fail("expected request identity rejection")
  } noraise {
    _ => fail("foreign request accepted")
  }

  write_u64(bytes, entry + 8, 7UL)
  write_u32(bytes, entry + 4, 0)
  write_i32(bytes, entry + 28, 0)
  refresh_completion_checksum(bytes)
  let wrong_kind = owner.load(bytes, bytes.length())
  try wrong_kind.validate_for(plan) catch {
    InvalidFrame(CompletionEntry) => ()
    _ => fail("expected completion-kind rejection")
  } noraise {
    _ => fail("wrong completion kind accepted")
  }

  write_u32(bytes, entry + 4, 1)
  write_i32(bytes, entry + 24, 1)
  write_i32(bytes, entry + 28, 3)
  refresh_completion_checksum(bytes)
  let wrong_count = owner.load(bytes, bytes.length())
  try wrong_count.validate_for(plan) catch {
    InvalidFrame(CompletionEntry) => ()
    _ => fail("expected processed-token rejection")
  } noraise {
    _ => fail("wrong processed-token count accepted")
  }
}
