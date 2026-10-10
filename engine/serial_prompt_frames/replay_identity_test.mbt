///|
fn identity_frame(
  start : Int,
  sequence : UInt64,
  request : UInt64,
  final : Bool,
) -> Bytes raise {
  let (limits, inference) = fixture_limits()
  let pages = @pages.PageAllocator::new(
    @pages.PageAllocatorLimits::new(
      page_capacity=1,
      max_run_pages=1,
      max_active_references_per_page=1,
      max_cached_references_per_page=1,
    ),
  )
  let page = pages.allocate_one()
  defer pages.release_active(page)
  let owner = @protocol.SchedulePlanBuffer::new(limits)
  let id = @inference.RequestId::new(request)
  let generation = @inference.RequestGeneration::initial()
  let slot = @protocol.CompletionSlot::new(0, limits)
  let capabilities = [@plan.KernelCapabilityId::paged_causal_attention()]
  if final {
    owner.append_final_prefill(
      request_id=id,
      request_generation=generation,
      sequence_token_start=start,
      tokens=[64],
      pages=[page],
      capabilities~,
      sampling=@inference.SamplingParameters::greedy(),
      sampling_seed=@inference.SamplingSeed::new(3UL),
      sample_index=start,
      completion_slot=slot,
    )
  } else {
    owner.append_prefill(
      request_id=id,
      request_generation=generation,
      sequence_token_start=start,
      tokens=[64],
      pages=[page],
      capabilities~,
      completion_slot=slot,
    )
  }
  let submitted = owner.submit_wire(
    sequence_value=sequence,
    model_plan_generation_value=1UL,
    token_budget=1,
    has_previous=sequence > 1UL,
    previous_sequence_value=sequence - 1UL,
    loaded_generation_value=1UL,
  )
  let wire = @wire.PlanFrameBuffer::new(limits, inference)
  let encoded = wire.encode_plan(submitted)
  let scratch = FixedArray::make(wire.capacity(), b'\x00')
  encoded.copy_to(scratch, destination_offset=0)
  let bytes = Bytes::from_array(scratch[:encoded.length()].to_owned())
  ignore(owner.retire(submitted))
  bytes
}

///|
test "prepared replay rejects individually valid chunks with gaps foreign request or early sampling" {
  let (limits, inference) = fixture_limits()
  let first = identity_frame(0, 1UL, 1UL, false)
  let last = identity_frame(1, 2UL, 1UL, true)
  let valid = @serial_prompt_frames.PreparedSerialPrompt::new(
    [first, last],
    limits,
    inference,
    maximum_new_tokens=14,
    maximum_context=16,
    maximum_encoded_bytes=1048576L,
    model_generation=1UL,
  )
  assert_eq(valid.input_token_count(), 2)
  for
    frames in [
      [first, identity_frame(2, 2UL, 1UL, true)],
      [first, identity_frame(1, 2UL, 2UL, true)],
      [identity_frame(0, 1UL, 1UL, true), last],
      [identity_frame(1, 1UL, 1UL, false), last],
    ] {
    try
      @serial_prompt_frames.PreparedSerialPrompt::new(
        frames,
        limits,
        inference,
        maximum_new_tokens=1,
        maximum_context=16,
        maximum_encoded_bytes=1048576L,
        model_generation=1UL,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("inconsistent contiguous prompt replay accepted")
    }
  }
}
