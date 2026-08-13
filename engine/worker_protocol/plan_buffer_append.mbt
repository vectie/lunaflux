///|
/// Appends one intermediate prefill row and copies its flattened table
/// segments into preallocated storage. All validation precedes mutation.
pub fn SchedulePlanBuffer::append_prefill(
  self : SchedulePlanBuffer,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  sequence_token_start~ : Int,
  tokens~ : ArrayView[Int],
  pages~ : ArrayView[@page_allocator.PageId],
  capabilities~ : ArrayView[@plan.KernelCapabilityId],
  completion_slot~ : CompletionSlot,
) -> Unit raise WorkerProtocolError {
  self.append_prefill_row(
    request_id,
    request_generation,
    sequence_token_start,
    tokens,
    pages,
    capabilities,
    false,
    @inference.SamplingParameters::greedy(),
    1UL,
    0,
    completion_slot,
  )
}

///|
/// Appends the final prompt chunk, which samples exactly one output token.
pub fn SchedulePlanBuffer::append_final_prefill(
  self : SchedulePlanBuffer,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  sequence_token_start~ : Int,
  tokens~ : ArrayView[Int],
  pages~ : ArrayView[@page_allocator.PageId],
  capabilities~ : ArrayView[@plan.KernelCapabilityId],
  sampling~ : @inference.SamplingParameters,
  sampling_seed~ : @inference.SamplingSeed,
  sample_index~ : Int,
  completion_slot~ : CompletionSlot,
) -> Unit raise WorkerProtocolError {
  self.append_prefill_row(
    request_id,
    request_generation,
    sequence_token_start,
    tokens,
    pages,
    capabilities,
    true,
    sampling,
    sampling_seed.value(),
    sample_index,
    completion_slot,
  )
}

///|
fn SchedulePlanBuffer::append_prefill_row(
  self : SchedulePlanBuffer,
  request_id : @inference.RequestId,
  request_generation : @inference.RequestGeneration,
  sequence_token_start : Int,
  tokens : ArrayView[Int],
  pages : ArrayView[@page_allocator.PageId],
  capabilities : ArrayView[@plan.KernelCapabilityId],
  produces_token : Bool,
  sampling : @inference.SamplingParameters,
  sampling_seed_value : UInt64,
  sample_index : Int,
  completion_slot : CompletionSlot,
) -> Unit raise WorkerProtocolError {
  self.preflight_row(
    Prefill,
    request_id,
    completion_slot,
    tokens.length(),
    pages.length(),
    capabilities.length(),
  )
  if sequence_token_start < 0 ||
    sequence_token_start > self.limits.max_sequence_tokens() ||
    tokens.length() > self.limits.max_sequence_tokens() - sequence_token_start {
    raise InvalidRowDescriptor(kind=Prefill, rule=InvalidSequencePosition)
  }
  if produces_token &&
    (
      sampling_seed_value == 0UL ||
      sample_index < 0 ||
      sample_index >= self.limits.max_sequence_tokens()
    ) {
    raise InvalidRowDescriptor(kind=Prefill, rule=InvalidSampleIndex)
  }
  for index, token in tokens {
    validate_token(token, self.token_count + index, self.limits)
  }
  let row = self.prefill_count
  let token_start = self.token_count
  let page_start = self.page_count
  let capability_start = self.capability_count
  self.copy_tables(tokens, pages, capabilities)
  self.prefill_request_ids[row] = request_id
  self.prefill_request_generations[row] = request_generation
  self.prefill_sequence_starts[row] = sequence_token_start
  self.prefill_token_starts[row] = token_start
  self.prefill_token_lengths[row] = tokens.length()
  self.prefill_page_starts[row] = page_start
  self.prefill_page_lengths[row] = pages.length()
  self.prefill_capability_starts[row] = capability_start
  self.prefill_capability_lengths[row] = capabilities.length()
  self.prefill_produces_tokens[row] = produces_token
  self.prefill_sampling[row] = sampling
  self.prefill_sampling_seed_values[row] = sampling_seed_value
  self.prefill_sample_indices[row] = sample_index
  self.prefill_completion_slots[row] = completion_slot
  self.token_count = token_start + tokens.length()
  self.page_count = page_start + pages.length()
  self.capability_count = capability_start + capabilities.length()
  self.prefill_count = row + 1
}

///|
/// Appends one exact decode row. The single input token and all table segments
/// are copied only after every bound and duplicate check succeeds.
pub fn SchedulePlanBuffer::append_decode(
  self : SchedulePlanBuffer,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  sequence_token_position~ : Int,
  input_token~ : Int,
  pages~ : ArrayView[@page_allocator.PageId],
  capabilities~ : ArrayView[@plan.KernelCapabilityId],
  sampling~ : @inference.SamplingParameters,
  sampling_seed~ : @inference.SamplingSeed,
  sample_index~ : Int,
  completion_slot~ : CompletionSlot,
) -> Unit raise WorkerProtocolError {
  self.preflight_row(
    Decode,
    request_id,
    completion_slot,
    1,
    pages.length(),
    capabilities.length(),
  )
  if sequence_token_position < 0 ||
    sequence_token_position >= self.limits.max_sequence_tokens() {
    raise InvalidRowDescriptor(kind=Decode, rule=InvalidSequencePosition)
  }
  if sample_index < 0 || sample_index >= self.limits.max_sequence_tokens() {
    raise InvalidRowDescriptor(kind=Decode, rule=InvalidSampleIndex)
  }
  validate_token(input_token, self.token_count, self.limits)
  let row = self.decode_count
  let token_start = self.token_count
  let page_start = self.page_count
  let capability_start = self.capability_count
  self.tokens[token_start] = input_token
  for index, page in pages {
    map_page_storage_write(self.pages, page_start + index, page)
  }
  for index, capability in capabilities {
    self.capabilities[capability_start + index] = capability
  }
  self.decode_request_ids[row] = request_id
  self.decode_request_generations[row] = request_generation
  self.decode_sequence_positions[row] = sequence_token_position
  self.decode_token_starts[row] = token_start
  self.decode_page_starts[row] = page_start
  self.decode_page_lengths[row] = pages.length()
  self.decode_capability_starts[row] = capability_start
  self.decode_capability_lengths[row] = capabilities.length()
  self.decode_sampling[row] = sampling
  self.decode_sampling_seed_values[row] = sampling_seed.value()
  self.decode_sample_indices[row] = sample_index
  self.decode_completion_slots[row] = completion_slot
  self.token_count = token_start + 1
  self.page_count = page_start + pages.length()
  self.capability_count = capability_start + capabilities.length()
  self.decode_count = row + 1
}
