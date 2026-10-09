///|
/// Return a successful greedy token only when this single-row completion
/// exactly matches the immutable plan. This does not admit model/device
/// identity or allocate KV: those remain owned by the caller's loaded runtime.
pub fn ValidatedCompletionFrame::greedy_token_for(
  self : Self,
  plan : ValidatedPlanFrame,
) -> Int raise WorkerWireError {
  guard self.sequence_value() == plan.sequence_value() &&
    self.model_generation_value() == plan.model_generation_value() &&
    self.entry_count() == 1 else {
    raise InvalidFrame(CompletionEntry)
  }
  let entry = self.entry(0)
  let (id, generation, slot, processed, kind, sampling) = match
    (plan.prefill_count(), plan.decode_count()) {
    (1, 0) => {
      let row = plan.prefill_row(0)
      guard row.produces_token() else { raise InvalidFrame(PrefillRow) }
      (
        row.request_id_value(),
        row.request_generation_value(),
        row.completion_slot(),
        row.token_length(),
        @worker_protocol.CompletionEntryKind::FinalPrefillCompletion,
        row.sampling_mode_value(),
      )
    }
    (0, 1) => {
      let row = plan.decode_row(0)
      (
        row.request_id_value(),
        row.request_generation_value(),
        row.completion_slot(),
        0,
        @worker_protocol.CompletionEntryKind::DecodeCompletion,
        row.sampling_mode_value(),
      )
    }
    _ => raise InvalidFrame(Counts)
  }
  guard sampling == 0 else { raise InvalidFrame(Sampling) }
  guard entry.request_id_value() == id &&
    entry.request_generation_value() == generation &&
    entry.slot() == slot &&
    entry.kind() == kind &&
    entry.processed_tokens() == processed else {
    raise InvalidFrame(CompletionEntry)
  }
  let token = entry.token_id()
  let owner = plan.owner()
  guard token >= 0 &&
    token <= owner.worker_limits.max_token_id() &&
    token <= owner.inference_limits.max_token_id() else {
    raise InvalidFrame(CompletionEntry)
  }
  token
}

///|
/// Encode the next serial greedy step using the actual previous output token.
/// Preserve request/model identities, the caller's complete reserved page
/// table, capability order, completion slot and sampling seed. No synthetic
/// page/capability is constructed. Use two distinct owners until work retires.
pub fn PlanFrameBuffer::encode_greedy_decode_after(
  self : Self,
  previous : ValidatedPlanFrame,
  completion : ValidatedCompletionFrame,
) -> ValidatedPlanFrame raise WorkerWireError {
  let previous_owner = previous.owner()
  guard !physical_equal(self, previous_owner) &&
    self.worker_limits == previous_owner.worker_limits &&
    self.inference_limits == previous_owner.inference_limits else {
    raise InvalidFrame(Counts)
  }
  let token = completion.greedy_token_for(previous)
  let (id, generation, position, seed, sample, slot) = if previous.prefill_count() ==
    1 {
    let row = previous.prefill_row(0)
    (
      row.request_id_value(),
      row.request_generation_value(),
      row.sequence_token_start() + row.token_length(),
      row.sampling_seed_value(),
      row.sample_index(),
      row.completion_slot(),
    )
  } else {
    let row = previous.decode_row(0)
    (
      row.request_id_value(),
      row.request_generation_value(),
      row.sequence_token_position() + 1,
      row.sampling_seed_value(),
      row.sample_index(),
      row.completion_slot(),
    )
  }
  let sequence = previous.sequence_value()
  guard position < self.worker_limits.max_sequence_tokens() &&
    position < self.inference_limits.max_context_tokens() &&
    sample < self.worker_limits.max_sequence_tokens() - 1 &&
    sequence < 0xffffffffffffffffUL &&
    self.epoch < 0xffffffffffffffffUL else {
    raise InvalidFrame(DecodeRow)
  }
  let pages = previous.page_count()
  let capabilities = previous.capability_count()
  validate_plan_counts_for_encode(self, 1, 1, pages, capabilities, 0, 1)
  let length = exact_plan_frame_bytes(1, pages, capabilities, 0, 1)
  guard length <= self.capacity() else { raise InvalidLimit(PlanFrameBytes) }
  let bytes = self.storage
  let base = self.frame_offset
  write_plan_header(
    bytes,
    base,
    length,
    sequence + 1UL,
    previous.model_generation_value(),
    1,
    1,
    pages,
    capabilities,
    0,
    1,
  )
  write_i32(bytes, base + token_table_offset(), token)
  for index in 0..<pages {
    let offset = base + page_table_offset(1) + index * 8
    write_i32(bytes, offset, previous.page_index_at(index))
    write_i32(bytes, offset + 4, previous.page_generation_at(index))
  }
  for index in 0..<capabilities {
    write_i32(
      bytes,
      base + capability_table_offset(1, pages) + index * 4,
      previous.capability_value_at(index),
    )
  }
  let offset = base + decode_table_offset(1, pages, capabilities, 0)
  write_u64(bytes, offset, id)
  write_u64(bytes, offset + 8, generation)
  write_i32(bytes, offset + 16, position)
  write_i32(bytes, offset + 20, 0)
  write_i32(bytes, offset + 24, 0)
  write_i32(bytes, offset + 28, pages)
  write_i32(bytes, offset + 32, 0)
  write_i32(bytes, offset + 36, capabilities)
  write_greedy_sampling_parameters(bytes, offset + 40)
  write_u64(bytes, offset + 72, seed)
  write_i32(bytes, offset + 80, sample + 1)
  write_i32(bytes, offset + 84, slot)
  write_u32(
    bytes,
    base + plan_checksum_offset,
    frame_checksum_at(bytes, base, length),
  )
  self.epoch += 1UL
  self.length = length
  ValidatedPlanFrame(self, self.epoch)
}
