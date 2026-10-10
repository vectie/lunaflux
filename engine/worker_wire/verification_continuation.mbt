///|
/// Append a draft block after the committed input prefix of the previous plan.
/// The prefix may be shorter than the previous tentative submission. Preserve
/// its actual request/page/capability bindings; never append after rejected KV.
/// Row zero is the already-emitted seed. The caller owns target verification
/// and physical commit/rollback before using this continuation.
pub fn PlanFrameBuffer::encode_verification_after(
  self : Self,
  previous : ValidatedPlanFrame,
  committed_input_rows~ : Int,
  draft_inputs~ : ArrayView[Int],
  sequence_value~ : UInt64,
) -> ValidatedPlanFrame raise WorkerWireError {
  let source = previous.owner()
  guard !physical_equal(self, source) &&
    self.worker_limits == source.worker_limits &&
    self.inference_limits == source.inference_limits &&
    previous.prefill_count() == 1 &&
    previous.decode_count() == 0 &&
    committed_input_rows > 0 &&
    committed_input_rows <= previous.token_count() &&
    !draft_inputs.is_empty() &&
    sequence_value > previous.sequence_value() else {
    raise InvalidFrame(PrefillRow)
  }
  let row = previous.prefill_row(0)
  guard row.token_start() == 0 &&
    row.token_length() == previous.token_count() &&
    row.produces_token() &&
    row.sampling_mode_value() == 0 else {
    raise InvalidFrame(PrefillRow)
  }
  let position = row.sequence_token_start() + committed_input_rows
  let count = draft_inputs.length()
  guard position <= self.worker_limits.max_sequence_tokens() &&
    position <= self.inference_limits.max_context_tokens() &&
    count <= self.worker_limits.max_sequence_tokens() - position &&
    count <= self.inference_limits.max_context_tokens() - position else {
    raise InvalidFrame(PrefillRow)
  }
  for token in draft_inputs {
    guard token >= 0 &&
      token <= self.worker_limits.max_token_id() &&
      token <= self.inference_limits.max_token_id() else {
      raise InvalidFrame(TokenTable)
    }
  }
  let pages = previous.page_count()
  let capabilities = previous.capability_count()
  validate_plan_counts_for_encode(self, count, count, pages, capabilities, 1, 0)
  let length = exact_plan_frame_bytes(count, pages, capabilities, 1, 0)
  guard length <= self.capacity() else { raise InvalidLimit(PlanFrameBytes) }
  guard self.epoch < 0xffffffffffffffffUL else { raise FrameEpochOverflow }
  let bytes = self.storage
  let base = self.frame_offset
  write_plan_header(
    bytes,
    base,
    length,
    sequence_value,
    previous.model_generation_value(),
    count,
    count,
    pages,
    capabilities,
    1,
    0,
  )
  for i, token in draft_inputs {
    write_i32(bytes, base + token_table_offset() + i * 4, token)
  }
  for i in 0..<pages {
    let offset = base + page_table_offset(count) + i * 8
    write_i32(bytes, offset, previous.page_index_at(i))
    write_i32(bytes, offset + 4, previous.page_generation_at(i))
  }
  for i in 0..<capabilities {
    write_i32(
      bytes,
      base + capability_table_offset(count, pages) + i * 4,
      previous.capability_value_at(i),
    )
  }
  let offset = base + prefill_table_offset(count, pages, capabilities)
  write_u64(bytes, offset, row.request_id_value())
  write_u64(bytes, offset + 8, row.request_generation_value())
  write_i32(bytes, offset + 16, position)
  write_i32(bytes, offset + 20, 0)
  write_i32(bytes, offset + 24, count)
  write_i32(bytes, offset + 28, 0)
  write_i32(bytes, offset + 32, pages)
  write_i32(bytes, offset + 36, 0)
  write_i32(bytes, offset + 40, capabilities)
  write_u32(bytes, offset + 44, 1)
  write_greedy_sampling_parameters(bytes, offset + 48)
  write_u64(bytes, offset + 80, row.sampling_seed_value())
  write_i32(bytes, offset + 88, 0)
  write_i32(bytes, offset + 92, row.completion_slot())
  write_u32(bytes, base + plan_checksum_offset, 0)
  self.epoch += 1UL
  self.length = length
  ValidatedPlanFrame(self, self.epoch)
}
