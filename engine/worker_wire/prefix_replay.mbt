///|
/// Rebuild only the accepted input prefix after device-state rollback. The
/// replay has no sampling/publication side effect. Pages, capabilities, request
/// identity and position come from the actual verification, not a synthetic
/// model-specific recipe. Destination storage is prepared once and distinct.
pub fn PlanFrameBuffer::encode_prefix_replay(
  self : Self,
  verified : ValidatedPlanFrame,
  accepted_rows~ : Int,
  sequence_value~ : UInt64,
) -> ValidatedPlanFrame raise WorkerWireError {
  let source = verified.owner()
  guard !physical_equal(self, source) &&
    self.worker_limits == source.worker_limits &&
    self.inference_limits == source.inference_limits &&
    verified.prefill_count() == 1 &&
    verified.decode_count() == 0 &&
    accepted_rows > 0 &&
    accepted_rows <= verified.token_count() &&
    sequence_value > verified.sequence_value() else {
    raise InvalidFrame(PrefillRow)
  }
  let row = verified.prefill_row(0)
  guard row.token_start() == 0 &&
    row.token_length() == verified.token_count() &&
    row.sampling_mode_value() == 0 else {
    raise InvalidFrame(PrefillRow)
  }
  let id = row.request_id_value()
  let generation = row.request_generation_value()
  let position = row.sequence_token_start()
  let slot = row.completion_slot()
  let model = verified.model_generation_value()
  let pages = verified.page_count()
  let capabilities = verified.capability_count()
  validate_plan_counts_for_encode(
    self, accepted_rows, accepted_rows, pages, capabilities, 1, 0,
  )
  let length = exact_plan_frame_bytes(accepted_rows, pages, capabilities, 1, 0)
  guard length <= self.capacity() else { raise InvalidLimit(PlanFrameBytes) }
  guard self.epoch < 0xffffffffffffffffUL else { raise FrameEpochOverflow }
  let bytes = self.storage
  let base = self.frame_offset
  write_plan_header(
    bytes, base, length, sequence_value, model, accepted_rows, accepted_rows, pages,
    capabilities, 1, 0,
  )
  for i in 0..<accepted_rows {
    write_i32(bytes, base + token_table_offset() + i * 4, verified.token_at(i))
  }
  for i in 0..<pages {
    let offset = base + page_table_offset(accepted_rows) + i * 8
    write_i32(bytes, offset, verified.page_index_at(i))
    write_i32(bytes, offset + 4, verified.page_generation_at(i))
  }
  for i in 0..<capabilities {
    write_i32(
      bytes,
      base + capability_table_offset(accepted_rows, pages) + i * 4,
      verified.capability_value_at(i),
    )
  }
  let offset = base + prefill_table_offset(accepted_rows, pages, capabilities)
  write_u64(bytes, offset, id)
  write_u64(bytes, offset + 8, generation)
  write_i32(bytes, offset + 16, position)
  write_i32(bytes, offset + 20, 0)
  write_i32(bytes, offset + 24, accepted_rows)
  write_i32(bytes, offset + 28, 0)
  write_i32(bytes, offset + 32, pages)
  write_i32(bytes, offset + 36, 0)
  write_i32(bytes, offset + 40, capabilities)
  write_u32(bytes, offset + 44, 0)
  write_greedy_sampling_parameters(bytes, offset + 48)
  write_u64(bytes, offset + 80, 1UL)
  write_i32(bytes, offset + 88, 0)
  write_i32(bytes, offset + 92, slot)
  write_u32(bytes, base + plan_checksum_offset, 0)
  self.epoch += 1UL
  self.length = length
  ValidatedPlanFrame(self, self.epoch)
}
