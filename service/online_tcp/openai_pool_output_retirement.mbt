///|
/// Releases the single output cell in the same bounded transition that writes
/// its last byte. A short or blocked write retains its flight until retirement.
fn LunaOnlineOpenAIConnectionPool::release_completed_slot_output(
  self : LunaOnlineOpenAIConnectionPool,
  index : Int,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  let slot = self.slots[index]
  if slot.flights.length() != 1 ||
    slot.output_length <= 0 ||
    slot.output_cursor != slot.output_length {
    self.begin_slot_retirement(index)
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  slot.flights[0].release() catch {
    error => {
      self.fail_pool("release output flight", error)
      return LunaOnlineOpenAIConnectionPoolDrainingProgress
    }
  }
  slot.flights.clear()
  slot.output_cursor = 0
  slot.output_length = 0
  if slot.outbound_views.length() == 1 {
    // Resume a large JSON body only after its previous chunk was written.
    // The semantic terminal is ACKed only after its final chunk is copied.
    if !slot.begin_output_write(continuing=true) {
      self.begin_slot_retirement(index)
    } else {
      slot.phase = LUNA_OPENAI_POOL_RESPONSE_PROGRESS
    }
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  // Transport closure remains a separate transition after write confirmation.
  slot.phase = LUNA_OPENAI_POOL_REQUEST_ACTIVE
  LunaOnlineOpenAIConnectionPoolAdvanced
}
