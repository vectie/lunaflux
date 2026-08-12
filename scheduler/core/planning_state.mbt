///|
fn Scheduler::rollback_tables(
  self : Scheduler,
  checkpoint : @block_table.BlockTableAllocationCheckpoint,
) -> Unit {
  for offset in 0..<self.journal_count {
    let index = self.journal_count - offset - 1
    let table = try! self.journal_tables.read(index)
    if self.journal_appended[index] {
      try! self.block_tables.detach_suffix_for_checkpoint(
        checkpoint,
        table,
        self.journal_old_page_counts[index],
      )
    }
    try! self.journal_tables.clear(index)
  }
  self.journal_count = 0
}

///|
fn Scheduler::commit_plan_state(self : Scheduler, side : Int) -> Unit {
  let slots = if side == 0 { self.plan_a_slots } else { self.plan_b_slots }
  for row in 0..<self.prefill_count {
    let slot = self.prefill_slots[row]
    slots[row] = slot
    if self.slot_states[slot] == SLOT_WAITING {
      self.remove_waiting(slot)
      self.active_slots[self.active_count] = slot
      self.active_count += 1
    }
    self.slot_states[slot] = SLOT_PREFILL
    self.in_flight[slot] = true
  }
  for row in 0..<self.decode_count {
    let slot = self.decode_slots[row]
    slots[self.prefill_count + row] = slot
    self.slot_states[slot] = SLOT_DECODE
    self.in_flight[slot] = true
  }
  for index in 0..<self.journal_count {
    let slot = self.journal_slots[index]
    if !(try! self.request_tables.is_occupied(slot)) {
      try! self.request_tables.write(slot, try! self.journal_tables.read(index))
    }
    try! self.journal_tables.clear(index)
  }
  self.journal_count = 0
}
