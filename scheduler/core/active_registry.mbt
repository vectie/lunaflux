///|
fn Scheduler::active_cursor_slot(self : Scheduler, cursor : Int) -> Int {
  if cursor >= 0 &&
    cursor < self.active_positions.length() &&
    self.active_positions[cursor] != NO_INDEX {
    cursor
  } else if self.active_count > 0 {
    self.active_slots[0]
  } else {
    NO_INDEX
  }
}

///|
fn Scheduler::active_cursor_position(self : Scheduler, cursor : Int) -> Int {
  let slot = self.active_cursor_slot(cursor)
  if slot == NO_INDEX {
    0
  } else {
    self.active_positions[slot]
  }
}

///|
fn Scheduler::add_active(self : Scheduler, slot : Int) -> Unit {
  if slot < 0 ||
    slot >= self.active_positions.length() ||
    self.active_positions[slot] != NO_INDEX ||
    self.active_count >= self.active_slots.length() {
    abort("scheduler active enqueue invariant")
  }
  self.active_slots[self.active_count] = slot
  self.active_positions[slot] = self.active_count
  self.active_count += 1
  if self.active_count == 1 {
    self.decode_cursor = slot
    self.pending_decode_cursor = slot
    self.preemption_cursor = slot
  }
}

///|
fn Scheduler::remove_active(self : Scheduler, slot : Int) -> Unit {
  if slot < 0 || slot >= self.active_positions.length() {
    abort("scheduler active remove range invariant")
  }
  let position = self.active_positions[slot]
  if position < 0 ||
    position >= self.active_count ||
    self.active_slots[position] != slot {
    abort("scheduler active ownership invariant")
  }
  let last_position = self.active_count - 1
  let replacement = self.active_slots[last_position]
  if position != last_position {
    self.active_slots[position] = replacement
    self.active_positions[replacement] = position
  }
  self.active_slots[last_position] = NO_INDEX
  self.active_positions[slot] = NO_INDEX
  self.active_count -= 1
  if self.active_count == 0 {
    self.decode_cursor = NO_INDEX
    self.pending_decode_cursor = NO_INDEX
    self.preemption_cursor = NO_INDEX
  } else {
    let successor = if position < self.active_count {
      self.active_slots[position]
    } else {
      self.active_slots[0]
    }
    if self.decode_cursor == slot {
      self.decode_cursor = successor
    }
    if self.pending_decode_cursor == slot {
      self.pending_decode_cursor = successor
    }
    if self.preemption_cursor == slot {
      self.preemption_cursor = successor
    }
  }
}
