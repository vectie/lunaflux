///|
fn Scheduler::enqueue_waiting(self : Scheduler, index : Int) -> Unit {
  if self.waiting_link_generations[index] != 0 ||
    self.waiting_prev[index] != NO_INDEX ||
    self.waiting_next[index] != NO_INDEX {
    abort("scheduler waiting enqueue invariant")
  }
  self.waiting_prev[index] = self.waiting_tail
  self.waiting_next[index] = NO_INDEX
  self.waiting_link_generations[index] = self.slot_generations[index]
  if self.waiting_tail == NO_INDEX {
    self.waiting_head = index
    self.waiting_tail = index
  } else {
    if self.waiting_link_generations[self.waiting_tail] !=
      self.slot_generations[self.waiting_tail] {
      abort("scheduler waiting tail generation invariant")
    }
    self.waiting_next[self.waiting_tail] = index
    self.waiting_tail = index
  }
  self.waiting_count += 1
}

///|
fn Scheduler::remove_waiting(self : Scheduler, index : Int) -> Unit {
  self.waiting_unlink_link_touches = 1
  if index < 0 ||
    index >= self.waiting_next.length() ||
    self.waiting_link_generations[index] != self.slot_generations[index] ||
    self.slot_states[index] != SLOT_WAITING {
    abort("scheduler waiting unlink generation invariant")
  }
  let previous = self.waiting_prev[index]
  let next = self.waiting_next[index]
  if previous == NO_INDEX {
    if self.waiting_head != index {
      abort("scheduler waiting head invariant")
    }
    self.waiting_head = next
  } else {
    self.waiting_unlink_link_touches += 1
    if self.waiting_link_generations[previous] !=
      self.slot_generations[previous] ||
      self.waiting_next[previous] != index {
      abort("scheduler waiting previous invariant")
    }
    self.waiting_next[previous] = next
  }
  if next == NO_INDEX {
    if self.waiting_tail != index {
      abort("scheduler waiting tail invariant")
    }
    self.waiting_tail = previous
  } else {
    self.waiting_unlink_link_touches += 1
    if self.waiting_link_generations[next] != self.slot_generations[next] ||
      self.waiting_prev[next] != index {
      abort("scheduler waiting next invariant")
    }
    self.waiting_prev[next] = previous
  }
  self.waiting_prev[index] = NO_INDEX
  self.waiting_next[index] = NO_INDEX
  self.waiting_link_generations[index] = 0
  self.waiting_count -= 1
  if self.waiting_unlink_link_touches < 1 ||
    self.waiting_unlink_link_touches > 3 {
    abort("scheduler waiting unlink complexity invariant")
  }
}
