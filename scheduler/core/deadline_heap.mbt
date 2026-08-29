///|
fn Scheduler::deadline_precedes(
  self : Scheduler,
  left : Int,
  right : Int,
) -> Bool {
  self.deadlines[left] < self.deadlines[right] ||
  (self.deadlines[left] == self.deadlines[right] && left < right)
}

///|
fn Scheduler::deadline_swap(self : Scheduler, left : Int, right : Int) -> Unit {
  if left == right {
    return
  }
  self.deadline_heap_slots.swap(left, right)
  self.deadline_heap_generations.swap(left, right)
  self.deadline_heap_positions[self.deadline_heap_slots[left]] = left
  self.deadline_heap_positions[self.deadline_heap_slots[right]] = right
}

///|
fn Scheduler::deadline_sift_up(self : Scheduler, start : Int) -> Unit {
  let mut child = start
  while child > 0 {
    let parent = (child - 1) / 2
    if !self.deadline_precedes(
        self.deadline_heap_slots[child],
        self.deadline_heap_slots[parent],
      ) {
      break
    }
    self.deadline_swap(child, parent)
    child = parent
  }
}

///|
fn Scheduler::deadline_sift_down(self : Scheduler, start : Int) -> Unit {
  let mut parent = start
  while parent < self.deadline_heap_count / 2 {
    let left = parent * 2 + 1
    let right = left + 1
    let mut best = left
    if right < self.deadline_heap_count &&
      self.deadline_precedes(
        self.deadline_heap_slots[right],
        self.deadline_heap_slots[left],
      ) {
      best = right
    }
    if self.deadline_precedes(
        self.deadline_heap_slots[parent],
        self.deadline_heap_slots[best],
      ) {
      break
    }
    self.deadline_swap(parent, best)
    parent = best
  }
}

///|
fn Scheduler::deadline_insert(self : Scheduler, slot : Int) -> Unit {
  if slot < 0 ||
    slot >= self.deadline_heap_positions.length() ||
    self.deadline_heap_positions[slot] != NO_INDEX ||
    self.deadline_heap_count >= self.deadline_heap_slots.length() {
    abort("scheduler deadline insert invariant")
  }
  let position = self.deadline_heap_count
  self.deadline_heap_count += 1
  self.deadline_heap_slots[position] = slot
  self.deadline_heap_generations[position] = self.slot_generations[slot]
  self.deadline_heap_positions[slot] = position
  self.deadline_sift_up(position)
}

///|
fn Scheduler::deadline_remove(self : Scheduler, slot : Int) -> Unit {
  if slot < 0 || slot >= self.deadline_heap_positions.length() {
    abort("scheduler deadline remove range invariant")
  }
  let position = self.deadline_heap_positions[slot]
  if position == NO_INDEX {
    return
  }
  if position >= self.deadline_heap_count ||
    self.deadline_heap_slots[position] != slot ||
    self.deadline_heap_generations[position] != self.slot_generations[slot] {
    abort("scheduler deadline remove generation invariant")
  }
  let last = self.deadline_heap_count - 1
  self.deadline_swap(position, last)
  self.deadline_heap_count -= 1
  self.deadline_heap_slots[last] = NO_INDEX
  self.deadline_heap_generations[last] = 0
  self.deadline_heap_positions[slot] = NO_INDEX
  if position < self.deadline_heap_count {
    if position > 0 &&
      self.deadline_precedes(
        self.deadline_heap_slots[position],
        self.deadline_heap_slots[(position - 1) / 2],
      ) {
      self.deadline_sift_up(position)
    } else {
      self.deadline_sift_down(position)
    }
  }
}

///|
fn Scheduler::deadline_root_due(self : Scheduler, now_millis : UInt64) -> Bool {
  if self.deadline_heap_count == 0 {
    return false
  }
  let slot = self.deadline_heap_slots[0]
  if self.deadline_heap_generations[0] != self.slot_generations[slot] ||
    self.deadline_heap_positions[slot] != 0 {
    abort("scheduler deadline root generation invariant")
  }
  self.deadlines[slot] <= now_millis
}

///|
fn Scheduler::deadline_due_sift_down(
  self : Scheduler,
  start : Int,
  count : Int,
) -> Unit {
  let mut parent = start
  while parent < count / 2 {
    let left = parent * 2 + 1
    let right = left + 1
    let best = if right < count &&
      self.deadline_due_heap[right] < self.deadline_due_heap[left] {
      right
    } else {
      left
    }
    if self.deadline_due_heap[parent] <= self.deadline_due_heap[best] {
      break
    }
    self.deadline_due_heap.swap(parent, best)
    parent = best
  }
}

///|
fn Scheduler::prepare_deadline_due_heap(
  self : Scheduler,
  now_millis : UInt64,
) -> Int {
  self.deadline_live_visits = 0
  self.deadline_due_pops = 0
  let mut due_count = 0
  for position in 0..<self.deadline_heap_count {
    let slot = self.deadline_heap_slots[position]
    if self.deadline_heap_generations[position] != self.slot_generations[slot] ||
      self.deadline_heap_positions[slot] != position {
      abort("scheduler deadline traversal generation invariant")
    }
    self.deadline_live_visits += 1
    if self.deadlines[slot] <= now_millis {
      self.deadline_due_heap[due_count] = slot
      due_count += 1
    }
  }
  let mut parent = due_count / 2 - 1
  while parent >= 0 {
    self.deadline_due_sift_down(parent, due_count)
    parent -= 1
  }
  if self.deadline_live_visits != self.deadline_heap_count {
    abort("scheduler deadline traversal complexity invariant")
  }
  due_count
}

///|
fn Scheduler::pop_deadline_due(self : Scheduler, count : Int) -> Int {
  if count <= 0 {
    abort("scheduler deadline due pop invariant")
  }
  let slot = self.deadline_due_heap[0]
  let remaining = count - 1
  if remaining > 0 {
    self.deadline_due_heap[0] = self.deadline_due_heap[remaining]
    self.deadline_due_sift_down(0, remaining)
  }
  self.deadline_due_heap[remaining] = NO_INDEX
  self.deadline_due_pops += 1
  if self.deadline_due_pops > self.deadline_due_heap.length() {
    abort("scheduler deadline due-pop complexity invariant")
  }
  slot
}
