///|
fn Scheduler::waiting_successor(self : Scheduler, slot : Int) -> Int {
  if slot < 0 ||
    slot >= self.waiting_next.length() ||
    self.waiting_link_generations[slot] != self.slot_generations[slot] ||
    self.slot_states[slot] != SLOT_WAITING {
    abort("scheduler waiting traversal generation invariant")
  }
  let next = self.waiting_next[slot]
  if next != NO_INDEX &&
    (
      self.waiting_link_generations[next] != self.slot_generations[next] ||
      self.waiting_prev[next] != slot
    ) {
    abort("scheduler waiting traversal link invariant")
  }
  next
}

///|
fn Scheduler::waiting_selection_precedes(
  self : Scheduler,
  left : Int,
  right : Int,
) -> Bool {
  let left_prefix = self.prefix_candidate_tokens[left]
  let right_prefix = self.prefix_candidate_tokens[right]
  left_prefix > right_prefix ||
  (
    left_prefix == right_prefix &&
    self.waiting_selection_ranks[left] < self.waiting_selection_ranks[right]
  )
}

///|
fn Scheduler::waiting_selection_sift_down(
  self : Scheduler,
  start : Int,
) -> Unit {
  let mut parent = start
  while parent < self.waiting_selection_count / 2 {
    let left = parent * 2 + 1
    let right = left + 1
    let mut best = left
    if right < self.waiting_selection_count &&
      self.waiting_selection_precedes(
        self.waiting_selection_heap[right],
        self.waiting_selection_heap[left],
      ) {
      best = right
    }
    if self.waiting_selection_precedes(
        self.waiting_selection_heap[parent],
        self.waiting_selection_heap[best],
      ) {
      break
    }
    self.waiting_selection_heap.swap(parent, best)
    parent = best
  }
}

///|
fn Scheduler::prepare_waiting_selection(self : Scheduler) -> Unit {
  self.waiting_selection_count = 0
  let mut current = self.waiting_head
  let mut rank = 0
  while current != NO_INDEX {
    self.waiting_selection_heap[self.waiting_selection_count] = current
    self.waiting_selection_ranks[current] = rank
    self.waiting_selection_count += 1
    self.selection_waiting_visits += 1
    rank += 1
    current = self.waiting_successor(current)
  }
  if self.waiting_selection_count != self.waiting_count {
    abort("scheduler waiting selection count invariant")
  }
  let mut parent = self.waiting_selection_count / 2 - 1
  while parent >= 0 {
    self.waiting_selection_sift_down(parent)
    parent -= 1
  }
}

///|
fn Scheduler::pop_waiting_selection(self : Scheduler) -> Int {
  if self.waiting_selection_count == 0 {
    return NO_INDEX
  }
  let selected = self.waiting_selection_heap[0]
  self.waiting_selection_count -= 1
  if self.waiting_selection_count > 0 {
    self.waiting_selection_heap[0] = self.waiting_selection_heap[self.waiting_selection_count]
    self.waiting_selection_sift_down(0)
  }
  self.waiting_selection_heap[self.waiting_selection_count] = NO_INDEX
  selected
}
