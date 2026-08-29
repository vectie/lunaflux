///|
fn PrefixIndex::eviction_precedes(
  self : PrefixIndex,
  left : Int,
  right : Int,
) -> Bool {
  self.entry_priorities[left] < self.entry_priorities[right] ||
  (
    self.entry_priorities[left] == self.entry_priorities[right] &&
    (
      self.entry_recencies[left] < self.entry_recencies[right] ||
      (
        self.entry_recencies[left] == self.entry_recencies[right] &&
        left < right
      )
    )
  )
}

///|
fn PrefixIndex::eviction_heap_swap(
  self : PrefixIndex,
  left : Int,
  right : Int,
) -> Unit {
  let left_entry = self.eviction_heap[left]
  let right_entry = self.eviction_heap[right]
  self.eviction_heap[left] = right_entry
  self.eviction_heap[right] = left_entry
  self.eviction_heap_positions[left_entry] = right
  self.eviction_heap_positions[right_entry] = left
  self.eviction_heap_touches += 1
}

///|
fn PrefixIndex::eviction_heap_sift_up(self : PrefixIndex, start : Int) -> Unit {
  let mut position = start
  while position > 0 {
    let parent = (position - 1) / 2
    self.eviction_heap_touches += 1
    if self.eviction_precedes(
        self.eviction_heap[parent],
        self.eviction_heap[position],
      ) {
      break
    }
    self.eviction_heap_swap(parent, position)
    position = parent
  }
}

///|
fn PrefixIndex::eviction_heap_sift_down(
  self : PrefixIndex,
  start : Int,
) -> Unit {
  let mut position = start
  while position < self.eviction_heap_count / 2 {
    let left = position * 2 + 1
    let right = left + 1
    let mut best = left
    if right < self.eviction_heap_count {
      self.eviction_heap_touches += 1
      if self.eviction_precedes(
          self.eviction_heap[right],
          self.eviction_heap[left],
        ) {
        best = right
      }
    }
    self.eviction_heap_touches += 1
    if self.eviction_precedes(
        self.eviction_heap[position],
        self.eviction_heap[best],
      ) {
      break
    }
    self.eviction_heap_swap(position, best)
    position = best
  }
}

///|
fn PrefixIndex::eviction_heap_insert(self : PrefixIndex, entry : Int) -> Unit {
  if self.eviction_heap_positions[entry] >= 0 {
    abort("prefix duplicate eviction-heap entry")
  }
  let position = self.eviction_heap_count
  self.eviction_heap[position] = entry
  self.eviction_heap_positions[entry] = position
  self.eviction_heap_count += 1
  self.eviction_heap_touches += 1
  self.eviction_heap_sift_up(position)
}

///|
fn PrefixIndex::eviction_heap_remove(self : PrefixIndex, entry : Int) -> Unit {
  let position = self.eviction_heap_positions[entry]
  if position < 0 {
    abort("prefix absent eviction-heap entry")
  }
  self.eviction_heap_count -= 1
  let last = self.eviction_heap[self.eviction_heap_count]
  self.eviction_heap[self.eviction_heap_count] = -1
  self.eviction_heap_positions[entry] = -1
  self.eviction_heap_touches += 1
  if position < self.eviction_heap_count {
    self.eviction_heap[position] = last
    self.eviction_heap_positions[last] = position
    if position > 0 &&
      self.eviction_precedes(last, self.eviction_heap[(position - 1) / 2]) {
      self.eviction_heap_sift_up(position)
    } else {
      self.eviction_heap_sift_down(position)
    }
  }
}

///|
fn PrefixIndex::eviction_heap_refresh(self : PrefixIndex, entry : Int) -> Unit {
  let position = self.eviction_heap_positions[entry]
  if position < 0 {
    abort("prefix absent eviction-heap refresh")
  }
  if position > 0 &&
    self.eviction_precedes(entry, self.eviction_heap[(position - 1) / 2]) {
    self.eviction_heap_sift_up(position)
  } else {
    self.eviction_heap_sift_down(position)
  }
}

///|
fn PrefixIndex::eviction_candidate(self : PrefixIndex) -> Int {
  if self.eviction_heap_count == 0 {
    -1
  } else {
    self.eviction_heap[0]
  }
}

///|
fn PrefixIndex::eviction_heap_rebuild(self : PrefixIndex) -> Unit {
  let mut parent = self.eviction_heap_count / 2 - 1
  while parent >= 0 {
    self.eviction_heap_sift_down(parent)
    parent -= 1
  }
}
