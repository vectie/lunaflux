///|
fn PrefixIndex::recency_normalization_precedes(
  self : PrefixIndex,
  left : Int,
  right : Int,
) -> Bool {
  self.recency_normalization_comparisons += 1
  self.entry_recencies[left] < self.entry_recencies[right] ||
  (self.entry_recencies[left] == self.entry_recencies[right] && left < right)
}

///|
fn PrefixIndex::recency_normalization_swap(
  self : PrefixIndex,
  left : Int,
  right : Int,
) -> Unit {
  let entry = self.recency_entry_scratch[left]
  self.recency_entry_scratch[left] = self.recency_entry_scratch[right]
  self.recency_entry_scratch[right] = entry
  self.recency_normalization_swaps += 1
}

///|
fn PrefixIndex::recency_normalization_sift_down(
  self : PrefixIndex,
  start : Int,
  length : Int,
) -> Unit {
  let mut position = start
  while position < length / 2 {
    let left = position * 2 + 1
    let right = left + 1
    let mut greatest = left
    if right < length &&
      self.recency_normalization_precedes(
        self.recency_entry_scratch[left],
        self.recency_entry_scratch[right],
      ) {
      greatest = right
    }
    if !self.recency_normalization_precedes(
        self.recency_entry_scratch[position],
        self.recency_entry_scratch[greatest],
      ) {
      break
    }
    self.recency_normalization_swap(position, greatest)
    position = greatest
  }
}

///|
fn PrefixIndex::normalize_recencies(self : PrefixIndex) -> Unit {
  self.recency_normalization_comparisons = 0
  self.recency_normalization_swaps = 0
  let mut length = 0
  for entry in 0..<self.entry_states.length() {
    if self.entry_states[entry] == ENTRY_USED {
      self.recency_entry_scratch[length] = entry
      length += 1
    }
  }
  let mut parent = length / 2 - 1
  while parent >= 0 {
    self.recency_normalization_sift_down(parent, length)
    parent -= 1
  }
  let mut end = length - 1
  while end > 0 {
    self.recency_normalization_swap(0, end)
    self.recency_normalization_sift_down(0, end)
    end -= 1
  }
  for rank in 0..<length {
    let entry = self.recency_entry_scratch[rank]
    self.entry_recencies[entry] = (rank + 1).to_uint64()
    self.recency_entry_scratch[rank] = -1
  }
  self.recency_clock = length.to_uint64()
  self.eviction_heap_rebuild()
}
