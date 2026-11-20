///|
fn LunaTokenizerWorker::start_compaction(self : LunaTokenizerWorker) -> Unit {
  self.compact_node = self.head
  self.compact_actual = 0
  self.output_count = 0
  self.phase = LUNA_WORKER_COMPACT
}

///|
fn LunaTokenizerWorker::progress_select_adjacency(
  self : LunaTokenizerWorker,
) -> Unit {
  if self.scan_left < 0 || self.next[self.scan_left] < 0 {
    if self.selected_rank < 0 {
      self.start_compaction()
    } else {
      self.scan_left = self.head
      self.phase = LUNA_WORKER_MERGE_BATCH
    }
    return
  }
  let left_token = self.tokens[self.scan_left]
  if left_token < 0 || left_token >= self.spec.vocab_size {
    self.scan_left = self.next[self.scan_left]
    return
  }
  self.lookup_low = self.spec.merge_left_offsets[left_token]
  self.lookup_high = self.spec.merge_left_offsets[left_token + 1]
  self.phase = LUNA_WORKER_SELECT_RULE
}

///|
fn LunaTokenizerWorker::progress_select_rule(
  self : LunaTokenizerWorker,
) -> Unit {
  let right = self.next[self.scan_left]
  if self.lookup_low >= self.lookup_high {
    self.scan_left = right
    self.phase = LUNA_WORKER_SELECT_ADJACENCY
    return
  }
  let middle = self.lookup_low + (self.lookup_high - self.lookup_low) / 2
  let candidate = self.spec.merge_right_sorted[middle]
  let right_token = self.tokens[right]
  if candidate < right_token {
    self.lookup_low = middle + 1
  } else if candidate > right_token {
    self.lookup_high = middle
  } else {
    let rank = self.spec.merge_rank_sorted[middle]
    if self.selected_rank < 0 || rank < self.selected_rank {
      self.selected_rank = rank
    }
    self.scan_left = right
    self.phase = LUNA_WORKER_SELECT_ADJACENCY
  }
}

///|
fn LunaTokenizerWorker::progress_merge_batch(
  self : LunaTokenizerWorker,
) -> Unit {
  if self.scan_left < 0 || self.next[self.scan_left] < 0 {
    self.scan_left = self.head
    self.selected_rank = -1
    self.phase = LUNA_WORKER_SELECT_ADJACENCY
    return
  }
  let left = self.scan_left
  let right = self.next[left]
  if self.tokens[left] == self.spec.merge_left_ids[self.selected_rank] &&
    self.tokens[right] == self.spec.merge_right_ids[self.selected_rank] {
    let after = self.next[right]
    self.tokens[left] = self.spec.merge_result_ids[self.selected_rank]
    self.next[left] = after
    if after >= 0 {
      self.previous[after] = left
    } else {
      self.tail = left
    }
    self.previous[right] = LUNA_REMOVED_NODE
    self.next[right] = LUNA_REMOVED_NODE
    self.scan_left = after
  } else {
    self.scan_left = right
  }
}

///|
fn LunaTokenizerWorker::finish_compaction(
  self : LunaTokenizerWorker,
) -> Unit raise TokenizerError {
  if self.truncation is RejectOverflow &&
    self.compact_actual > self.output_limit {
    let actual = self.compact_actual
    let maximum = self.output_limit
    self.clear_request()
    raise OutputTooLong(actual~, maximum~)
  }
  self.output_count = Int32::min(self.compact_actual, self.output_limit)
  self.truncated = self.compact_actual > self.output_limit
  self.phase = LUNA_WORKER_READY
}

///|
fn LunaTokenizerWorker::progress_compaction(
  self : LunaTokenizerWorker,
) -> Unit raise TokenizerError {
  if self.compact_node < 0 {
    return self.finish_compaction()
  }
  if self.truncation is Truncate && self.compact_actual >= self.output_limit {
    self.truncated = true
    self.output_count = self.output_limit
    self.phase = LUNA_WORKER_READY
    return
  }
  let node = self.compact_node
  self.compact_node = self.next[node]
  if self.tokens[node] == LUNA_PRETOKEN_BOUNDARY {
    return
  }
  if self.compact_actual < self.output_limit {
    self.tokens[self.compact_actual] = self.tokens[node]
  }
  self.compact_actual += 1
}
