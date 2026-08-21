///|
fn LunaTokenizerWorker::append_symbol(
  self : LunaTokenizerWorker,
  token_id : Int,
) -> Unit {
  let node = self.node_count
  if node >= self.input_capacity {
    abort("Luna tokenizer symbol capacity invariant")
  }
  self.node_count += 1
  self.tokens[node] = token_id
  self.previous[node] = self.tail
  self.next[node] = -1
  if self.tail >= 0 {
    self.next[self.tail] = node
  } else {
    self.head = node
  }
  self.tail = node
}

///|
fn LunaTokenizerWorker::start_special_match(self : LunaTokenizerWorker) -> Unit {
  self.special_index = 0
  self.special_piece_offset = 0
  self.special_matches = true
  self.best_special_id = -1
  self.best_special_bytes = 0
  self.phase = LUNA_WORKER_SPECIAL_MATCH
}

///|
fn LunaTokenizerWorker::finish_special_match(
  self : LunaTokenizerWorker,
) -> Unit raise TokenizerError {
  if self.best_special_id >= 0 {
    if self.special_policy is Reject {
      let rejected = self.best_special_id
      self.clear_request()
      raise SpecialTokenRejected(token_id=rejected)
    }
    self.append_symbol(self.best_special_id)
    self.input_offset += self.best_special_bytes
  } else {
    self.append_symbol(
      self.spec.byte_token_ids[self.input[self.input_offset].to_int()],
    )
    self.input_offset += 1
  }
  self.phase = LUNA_WORKER_INITIALIZE
}

///|
fn LunaTokenizerWorker::progress_special_match(
  self : LunaTokenizerWorker,
) -> Unit raise TokenizerError {
  if self.special_index >= self.spec.special_tokens.length() {
    return self.finish_special_match()
  }
  let candidate = self.spec.special_tokens[self.special_index]
  if self.special_piece_offset >= candidate.piece.length() {
    if candidate.piece.length() > self.best_special_bytes ||
      (
        candidate.piece.length() == self.best_special_bytes &&
        (self.best_special_id < 0 || candidate.token_id < self.best_special_id)
      ) {
      self.best_special_id = candidate.token_id
      self.best_special_bytes = candidate.piece.length()
    }
    self.special_index += 1
    self.special_piece_offset = 0
    self.special_matches = true
    return
  }
  if !self.special_matches ||
    self.input_offset + self.special_piece_offset >= self.input_length {
    self.special_index += 1
    self.special_piece_offset = 0
    self.special_matches = true
    return
  }
  if self.input[self.input_offset + self.special_piece_offset] !=
    candidate.piece[self.special_piece_offset] {
    self.special_matches = false
  } else {
    self.special_piece_offset += 1
  }
}

///|
fn LunaTokenizerWorker::progress_initialize(self : LunaTokenizerWorker) -> Unit {
  if self.input_offset >= self.input_length {
    self.scan_left = self.head
    self.selected_rank = -1
    self.phase = LUNA_WORKER_SELECT_ADJACENCY
  } else if self.special_policy is AsOrdinary ||
    self.spec.special_tokens.is_empty() {
    self.append_symbol(
      self.spec.byte_token_ids[self.input[self.input_offset].to_int()],
    )
    self.input_offset += 1
  } else {
    self.start_special_match()
  }
}

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
  self.output_count = Int::min(self.compact_actual, self.output_limit)
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
  if self.compact_actual < self.output_limit {
    self.tokens[self.compact_actual] = self.tokens[node]
  }
  self.compact_actual += 1
}

///|
fn LunaTokenizerWorker::progress_one(
  self : LunaTokenizerWorker,
) -> Unit raise TokenizerError {
  match self.phase {
    LUNA_WORKER_INITIALIZE => self.progress_initialize()
    LUNA_WORKER_SPECIAL_MATCH => self.progress_special_match()
    LUNA_WORKER_SELECT_ADJACENCY => self.progress_select_adjacency()
    LUNA_WORKER_SELECT_RULE => self.progress_select_rule()
    LUNA_WORKER_MERGE_BATCH => self.progress_merge_batch()
    LUNA_WORKER_COMPACT => self.progress_compaction()
    _ => abort("invalid Luna tokenizer worker phase")
  }
}

///|
pub fn LunaTokenizerWork::progress(
  self : LunaTokenizerWork,
) -> LunaTokenizerProgress raise TokenizerError {
  let worker = self.require_worker()
  if worker.phase == LUNA_WORKER_READY {
    worker.last_work_units = 0
    return LunaTokenizerReady
  }
  let maximum = worker.budget.work_units()
  let mut used = 0
  while used < maximum && worker.phase != LUNA_WORKER_READY {
    worker.progress_one()
    used += 1
  }
  worker.last_work_units = used
  if worker.phase == LUNA_WORKER_READY {
    LunaTokenizerReady
  } else {
    LunaTokenizerPending
  }
}
