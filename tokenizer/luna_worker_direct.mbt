///|
// HF BPE ignore_merges means exact whole-pretoken vocabulary lookup before
// ranked BPE, not disabling BPE for an unmatched pretoken. The initialization
// list already contains special/regex boundaries, so matches cannot swallow
// a special token or cross a pretoken. Every transition is allocation-free.
fn LunaTokenizerWorker::progress_direct_segment(
  self : LunaTokenizerWorker,
) -> Unit {
  let node = self.direct_node
  let ordinary = node >= 0 &&
    self.tokens[node] >= 0 &&
    self.spec.tokens_by_id[self.tokens[node]].kind is Normal
  if ordinary {
    if self.direct_start < 0 {
      self.direct_start = node
    }
    self.direct_node = self.next[node]
  } else if self.direct_start >= 0 {
    self.direct_end = node
    self.direct_low = 0
    self.direct_high = self.spec.normal_piece_ids.length()
    self.phase = LUNA_WORKER_DIRECT_LOOKUP
  } else if node >= 0 {
    self.direct_node = self.next[node]
  } else {
    self.scan_left = self.head
    self.selected_rank = -1
    self.phase = LUNA_WORKER_SELECT_ADJACENCY
  }
}

///|
fn LunaTokenizerWorker::finish_direct_segment(
  self : LunaTokenizerWorker,
) -> Unit {
  self.direct_start = -1
  self.direct_node = self.direct_end
  self.phase = LUNA_WORKER_DIRECT_SEGMENT
}

///|
fn LunaTokenizerWorker::progress_direct_lookup(
  self : LunaTokenizerWorker,
) -> Unit {
  if self.direct_low >= self.direct_high {
    self.finish_direct_segment()
    return
  }
  self.direct_middle = self.direct_low +
    (self.direct_high - self.direct_low) / 2
  self.direct_node = self.direct_start
  self.direct_byte = 0
  self.phase = LUNA_WORKER_DIRECT_COMPARE
}

///|
fn LunaTokenizerWorker::progress_direct_compare(
  self : LunaTokenizerWorker,
) -> Unit {
  let token_id = self.spec.normal_piece_ids[self.direct_middle]
  let piece = self.spec.tokens_by_id[token_id].piece
  let ended = self.direct_node == self.direct_end
  let comparison = if ended || self.direct_byte >= piece.length() {
    if ended && self.direct_byte == piece.length() {
      0
    } else if ended {
      -1
    } else {
      1
    }
  } else {
    // Before BPE every normal symbol is one raw byte. Specials and regex
    // boundaries terminate this scan and are not in the sorted normal index.
    let byte = self.spec.tokens_by_id[self.tokens[self.direct_node]].piece[0]
    let compared = byte.compare(piece[self.direct_byte])
    if compared == 0 {
      self.direct_node = self.next[self.direct_node]
      self.direct_byte += 1
      return
    }
    compared
  }
  if comparison < 0 {
    self.direct_high = self.direct_middle
    self.phase = LUNA_WORKER_DIRECT_LOOKUP
  } else if comparison > 0 {
    self.direct_low = self.direct_middle + 1
    self.phase = LUNA_WORKER_DIRECT_LOOKUP
  } else {
    let first = self.direct_start
    self.tokens[first] = token_id
    self.next[first] = self.direct_end
    if self.direct_end >= 0 {
      self.previous[self.direct_end] = first
    } else {
      self.tail = first
    }
    self.finish_direct_segment()
  }
}
