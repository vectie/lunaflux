///|
fn LunaTokenizerWorker::append_symbol(
  self : LunaTokenizerWorker,
  token_id : Int32,
) -> Unit {
  let node = self.node_count
  if node >= self.node_capacity {
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
    self.segment_start = true
    self.ordinary_ready = false
    if self.spec.has_regex_pretokens() {
      self.qwen_segment_end = -1
    }
  } else if self.spec.profile == TOKENIZER_PROFILE_SENTENCEPIECE_BPE {
    self.ordinary_ready = true
  } else if self.spec.has_regex_pretokens() {
    self.append_symbol(
      self.spec.byte_token_ids[self.input[self.input_offset].to_int()],
    )
    self.input_offset += 1
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
fn LunaTokenizerWorker::progress_initialize(
  self : LunaTokenizerWorker,
) -> Unit raise TokenizerError {
  if self.post_processor_pending {
    self.append_symbol(self.spec.post_processor_token_id)
    self.post_processor_pending = false
  } else if self.input_offset >= self.input_length {
    self.scan_left = self.head
    self.selected_rank = -1
    if self.spec.ignore_merges {
      self.direct_start = -1
      self.direct_node = self.head
      self.phase = LUNA_WORKER_DIRECT_SEGMENT
    } else {
      self.phase = LUNA_WORKER_SELECT_ADJACENCY
    }
  } else if self.spec.has_regex_pretokens() {
    if self.qwen_segment_end >= 0 && self.input_offset >= self.qwen_segment_end {
      self.append_symbol(LUNA_PRETOKEN_BOUNDARY)
      self.qwen_segment_end = -1
    } else {
      if self.qwen_segment_end < 0 {
        self.qwen_segment_end = if self.spec.profile ==
          TOKENIZER_PROFILE_ORDERED_SCRIPT_BPE {
          ordered_script_pretoken_end(
            self.input,
            self.input_length,
            self.input_offset,
          )
        } else {
          qwen3_pretoken_end(
            self.input,
            self.input_length,
            self.input_offset,
            include_marks=self.spec.profile == TOKENIZER_PROFILE_QWEN35_BPE,
            digit_triplets=self.spec.profile ==
              TOKENIZER_PROFILE_DIGIT_TRIPLET_BPE,
          )
        }
      }
      if self.special_policy is AsOrdinary ||
        self.spec.special_tokens.is_empty() {
        self.append_symbol(
          self.spec.byte_token_ids[self.input[self.input_offset].to_int()],
        )
        self.input_offset += 1
      } else {
        self.start_special_match()
      }
    }
  } else if self.spec.profile == TOKENIZER_PROFILE_SENTENCEPIECE_BPE {
    if !self.ordinary_ready {
      if self.special_policy is AsOrdinary ||
        self.spec.special_tokens.is_empty() {
        self.ordinary_ready = true
      } else {
        self.start_special_match()
        return
      }
    }
    if self.segment_start {
      self.start_sentencepiece_scalar(true, 0, 3)
    } else if self.input[self.input_offset] == b' ' {
      self.start_sentencepiece_scalar(true, 1, 3)
    } else {
      let width = self.sentencepiece_scalar_width()
      self.start_sentencepiece_scalar(false, width, width)
    }
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
    LUNA_WORKER_SENTENCEPIECE_SCALAR => self.progress_sentencepiece_scalar()
    LUNA_WORKER_QWEN3_NORMALIZE => self.progress_qwen3_normalize()
    LUNA_WORKER_DIRECT_SEGMENT => self.progress_direct_segment()
    LUNA_WORKER_DIRECT_LOOKUP => self.progress_direct_lookup()
    LUNA_WORKER_DIRECT_COMPARE => self.progress_direct_compare()
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
