///|
/// Immutable scheduler-to-worker command. Construction defensively copies all
/// table and row inputs and verifies that row views consume every flattened
/// table exactly once in prefill-then-decode order. This payload-bearing type
/// deliberately has no `Debug` implementation.
pub struct SchedulePlan {
  sequence : PlanSequence
  model_plan_generation : ModelPlanGeneration
  token_budget : Int
  tokens : ReadOnlyArray[Int]
  pages : ReadOnlyArray[@page_allocator.PageId]
  capabilities : ReadOnlyArray[@plan.KernelCapabilityId]
  prefill_rows : ReadOnlyArray[PrefillRow]
  decode_rows : ReadOnlyArray[DecodeRow]
} derive(Eq)

///|
fn is_duplicate_request(
  request_id : @inference.RequestId,
  prefill_rows : ArrayView[PrefillRow],
  prefill_end : Int,
  decode_rows : ArrayView[DecodeRow],
  decode_end : Int,
) -> Bool {
  for index in 0..<prefill_end {
    if prefill_rows[index].request_id() == request_id {
      return true
    }
  }
  for index in 0..<decode_end {
    if decode_rows[index].request_id() == request_id {
      return true
    }
  }
  false
}

///|
fn is_duplicate_slot(
  slot : CompletionSlot,
  prefill_rows : ArrayView[PrefillRow],
  prefill_end : Int,
  decode_rows : ArrayView[DecodeRow],
  decode_end : Int,
) -> Bool {
  for index in 0..<prefill_end {
    if prefill_rows[index].completion_slot() == slot {
      return true
    }
  }
  for index in 0..<decode_end {
    if decode_rows[index].completion_slot() == slot {
      return true
    }
  }
  false
}

///|
fn validate_tokens(
  tokens : ArrayView[Int],
  limits : WorkerProtocolLimits,
) -> Unit raise WorkerProtocolError {
  if tokens.length() > limits.max_plan_tokens() {
    raise BoundExceeded(PlanTokens)
  }
  for index, token in tokens {
    validate_token(token, index, limits)
  }
}

///|
/// Constructs an exact immutable plan. `token_budget` counts scheduled input
/// tokens: every prefill chunk token plus one token per decode row.
pub fn SchedulePlan::new(
  sequence~ : PlanSequence,
  model_plan_generation~ : ModelPlanGeneration,
  token_budget~ : Int,
  tokens~ : ArrayView[Int],
  pages~ : ArrayView[@page_allocator.PageId],
  capabilities~ : ArrayView[@plan.KernelCapabilityId],
  prefill_rows~ : ArrayView[PrefillRow],
  decode_rows~ : ArrayView[DecodeRow],
  limits~ : WorkerProtocolLimits,
) -> SchedulePlan raise WorkerProtocolError {
  let prefill_count = prefill_rows.length()
  let decode_count = decode_rows.length()
  validate_plan_counts(
    prefill_count,
    decode_count,
    tokens.length(),
    pages.length(),
    capabilities.length(),
    limits,
  )
  validate_tokens(tokens, limits)
  let mut token_end = 0
  let mut page_end = 0
  let mut capability_end = 0
  let mut scheduled_tokens = 0
  for index, row in prefill_rows {
    if is_duplicate_request(
        row.request_id(),
        prefill_rows,
        index,
        decode_rows,
        0,
      ) {
      row_error(Prefill, index, DuplicateRequest)
    }
    if is_duplicate_slot(
        row.completion_slot(),
        prefill_rows,
        index,
        decode_rows,
        0,
      ) {
      row_error(Prefill, index, DuplicateCompletionSlot)
    }
    token_end = validate_actual_range(
      Prefill,
      index,
      row.tokens().start(),
      row.tokens().length(),
      tokens.length(),
      token_end,
    )
    page_end = validate_actual_range(
      Prefill,
      index,
      row.page_table().start(),
      row.page_table().length(),
      pages.length(),
      page_end,
    )
    capability_end = validate_actual_range(
      Prefill,
      index,
      row.capabilities().start(),
      row.capabilities().length(),
      capabilities.length(),
      capability_end,
    )
    if scheduled_tokens > limits.max_plan_tokens() - row.tokens().length() {
      raise BoundExceeded(PlanTokens)
    }
    scheduled_tokens = scheduled_tokens + row.tokens().length()
  }
  for index, row in decode_rows {
    if is_duplicate_request(
        row.request_id(),
        prefill_rows,
        prefill_count,
        decode_rows,
        index,
      ) {
      row_error(Decode, index, DuplicateRequest)
    }
    if is_duplicate_slot(
        row.completion_slot(),
        prefill_rows,
        prefill_count,
        decode_rows,
        index,
      ) {
      row_error(Decode, index, DuplicateCompletionSlot)
    }
    token_end = validate_actual_range(
      Decode,
      index,
      row.input_token().start(),
      row.input_token().length(),
      tokens.length(),
      token_end,
    )
    page_end = validate_actual_range(
      Decode,
      index,
      row.page_table().start(),
      row.page_table().length(),
      pages.length(),
      page_end,
    )
    capability_end = validate_actual_range(
      Decode,
      index,
      row.capabilities().start(),
      row.capabilities().length(),
      capabilities.length(),
      capability_end,
    )
    if scheduled_tokens >= limits.max_plan_tokens() {
      raise BoundExceeded(PlanTokens)
    }
    scheduled_tokens = scheduled_tokens + 1
  }
  validate_plan_finish(
    token_end,
    tokens.length(),
    page_end,
    pages.length(),
    capability_end,
    capabilities.length(),
    scheduled_tokens,
    token_budget,
    limits,
  )
  {
    sequence,
    model_plan_generation,
    token_budget,
    tokens: ReadOnlyArray::from_array(tokens),
    pages: ReadOnlyArray::from_array(pages),
    capabilities: ReadOnlyArray::from_array(capabilities),
    prefill_rows: ReadOnlyArray::from_array(prefill_rows),
    decode_rows: ReadOnlyArray::from_array(decode_rows),
  }
}

///|
pub fn SchedulePlan::sequence(self : SchedulePlan) -> PlanSequence {
  self.sequence
}

///|
pub fn SchedulePlan::model_plan_generation(
  self : SchedulePlan,
) -> ModelPlanGeneration {
  self.model_plan_generation
}

///|
pub fn SchedulePlan::token_budget(self : SchedulePlan) -> Int {
  self.token_budget
}

///|
pub fn SchedulePlan::tokens(self : SchedulePlan) -> ReadOnlyArray[Int] {
  self.tokens
}

///|
pub fn SchedulePlan::pages(
  self : SchedulePlan,
) -> ReadOnlyArray[@page_allocator.PageId] {
  self.pages
}

///|
pub fn SchedulePlan::capabilities(
  self : SchedulePlan,
) -> ReadOnlyArray[@plan.KernelCapabilityId] {
  self.capabilities
}

///|
pub fn SchedulePlan::prefill_rows(
  self : SchedulePlan,
) -> ReadOnlyArray[PrefillRow] {
  self.prefill_rows
}

///|
pub fn SchedulePlan::decode_rows(
  self : SchedulePlan,
) -> ReadOnlyArray[DecodeRow] {
  self.decode_rows
}

///|
pub fn SchedulePlan::row_count(self : SchedulePlan) -> Int {
  self.prefill_rows.length() + self.decode_rows.length()
}

///|
/// Validates the exact submission order and loaded model generation without
/// changing either value.
pub fn SchedulePlan::validate_submission(
  self : SchedulePlan,
  previous_sequence : PlanSequence?,
  loaded_generation : ModelPlanGeneration,
) -> Unit raise WorkerProtocolError {
  validate_submission_identity(
    self.sequence,
    self.model_plan_generation,
    previous_sequence,
    loaded_generation,
  )
}
