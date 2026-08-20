///|
fn row_error(
  kind : WorkerRowKind,
  index : Int,
  rule : RowRule,
) -> Unit raise WorkerProtocolError {
  raise InvalidRow(kind~, index~, rule~)
}

///|
fn validate_actual_range(
  kind : WorkerRowKind,
  index : Int,
  start : Int,
  length : Int,
  actual_length : Int,
  expected_start : Int,
) -> Int raise WorkerProtocolError {
  if start < 0 {
    row_error(kind, index, NegativeRangeStart)
  }
  if length <= 0 {
    row_error(kind, index, EmptyRange)
  }
  if start > actual_length || length > actual_length - start {
    row_error(kind, index, RangeOutsideTable)
  }
  if start != expected_start {
    row_error(kind, index, NonContiguousRange)
  }
  start + length
}

///|
fn validate_token(
  token : Int,
  index : Int,
  limits : WorkerProtocolLimits,
) -> Unit raise WorkerProtocolError {
  if token < 0 || token > limits.max_token_id() {
    raise InvalidToken(index~)
  }
}

///|
fn validate_plan_counts(
  prefill_count : Int,
  decode_count : Int,
  token_count : Int,
  page_count : Int,
  capability_count : Int,
  limits : WorkerProtocolLimits,
) -> Unit raise WorkerProtocolError {
  if prefill_count == 0 && decode_count == 0 {
    raise EmptyPlan
  }
  if prefill_count > limits.max_prefill_rows() {
    raise BoundExceeded(PrefillRows)
  }
  if decode_count > limits.max_decode_rows() {
    raise BoundExceeded(DecodeRows)
  }
  if prefill_count > limits.max_plan_rows() - decode_count {
    raise BoundExceeded(PlanRows)
  }
  if token_count > limits.max_plan_tokens() {
    raise BoundExceeded(PlanTokens)
  }
  if page_count > limits.max_plan_pages() {
    raise BoundExceeded(PlanPages)
  }
  if capability_count > limits.max_capabilities() {
    raise BoundExceeded(Capabilities)
  }
}

///|
fn validate_plan_finish(
  token_end : Int,
  token_count : Int,
  page_end : Int,
  page_count : Int,
  capability_end : Int,
  capability_count : Int,
  scheduled_tokens : Int,
  token_budget : Int,
  limits : WorkerProtocolLimits,
) -> Unit raise WorkerProtocolError {
  if token_end != token_count {
    raise IncompleteTable(Tokens)
  }
  if page_end != page_count {
    raise IncompleteTable(Pages)
  }
  if capability_end != capability_count {
    raise IncompleteTable(Capabilities)
  }
  if token_budget <= 0 ||
    token_budget > limits.max_plan_tokens() ||
    token_budget != scheduled_tokens {
    raise TokenBudgetMismatch
  }
}

///|
fn validate_submission_identity_wire(
  sequence_value : UInt64,
  model_plan_generation_value : UInt64,
  has_previous : Bool,
  previous_sequence_value : UInt64,
  loaded_generation_value : UInt64,
) -> Unit raise WorkerProtocolError {
  if sequence_value == 0UL {
    raise InvalidIdentity(PlanSequenceIdentity)
  }
  if model_plan_generation_value == 0UL || loaded_generation_value == 0UL {
    raise InvalidIdentity(ModelPlanGenerationIdentity)
  }
  if has_previous {
    if previous_sequence_value == 0UL {
      raise InvalidIdentity(PlanSequenceIdentity)
    }
    if previous_sequence_value == 0xffffffffffffffffUL ||
      sequence_value != previous_sequence_value + 1UL {
      raise StalePlanSequence
    }
  } else if previous_sequence_value != 0UL || sequence_value != 1UL {
    raise StalePlanSequence
  }
  if model_plan_generation_value != loaded_generation_value {
    raise StaleModelPlanGeneration
  }
}
