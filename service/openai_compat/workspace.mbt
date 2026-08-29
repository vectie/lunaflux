///|
fn luna_openai_requirements(
  inference : @inference.InferenceLimits,
) -> (Int, UInt64, UInt64) raise LunaOpenAICompatError {
  let delta = inference.max_decoded_delta_bytes()
  if delta <= 0 || delta > 16384 {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatLimits)
  }
  let escaped_dynamic = (
      delta.to_uint64() +
      LUNA_OPENAI_COMPAT_MAX_ERROR_CODE_BYTES.to_uint64() +
      LUNA_OPENAI_COMPAT_MAX_MODEL_ALIAS_BYTES.to_uint64()
    ) *
    6UL
  let output = LUNA_OPENAI_COMPAT_FIXED_OUTPUT_BYTES +
    escaped_dynamic +
    LUNA_OPENAI_COMPAT_MAX_RESPONSE_PREFIX_BYTES.to_uint64() +
    20UL
  let total = output +
    LUNA_OPENAI_COMPAT_MAX_MODEL_ALIAS_BYTES.to_uint64() +
    LUNA_OPENAI_COMPAT_MAX_RESPONSE_PREFIX_BYTES.to_uint64()
  if output > 2147483647UL || total > 2147483647UL {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatCapacity)
  }
  (output.to_int(), total, 1UL)
}

///|
pub fn LunaOpenAICompatWorkspace::required_byte_cells(
  inference : @inference.InferenceLimits,
) -> UInt64 raise LunaOpenAICompatError {
  let (_, bytes, _) = luna_openai_requirements(inference)
  bytes
}

///|
pub fn LunaOpenAICompatWorkspace::required_reference_cells(
  inference : @inference.InferenceLimits,
) -> UInt64 raise LunaOpenAICompatError {
  let (_, _, references) = luna_openai_requirements(inference)
  references
}

///|
pub fn LunaOpenAICompatWorkspace::new(
  inference : @inference.InferenceLimits,
  budget : LunaOpenAICompatStepBudget,
  deployment_model_alias : FixedArray[Byte],
  model_alias_offset~ : Int,
  model_alias_length~ : Int,
  response_id_prefix : FixedArray[Byte],
  response_id_prefix_offset~ : Int,
  response_id_prefix_length~ : Int,
) -> LunaOpenAICompatWorkspace raise LunaOpenAICompatError {
  let (output_capacity, _, _) = luna_openai_requirements(inference)
  budget.validate()
  if !luna_openai_range_is_valid(
      deployment_model_alias.length(),
      model_alias_offset,
      model_alias_length,
    ) {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatModelAlias)
  }
  if !luna_openai_range_is_valid(
      response_id_prefix.length(),
      response_id_prefix_offset,
      response_id_prefix_length,
    ) {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatResponseId)
  }
  if model_alias_length <= 0 ||
    model_alias_length > LUNA_OPENAI_COMPAT_MAX_MODEL_ALIAS_BYTES ||
    !luna_openai_valid_utf8(
      deployment_model_alias, model_alias_offset, model_alias_length,
    ) {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatModelAlias)
  }
  if response_id_prefix_length <= 0 ||
    response_id_prefix_length > LUNA_OPENAI_COMPAT_MAX_RESPONSE_PREFIX_BYTES {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatResponseId)
  }
  for index in 0..<response_id_prefix_length {
    if !luna_openai_safe_id_byte(
        response_id_prefix[response_id_prefix_offset + index],
      ) {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatResponseId)
    }
  }
  let model_storage : FixedArray[Byte] = FixedArray::make(
    LUNA_OPENAI_COMPAT_MAX_MODEL_ALIAS_BYTES,
    0,
  )
  for index in 0..<model_alias_length {
    model_storage[index] = deployment_model_alias[model_alias_offset + index]
  }
  let prefix_storage : FixedArray[Byte] = FixedArray::make(
    LUNA_OPENAI_COMPAT_MAX_RESPONSE_PREFIX_BYTES,
    0,
  )
  for index in 0..<response_id_prefix_length {
    prefix_storage[index] = response_id_prefix[response_id_prefix_offset + index]
  }
  {
    inference,
    budget,
    storage: FixedArray::make(output_capacity, 0),
    model_alias: model_storage,
    model_alias_length,
    response_id_prefix: prefix_storage,
    response_id_prefix_length,
    semantic_views: Array::new(capacity=1),
    epoch: 0UL,
    phase: LUNA_OPENAI_COMPAT_IDLE,
    mode: 0,
    event_kind: 0,
    response_sequence: 0UL,
    setup_cursor: 0,
    part_index: 0,
    part_kind: 0,
    part_cursor: 0,
    source_cursor: 0,
    current_byte: 0,
    escape_kind: 0,
    escape_cursor: 0,
    escape_length: 0,
    decimal_value: 0UL,
    decimal_divisor: 1UL,
    token_id: 0,
    position: 0,
    input_tokens: 0,
    cached_input_tokens: 0,
    output_tokens: 0,
    total_tokens: 0,
    finish_reason: 0,
    retryable: false,
    payload_length: 0,
    length: 0,
    failure_rule: 0,
    last_work_units: 0,
    total_work_units: 0UL,
  }
}

///|
fn LunaOpenAICompatWorkspace::reset_operation(
  self : LunaOpenAICompatWorkspace,
) -> Unit {
  self.semantic_views.clear()
  self.mode = 0
  self.event_kind = 0
  self.response_sequence = 0UL
  self.setup_cursor = 0
  self.part_index = 0
  self.part_kind = 0
  self.part_cursor = 0
  self.source_cursor = 0
  self.current_byte = 0
  self.escape_kind = 0
  self.escape_cursor = 0
  self.escape_length = 0
  self.decimal_value = 0UL
  self.decimal_divisor = 1UL
  self.token_id = 0
  self.position = 0
  self.input_tokens = 0
  self.cached_input_tokens = 0
  self.output_tokens = 0
  self.total_tokens = 0
  self.finish_reason = 0
  self.retryable = false
  self.payload_length = 0
  self.length = 0
  self.failure_rule = 0
  self.last_work_units = 0
  self.total_work_units = 0UL
}

///|
fn luna_openai_event_kind(kind : @luna_event.LunaEventKind) -> Int {
  match kind {
    LunaAcceptedEventKind => LUNA_OPENAI_EVENT_ACCEPTED
    LunaTokenEventKind => LUNA_OPENAI_EVENT_TOKEN
    LunaUsageEventKind => LUNA_OPENAI_EVENT_USAGE
    LunaCompletedEventKind => LUNA_OPENAI_EVENT_COMPLETED
    LunaFailedEventKind => LUNA_OPENAI_EVENT_FAILED
  }
}

///|
pub fn LunaOpenAICompatWorkspace::begin(
  self : LunaOpenAICompatWorkspace,
  event : @luna_event.LunaEventView,
  mode : LunaOpenAICompatMode,
  response_sequence~ : UInt64,
) -> LunaOpenAICompatWork raise LunaOpenAICompatError {
  if self.phase != LUNA_OPENAI_COMPAT_IDLE {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatLifecycle)
  }
  if response_sequence == 0UL {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatResponseId)
  }
  let kind = event.kind() catch {
    _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
  }
  if self.epoch == 0xffffffffffffffffUL {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatEpoch)
  }
  self.reset_operation()
  if self.semantic_views.length() != 0 {
    abort("Luna OpenAI semantic reference-slot invariant")
  }
  self.epoch += 1UL
  self.mode = match mode {
    LunaOpenAIResponses => 1
    LunaOpenAIChatCompletions => 2
  }
  self.event_kind = luna_openai_event_kind(kind)
  self.response_sequence = response_sequence
  self.semantic_views.push(event)
  self.phase = LUNA_OPENAI_COMPAT_SETUP
  { workspace: self, epoch: self.epoch }
}

///|
fn LunaOpenAICompatWorkspace::event_view(
  self : LunaOpenAICompatWorkspace,
) -> @luna_event.LunaEventView {
  if self.semantic_views.length() != 1 {
    abort("Luna OpenAI compatibility semantic-view invariant")
  }
  self.semantic_views[0]
}

///|
fn LunaOpenAICompatWorkspace::authenticate_event(
  self : LunaOpenAICompatWorkspace,
) -> @luna_event.LunaEventView raise LunaOpenAICompatError {
  let event = self.event_view()
  let kind = event.kind() catch {
    _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
  }
  if luna_openai_event_kind(kind) != self.event_kind {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
  }
  event
}
