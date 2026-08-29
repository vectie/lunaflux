///|
pub fn LunaOpenAIInboundWorkspace::required_byte_cells(
  inference : @inference.InferenceLimits,
) -> UInt64 raise LunaOpenAICompatError {
  256UL +
  inference.max_stop_strings().to_uint64() *
  inference.max_stop_string_bytes().to_uint64() +
  (@inference.LunaTextRequestHandoffStorage::required_byte_cells(inference) catch {
    _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatLimits)
  })
}

///|
pub fn LunaOpenAIInboundWorkspace::required_int_cells(
  inference : @inference.InferenceLimits,
  limits : LunaOpenAIInboundLimits,
) -> UInt64 raise LunaOpenAICompatError {
  if limits.max_messages <= 0 || limits.max_messages > 64 {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatLimits)
  }
  limits.max_messages.to_uint64() * 3UL +
  inference.max_stop_token_ids().to_uint64() +
  inference.max_stop_strings().to_uint64() * 2UL
}

///|
pub fn LunaOpenAIInboundWorkspace::required_reference_cells() -> UInt64 {
  5UL + @inference.LunaTextRequestHandoffStorage::required_reference_cells()
}

///|
pub fn LunaOpenAIInboundWorkspace::new(
  inference : @inference.InferenceLimits,
  budget : LunaOpenAICompatStepBudget,
  limits : LunaOpenAIInboundLimits,
  template : LunaOpenAIChatTemplate,
  deployment_model_alias : FixedArray[Byte],
  model_alias_offset~ : Int,
  model_alias_length~ : Int,
  identity : @spec.ModelIdentity,
  cache_scope : @inference.CacheScope,
  max_new_tokens~ : Int,
  context_ceiling~ : Int,
  sampling : @inference.SamplingParameters,
  sampling_seed : @inference.SamplingSeed,
  deadline : @inference.DeadlineBudget,
) -> Self raise LunaOpenAICompatError {
  ignore(LunaOpenAIInboundWorkspace::required_byte_cells(inference))
  ignore(LunaOpenAIInboundWorkspace::required_int_cells(inference, limits))
  budget.validate()
  template.validate_for(inference)
  if !luna_openai_range_is_valid(
      deployment_model_alias.length(),
      model_alias_offset,
      model_alias_length,
    ) ||
    model_alias_length <= 0 ||
    model_alias_length > 256 ||
    !luna_openai_valid_utf8(
      deployment_model_alias, model_alias_offset, model_alias_length,
    ) ||
    max_new_tokens <= 0 ||
    max_new_tokens > inference.max_new_tokens() ||
    context_ceiling <= max_new_tokens ||
    context_ceiling > inference.max_context_tokens() ||
    deadline.milliseconds() == 0UL ||
    deadline.milliseconds() > inference.max_deadline_millis() {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatLimits)
  }
  let model_storage : FixedArray[Byte] = FixedArray::make(256, 0)
  for index in 0..<model_alias_length {
    model_storage[index] = deployment_model_alias[model_alias_offset + index]
  }
  let effective = @inference.EffectiveLimits::new(
    max_new_tokens~,
    context_ceiling~,
    inference,
  ) catch {
    _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatLimits)
  }
  let handoff_workspace = @inference.LunaTextRequestHandoffStorage::new(
    inference,
    identity,
    cache_scope,
    @inference.LunaRequestSemanticStepBudget::new(1) catch {
      _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatFrame)
    },
  ) catch {
    _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatFrame)
  }
  {
    inference,
    limits,
    budget,
    template,
    model_alias: model_storage,
    model_alias_length,
    message_cells: FixedArray::make(limits.max_messages * 3, 0),
    stop_tokens: FixedArray::make(inference.max_stop_token_ids(), 0),
    stop_string_cells: FixedArray::make(inference.max_stop_strings() * 2, 0),
    stop_string_bytes: FixedArray::make(
      inference.max_stop_strings() * inference.max_stop_string_bytes(),
      0,
    ),
    handoff_workspace,
    effective,
    sampling,
    sampling_seed,
    deadline,
    http_views: Array::new(capacity=1),
    request_ids: Array::new(capacity=1),
    handoff_writes: Array::new(capacity=1),
    handoff_works: Array::new(capacity=1),
    handoff_leases: Array::new(capacity=1),
    epoch: 0UL,
    phase: LUNA_INBOUND_IDLE,
    route: 0,
    body_length: 0,
    cursor: 0,
    top_seen: 0,
    message_seen: 0,
    current_key: 0,
    message_count: 0,
    current_role: 0,
    current_content_start: 0,
    current_content_length: 0,
    after_comma: false,
    string_purpose: 0,
    string_return_phase: 0,
    string_start: 0,
    string_decoded_index: 0,
    string_candidates: 0,
    escape_state: 0,
    unicode_value: 0,
    unicode_digits: 0,
    high_surrogate: 0,
    pending_codepoint: 0,
    pending_width: 0,
    pending_index: 0,
    utf8_remaining: 0,
    utf8_codepoint: 0,
    utf8_minimum: 0,
    true_index: 0,
    input_start: 0,
    input_length: 0,
    sampling_mode: if sampling.mode() == Sample {
      1
    } else {
      0
    },
    temperature: sampling.temperature(),
    has_top_k: sampling.top_k() is Some(_),
    top_k: sampling.top_k().unwrap_or(0),
    has_top_p: sampling.top_p() is Some(_),
    top_p: sampling.top_p().unwrap_or(0.0),
    sampling_seed_value: sampling_seed.value(),
    stop_token_count: 0,
    stop_string_count: 0,
    stop_string_byte_count: 0,
    current_stop_start: 0,
    current_stop_length: 0,
    number_return_phase: 0,
    number_target: 0,
    number_stage: 0,
    number_negative: false,
    number_significand: 0UL,
    number_fraction_digits: 0,
    number_exponent_negative: false,
    number_exponent: 0,
    number_leading_zero: false,
    number_value: 0.0,
    number_scale: 0,
    render_message: 0,
    render_segment: 0,
    render_segment_index: 0,
    render_source_start: 0,
    render_source_length: 0,
    render_source_index: 0,
    prompt_length: 0,
    handoff_push_stage: 0,
    handoff_stop_index: 0,
    handoff_stop_byte: -1,
    failure_rule: 0,
    last_work_units: 0,
    total_work_units: 0UL,
  }
}

///|
fn LunaOpenAIInboundWorkspace::reset_operation(self : Self) -> Unit {
  self.http_views.clear()
  self.request_ids.clear()
  self.handoff_writes.clear()
  self.handoff_works.clear()
  self.handoff_leases.clear()
  self.route = 0
  self.body_length = 0
  self.cursor = 0
  self.top_seen = 0
  self.message_seen = 0
  self.current_key = 0
  self.message_count = 0
  self.current_role = 0
  self.current_content_start = 0
  self.current_content_length = 0
  self.after_comma = false
  self.string_purpose = 0
  self.escape_state = 0
  self.pending_width = 0
  self.utf8_remaining = 0
  self.high_surrogate = 0
  self.input_start = 0
  self.input_length = 0
  self.sampling_mode = if self.sampling.mode() == Sample { 1 } else { 0 }
  self.temperature = self.sampling.temperature()
  self.has_top_k = self.sampling.top_k() is Some(_)
  self.top_k = self.sampling.top_k().unwrap_or(0)
  self.has_top_p = self.sampling.top_p() is Some(_)
  self.top_p = self.sampling.top_p().unwrap_or(0.0)
  self.sampling_seed_value = self.sampling_seed.value()
  self.stop_token_count = 0
  self.stop_string_count = 0
  self.stop_string_byte_count = 0
  self.current_stop_start = 0
  self.current_stop_length = 0
  self.number_stage = 0
  self.number_significand = 0UL
  self.number_fraction_digits = 0
  self.number_exponent = 0
  self.number_value = 0.0
  self.number_scale = 0
  self.render_message = 0
  self.prompt_length = 0
  self.handoff_push_stage = 0
  self.handoff_stop_index = 0
  self.handoff_stop_byte = -1
  self.failure_rule = 0
  self.last_work_units = 0
  self.total_work_units = 0UL
}

///|
pub fn LunaOpenAIInboundWorkspace::begin(
  self : Self,
  http : @http1.LunaHttp1View,
  request_id : @inference.RequestId,
) -> LunaOpenAIInboundWork raise LunaOpenAICompatError {
  if self.phase != LUNA_INBOUND_IDLE {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatLifecycle)
  }
  let route = http.route() catch {
    _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatHttp)
  }
  let length = http.body_length() catch {
    _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatHttp)
  }
  if length <= 0 || request_id.value() == 0UL {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
  }
  if self.epoch == 0xffffffffffffffffUL {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatEpoch)
  }
  self.reset_operation()
  if self.http_views.length() != 0 || self.request_ids.length() != 0 {
    abort("Luna OpenAI inbound reference-slot invariant")
  }
  self.epoch += 1UL
  self.route = match route {
    LunaHttp1Responses => 1
    LunaHttp1ChatCompletions => 2
  }
  self.body_length = length
  self.http_views.push(http)
  self.request_ids.push(request_id)
  self.phase = LUNA_INBOUND_PARSE_START
  { workspace: self, epoch: self.epoch }
}

///|
fn LunaOpenAIInboundWorkspace::http(self : Self) -> @http1.LunaHttp1View {
  if self.http_views.length() != 1 {
    abort("Luna OpenAI inbound HTTP invariant")
  }
  self.http_views[0]
}

///|
fn LunaOpenAIInboundWorkspace::body_byte(
  self : Self,
  index : Int,
) -> Byte raise LunaOpenAICompatError {
  self.http().body_byte_at(index) catch {
    _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatHttp)
  }
}
