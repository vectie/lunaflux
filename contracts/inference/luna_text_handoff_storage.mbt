///|
pub fn LunaTextRequestHandoffStorage::required_byte_cells(
  limits : InferenceLimits,
) -> UInt64 raise InferenceContractError {
  limits.max_text_bytes().to_uint64() +
  limits.max_cache_scope_bytes().to_uint64() +
  LunaRequestSemanticStorage::required_byte_cells(limits)
}

///|
pub fn LunaTextRequestHandoffStorage::required_int_cells(
  limits : InferenceLimits,
) -> UInt64 raise InferenceContractError {
  LunaRequestSemanticStorage::required_int_cells(limits)
}

///|
pub fn LunaTextRequestHandoffStorage::required_reference_cells() -> UInt64 {
  3UL
}

///|
pub fn LunaTextRequestHandoffStorage::new(
  limits : InferenceLimits,
  model_identity : @spec.ModelIdentity,
  cache_scope : CacheScope,
  semantic_budget : LunaRequestSemanticStepBudget,
) -> Self raise InferenceContractError {
  ignore(LunaTextRequestHandoffStorage::required_byte_cells(limits))
  ignore(LunaTextRequestHandoffStorage::required_int_cells(limits))
  let scope = @utf8.encode(cache_scope.as_string())
  if scope.length() <= 0 || scope.length() > limits.max_cache_scope_bytes() {
    raise invalid(LunaRequestSemantic, OutOfRange, scope.length())
  }
  let scope_storage : FixedArray[Byte] = FixedArray::make(
    limits.max_cache_scope_bytes(),
    b'\x00',
  )
  for index in 0..<scope.length() {
    scope_storage[index] = scope[index]
  }
  let default_effective = EffectiveLimits::new(
    max_new_tokens=1,
    context_ceiling=2,
    limits,
  )
  {
    limits,
    model_identity,
    cache_scope: scope_storage,
    cache_scope_length: scope.length(),
    prompt: FixedArray::make(limits.max_text_bytes(), b'\x00'),
    semantics: LunaRequestSemanticStorage::new(limits, semantic_budget),
    semantic_writes: Array::new(capacity=1),
    semantic_works: Array::new(capacity=1),
    semantic_leases: Array::new(capacity=1),
    epoch: 0UL,
    phase: LUNA_TEXT_HANDOFF_IDLE,
    prompt_limit: 0,
    prompt_length: 0,
    prompt_cursor: 0,
    request_id: RequestId::new(1UL),
    effective: default_effective,
    sampling: LunaValidatedSampling::greedy(),
    sampling_seed: SamplingSeed::new(1UL),
    stream_preference: Streaming,
    deadline: DeadlineBudget::from_milliseconds(1UL, limits),
  }
}

///|
fn LunaTextRequestHandoffStorage::reset_scalars(self : Self) -> Unit {
  self.prompt_limit = 0
  self.prompt_length = 0
  self.prompt_cursor = 0
}

///|
pub fn LunaTextRequestHandoffStorage::begin(
  self : Self,
  prompt_byte_limit~ : Int,
  stop_token_count~ : Int,
  stop_string_count~ : Int,
  request_id~ : RequestId,
  effective~ : EffectiveLimits,
  sampling~ : LunaValidatedSampling,
  sampling_seed~ : SamplingSeed,
  stream_preference~ : StreamPreference,
  deadline~ : DeadlineBudget,
) -> LunaTextRequestHandoffWrite raise InferenceContractError {
  if self.phase != LUNA_TEXT_HANDOFF_IDLE ||
    prompt_byte_limit <= 0 ||
    prompt_byte_limit > self.limits.max_text_bytes() ||
    stop_token_count < 0 ||
    stop_token_count > self.limits.max_stop_token_ids() ||
    stop_string_count < 0 ||
    stop_string_count > self.limits.max_stop_strings() ||
    effective.max_new_tokens() <= 0 ||
    effective.context_ceiling() > self.limits.max_context_tokens() ||
    request_id.value() == 0UL ||
    sampling_seed.value() == 0UL ||
    deadline.milliseconds() == 0UL ||
    deadline.milliseconds() > self.limits.max_deadline_millis() {
    raise invalid(LunaRequestSemantic, OutOfRange, prompt_byte_limit)
  }
  if self.epoch == 0xffffffffffffffffUL {
    raise invalid(LunaRequestSemantic, LunaStorageStale, 0)
  }
  let semantic_write = self.semantics.begin(
    stop_token_count~,
    stop_string_count~,
    cache_scope_byte_count=self.cache_scope_length,
    cache_permission=Disabled,
  )
  self.reset_scalars()
  self.epoch += 1UL
  self.prompt_limit = prompt_byte_limit
  self.request_id = request_id
  self.effective = effective
  self.sampling = sampling
  self.sampling_seed = sampling_seed
  self.stream_preference = stream_preference
  self.deadline = deadline
  self.semantic_writes.push(semantic_write)
  self.phase = LUNA_TEXT_HANDOFF_WRITING
  { storage: self, epoch: self.epoch }
}
