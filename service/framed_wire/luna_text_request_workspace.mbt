///|
pub fn LunaFramedTextRequestWorkspace::required_byte_cells(
  limits : FramedWireLimits,
) -> UInt64 raise FramedWireError {
  if limits.max_frame_bytes() < REQUEST_HEADER_BYTES {
    raise InvalidLimit(FrameBytes)
  }
  let inference = limits.inference_limits()
  limits.max_frame_bytes().to_uint64() +
  128UL +
  inference.max_cache_scope_bytes().to_uint64()
}

///|
pub fn LunaFramedTextRequestWorkspace::required_reference_cells(
  _limits : FramedWireLimits,
) -> UInt64 raise FramedWireError {
  ignore(LunaFramedTextRequestWorkspace::required_byte_cells(_limits))
  0UL
}

///|
pub fn LunaFramedTextRequestWorkspace::new(
  limits : FramedWireLimits,
  budget : LunaFramedTextRequestStepBudget,
  identity : @spec.ModelIdentity,
  cache_scope : @inference.CacheScope,
) -> Self raise FramedWireError {
  let inference = limits.inference_limits()
  let content = @utf8.encode(identity.content().as_hex())
  let plan = @utf8.encode(identity.plan().as_hex())
  let scope = @utf8.encode(cache_scope.as_string())
  if content.length() != 64 ||
    plan.length() != 64 ||
    scope.length() <= 0 ||
    scope.length() > inference.max_cache_scope_bytes() {
    raise InvalidFrame(frame=RequestFrame, rule=Identity)
  }
  let content_store : FixedArray[Byte] = FixedArray::make(64, 0)
  let plan_store : FixedArray[Byte] = FixedArray::make(64, 0)
  let scope_store : FixedArray[Byte] = FixedArray::make(
    inference.max_cache_scope_bytes(),
    0,
  )
  for index in 0..<64 {
    content_store[index] = content[index]
    plan_store[index] = plan[index]
  }
  for index in 0..<scope.length() {
    scope_store[index] = scope[index]
  }
  {
    limits,
    budget,
    storage: FixedArray::make(limits.max_frame_bytes(), 0),
    content_digest: content_store,
    plan_digest: plan_store,
    cache_scope: scope_store,
    cache_scope_length: scope.length(),
    epoch: 0UL,
    phase: LUNA_TEXT_IDLE,
    input_length: 0,
    input_cursor: 0,
    stop_token_count: 0,
    stop_token_cursor: 0,
    stop_string_count: 0,
    stop_string_cursor: 0,
    stop_string_bytes: 0,
    stop_string_remaining: 0,
    payload_cursor: REQUEST_HEADER_BYTES,
    header_cursor: 0,
    scope_cursor: 0,
    checksum_cursor: 0,
    checksum: 2166136261,
    checksum_write_cursor: 0,
    protocol_version: 0,
    request_id: 0UL,
    max_new_tokens: 0,
    context_ceiling: 0,
    sampling_mode: 0,
    top_k: 0,
    temperature_bits: 0UL,
    top_p_bits: 0UL,
    sampling_seed: 0UL,
    stream: false,
    deadline_millis: 0UL,
    last_work_units: 0,
    total_work_units: 0UL,
  }
}

///|
fn LunaFramedTextRequestWorkspace::reset(self : Self) -> Unit {
  self.input_length = 0
  self.input_cursor = 0
  self.stop_token_count = 0
  self.stop_token_cursor = 0
  self.stop_string_count = 0
  self.stop_string_cursor = 0
  self.stop_string_bytes = 0
  self.stop_string_remaining = 0
  self.payload_cursor = REQUEST_HEADER_BYTES
  self.header_cursor = 0
  self.scope_cursor = 0
  self.checksum_cursor = 0
  self.checksum = 2166136261
  self.checksum_write_cursor = 0
  self.last_work_units = 0
  self.total_work_units = 0UL
}
