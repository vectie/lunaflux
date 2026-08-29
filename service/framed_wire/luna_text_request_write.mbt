///|
pub fn LunaFramedTextRequestWorkspace::begin_text(
  self : Self,
  byte_count~ : Int,
  protocol_version~ : @inference.ProtocolVersion,
  request_id~ : @inference.RequestId,
  max_new_tokens~ : Int,
  context_ceiling~ : Int,
  sampling~ : @inference.SamplingParameters,
  sampling_seed~ : @inference.SamplingSeed,
  stream_preference~ : @inference.StreamPreference,
  deadline~ : @inference.DeadlineBudget,
) -> LunaFramedTextRequestWrite raise FramedWireError {
  self.begin_text_with_sampling_scalars(
    byte_count~,
    protocol_version~,
    request_id~,
    max_new_tokens~,
    context_ceiling~,
    sampling_mode=sampling.mode(),
    temperature=sampling.temperature(),
    has_top_k=sampling.top_k() is Some(_),
    top_k=sampling.top_k().unwrap_or(0),
    has_top_p=sampling.top_p() is Some(_),
    top_p=sampling.top_p().unwrap_or(0.0),
    sampling_seed_value=sampling_seed.value(),
    stop_token_count=0,
    stop_string_count=0,
    stream_preference~,
    deadline~,
  )
}

///|
/// Begins one canonical text frame from already validated scalar sampling
/// fields and declared stop counts. The caller streams input and stop payload
/// into the returned writer; no request object or payload collection is
/// retained.
pub fn LunaFramedTextRequestWorkspace::begin_text_with_sampling_scalars(
  self : Self,
  byte_count~ : Int,
  protocol_version~ : @inference.ProtocolVersion,
  request_id~ : @inference.RequestId,
  max_new_tokens~ : Int,
  context_ceiling~ : Int,
  sampling_mode~ : @inference.SamplingMode,
  temperature~ : Double,
  has_top_k~ : Bool,
  top_k~ : Int,
  has_top_p~ : Bool,
  top_p~ : Double,
  sampling_seed_value~ : UInt64,
  stop_token_count~ : Int,
  stop_string_count~ : Int,
  stream_preference~ : @inference.StreamPreference,
  deadline~ : @inference.DeadlineBudget,
) -> LunaFramedTextRequestWrite raise FramedWireError {
  if self.phase != LUNA_TEXT_IDLE {
    raise LunaFramedTextRequestBusy
  }
  let inference = self.limits.inference_limits()
  let minimum_stop_bytes = if stop_token_count < 0 || stop_string_count < 0 {
    -1
  } else if stop_token_count > inference.max_stop_token_ids() ||
    stop_string_count > inference.max_stop_strings() ||
    stop_token_count > 0x7fffffff / 4 ||
    stop_string_count > 0x7fffffff / 4 ||
    stop_token_count * 4 > 0x7fffffff - stop_string_count * 4 {
    -1
  } else {
    stop_token_count * 4 + stop_string_count * 4
  }
  if byte_count <= 0 ||
    byte_count > inference.max_text_bytes() ||
    minimum_stop_bytes < 0 ||
    byte_count >
    self.limits.max_frame_bytes() -
    REQUEST_HEADER_BYTES -
    self.cache_scope_length -
    minimum_stop_bytes ||
    protocol_version.wire_code() != 1 ||
    request_id.value() == 0UL ||
    max_new_tokens <= 0 ||
    max_new_tokens > inference.max_new_tokens() ||
    context_ceiling <= 0 ||
    context_ceiling > inference.max_context_tokens() ||
    max_new_tokens >= context_ceiling ||
    sampling_seed_value == 0UL ||
    deadline.milliseconds() == 0UL ||
    deadline.milliseconds() > inference.max_deadline_millis() {
    raise InvalidFrame(frame=RequestFrame, rule=Scalar)
  }
  let (mode, effective_temperature, effective_top_k, effective_top_p) = match
    sampling_mode {
    Greedy => {
      if temperature != 1.0 ||
        has_top_k ||
        top_k != 0 ||
        has_top_p ||
        top_p != 0.0 {
        raise InvalidFrame(frame=RequestFrame, rule=Scalar)
      }
      (0, 1.0, 0, 0.0)
    }
    Sample => {
      if temperature.is_nan() ||
        temperature.is_inf() ||
        temperature <= 0.0 ||
        temperature > inference.max_temperature() ||
        has_top_k != (top_k != 0) ||
        (has_top_k && (top_k <= 0 || top_k > inference.max_top_k())) ||
        has_top_p != (top_p != 0.0) ||
        (
          has_top_p &&
          (top_p.is_nan() || top_p.is_inf() || top_p <= 0.0 || top_p > 1.0)
        ) {
        raise InvalidFrame(frame=RequestFrame, rule=Scalar)
      }
      (1, temperature, top_k, top_p)
    }
  }
  if self.epoch == 0xffffffffffffffffUL {
    raise LunaFramedTextRequestStale
  }
  self.reset()
  self.epoch += 1UL
  self.input_length = byte_count
  self.stop_token_count = stop_token_count
  self.stop_string_count = stop_string_count
  self.payload_cursor = REQUEST_HEADER_BYTES + byte_count + stop_token_count * 4
  self.protocol_version = protocol_version.wire_code()
  self.request_id = request_id.value()
  self.max_new_tokens = max_new_tokens
  self.context_ceiling = context_ceiling
  self.sampling_mode = mode
  self.temperature_bits = effective_temperature.reinterpret_as_uint64()
  self.top_k = effective_top_k
  self.top_p_bits = effective_top_p.reinterpret_as_uint64()
  self.sampling_seed = sampling_seed_value
  self.stream = stream_preference is Streaming
  self.deadline_millis = deadline.milliseconds()
  self.phase = LUNA_TEXT_WRITING
  { workspace: self, epoch: self.epoch }
}

///|
fn LunaFramedTextRequestWrite::require(
  self : Self,
) -> LunaFramedTextRequestWorkspace raise FramedWireError {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase != LUNA_TEXT_WRITING {
    raise LunaFramedTextRequestStale
  }
  self.workspace
}

///|
pub fn LunaFramedTextRequestWrite::push_byte(
  self : Self,
  byte : Byte,
) -> Unit raise FramedWireError {
  let workspace = self.require()
  if workspace.input_cursor >= workspace.input_length {
    raise LunaFramedTextRequestInvalidRange
  }
  workspace.storage[REQUEST_HEADER_BYTES + workspace.input_cursor] = byte
  workspace.input_cursor += 1
}

///|
pub fn LunaFramedTextRequestWrite::finish(
  self : Self,
) -> LunaFramedTextRequestWork raise FramedWireError {
  let workspace = self.require()
  if workspace.input_cursor != workspace.input_length ||
    workspace.stop_token_cursor != workspace.stop_token_count ||
    workspace.stop_string_cursor != workspace.stop_string_count ||
    workspace.stop_string_remaining != 0 {
    raise LunaFramedTextRequestNotReady
  }
  workspace.phase = LUNA_TEXT_HEADER
  { workspace, epoch: self.epoch }
}

///|
pub fn LunaFramedTextRequestWrite::abort(
  self : Self,
) -> Unit raise FramedWireError {
  let workspace = self.require()
  workspace.reset()
  workspace.phase = LUNA_TEXT_IDLE
}
