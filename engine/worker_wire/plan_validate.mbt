///|
fn validate_wire_sampling(
  bytes : FixedArray[Byte],
  offset : Int,
  limits : @inference.InferenceLimits,
) -> Unit raise WorkerWireError {
  let mode = read_u32(bytes, offset)
  let temperature = read_u64(bytes, offset + 4).reinterpret_as_double()
  let has_top_k = read_u32(bytes, offset + 12)
  let top_k = read_i32(bytes, offset + 16)
  let has_top_p = read_u32(bytes, offset + 20)
  let top_p_bits = read_u64(bytes, offset + 24)
  if has_top_k > 1 || has_top_p > 1 {
    raise InvalidFrame(Sampling)
  }
  match mode {
    0 =>
      if temperature != 1.0 ||
        has_top_k != 0 ||
        top_k != 0 ||
        has_top_p != 0 ||
        top_p_bits != 0UL {
        raise InvalidFrame(Sampling)
      }
    1 => {
      if temperature.is_nan() ||
        temperature.is_inf() ||
        temperature <= 0.0 ||
        temperature > limits.max_temperature() {
        raise InvalidFrame(Sampling)
      }
      if (has_top_k == 0 && top_k != 0) ||
        (has_top_k == 1 && (top_k <= 0 || top_k > limits.max_top_k())) {
        raise InvalidFrame(Sampling)
      }
      if has_top_p == 0 {
        if top_p_bits != 0UL {
          raise InvalidFrame(Sampling)
        }
      } else {
        let top_p = top_p_bits.reinterpret_as_double()
        if top_p.is_nan() || top_p.is_inf() || top_p <= 0.0 || top_p > 1.0 {
          raise InvalidFrame(Sampling)
        }
      }
    }
    _ => raise InvalidFrame(Sampling)
  }
}

///|
fn validate_wire_range(
  start : Int,
  length : Int,
  table_length : Int,
  expected_start : Int,
  rule : WorkerWireFrameRule,
) -> Int raise WorkerWireError {
  if start != expected_start ||
    length <= 0 ||
    start < 0 ||
    start > table_length ||
    length > table_length - start {
    raise InvalidFrame(rule)
  }
  start + length
}

///|
fn validate_unique_wire_rows(
  bytes : FixedArray[Byte],
  prefill_offset : Int,
  prefill_count : Int,
  decode_offset : Int,
  decode_count : Int,
) -> Unit raise WorkerWireError {
  for left in 0..<prefill_count {
    let left_offset = prefill_offset + left * prefill_row_bytes
    let left_request = read_u64(bytes, left_offset)
    let left_slot = read_i32(bytes, left_offset + 92)
    for right in 0..<left {
      let right_offset = prefill_offset + right * prefill_row_bytes
      if left_request == read_u64(bytes, right_offset) {
        raise InvalidFrame(RequestIdentity)
      }
      if left_slot == read_i32(bytes, right_offset + 92) {
        raise InvalidFrame(CompletionSlot)
      }
    }
    for right in 0..<decode_count {
      let right_offset = decode_offset + right * decode_row_bytes
      if left_request == read_u64(bytes, right_offset) {
        raise InvalidFrame(RequestIdentity)
      }
      if left_slot == read_i32(bytes, right_offset + 84) {
        raise InvalidFrame(CompletionSlot)
      }
    }
  }
  for left in 0..<decode_count {
    let left_offset = decode_offset + left * decode_row_bytes
    let left_request = read_u64(bytes, left_offset)
    let left_slot = read_i32(bytes, left_offset + 84)
    for right in 0..<left {
      let right_offset = decode_offset + right * decode_row_bytes
      if left_request == read_u64(bytes, right_offset) {
        raise InvalidFrame(RequestIdentity)
      }
      if left_slot == read_i32(bytes, right_offset + 84) {
        raise InvalidFrame(CompletionSlot)
      }
    }
  }
}

///|
fn validate_plan_frame_source(
  bytes : FixedArray[Byte],
  supplied_length : Int,
  worker_limits : @worker_protocol.WorkerProtocolLimits,
  inference_limits : @inference.InferenceLimits,
  destination_capacity : Int,
) -> Unit raise WorkerWireError {
  if supplied_length < plan_header_bytes || supplied_length > bytes.length() {
    raise SourceTooSmall
  }
  if supplied_length > destination_capacity {
    raise InvalidLimit(PlanFrameBytes)
  }
  if read_u32(bytes, 0) != plan_magic {
    raise InvalidFrame(Magic)
  }
  if read_u32(bytes, 4) != wire_version {
    raise InvalidFrame(Version)
  }
  if read_u32(bytes, 8) != plan_kind {
    raise InvalidFrame(Kind)
  }
  if read_i32(bytes, 12) != supplied_length {
    raise InvalidFrame(Length)
  }
  if read_u32(bytes, 56) != 0 {
    raise InvalidFrame(Reserved)
  }
  if frame_checksum(bytes, supplied_length) !=
    read_u32(bytes, plan_checksum_offset) {
    raise InvalidFrame(Checksum)
  }
  let sequence = read_u64(bytes, 16)
  let generation = read_u64(bytes, 24)
  let token_budget = read_i32(bytes, 32)
  let token_count = read_i32(bytes, 36)
  let page_count = read_i32(bytes, 40)
  let capability_count = read_i32(bytes, 44)
  let prefill_count = read_i32(bytes, 48)
  let decode_count = read_i32(bytes, 52)
  if sequence == 0UL ||
    generation == 0UL ||
    token_budget <= 0 ||
    token_count < 0 ||
    page_count < 0 ||
    capability_count < 0 ||
    prefill_count < 0 ||
    decode_count < 0 ||
    token_budget > worker_limits.max_plan_tokens() ||
    token_count > worker_limits.max_plan_tokens() ||
    page_count > worker_limits.max_plan_pages() ||
    capability_count > worker_limits.max_capabilities() ||
    prefill_count > worker_limits.max_prefill_rows() ||
    decode_count > worker_limits.max_decode_rows() ||
    prefill_count + decode_count <= 0 ||
    prefill_count > worker_limits.max_plan_rows() - decode_count {
    raise InvalidFrame(Counts)
  }
  let expected_length = exact_plan_frame_bytes(
    token_count, page_count, capability_count, prefill_count, decode_count,
  )
  if expected_length != supplied_length {
    raise InvalidFrame(Length)
  }
  let token_offset = token_table_offset()
  let max_token_id = if worker_limits.max_token_id() <
    inference_limits.max_token_id() {
    worker_limits.max_token_id()
  } else {
    inference_limits.max_token_id()
  }
  for index in 0..<token_count {
    let token = read_i32(bytes, token_offset + index * 4)
    if token < 0 || token > max_token_id {
      raise InvalidFrame(TokenTable)
    }
  }
  let page_offset = page_table_offset(token_count)
  for index in 0..<page_count {
    if read_i32(bytes, page_offset + index * 8) < 0 ||
      read_i32(bytes, page_offset + index * 8 + 4) <= 0 {
      raise InvalidFrame(PageTable)
    }
  }
  let capability_offset = capability_table_offset(token_count, page_count)
  for index in 0..<capability_count {
    if read_i32(bytes, capability_offset + index * 4) <= 0 {
      raise InvalidFrame(CapabilityTable)
    }
  }
  let prefill_offset = prefill_table_offset(
    token_count, page_count, capability_count,
  )
  let decode_offset = decode_table_offset(
    token_count, page_count, capability_count, prefill_count,
  )
  let mut token_cursor = 0
  let mut page_cursor = 0
  let mut capability_cursor = 0
  let mut scheduled_tokens = 0
  for row in 0..<prefill_count {
    let offset = prefill_offset + row * prefill_row_bytes
    let sequence_start = read_i32(bytes, offset + 16)
    let token_length = read_i32(bytes, offset + 24)
    if read_u64(bytes, offset) == 0UL || read_u64(bytes, offset + 8) == 0UL {
      raise InvalidFrame(RequestIdentity)
    }
    token_cursor = validate_wire_range(
      read_i32(bytes, offset + 20),
      token_length,
      token_count,
      token_cursor,
      PrefillRow,
    )
    page_cursor = validate_wire_range(
      read_i32(bytes, offset + 28),
      read_i32(bytes, offset + 32),
      page_count,
      page_cursor,
      PrefillRow,
    )
    capability_cursor = validate_wire_range(
      read_i32(bytes, offset + 36),
      read_i32(bytes, offset + 40),
      capability_count,
      capability_cursor,
      PrefillRow,
    )
    if sequence_start < 0 ||
      sequence_start > worker_limits.max_sequence_tokens() ||
      token_length > worker_limits.max_sequence_tokens() - sequence_start {
      raise InvalidFrame(PrefillRow)
    }
    let produces = read_u32(bytes, offset + 44)
    if produces > 1 {
      raise InvalidFrame(PrefillRow)
    }
    validate_wire_sampling(bytes, offset + 48, inference_limits)
    let seed = read_u64(bytes, offset + 80)
    let sample_index = read_i32(bytes, offset + 88)
    if produces == 1 {
      if seed == 0UL ||
        sample_index < 0 ||
        sample_index >= worker_limits.max_sequence_tokens() {
        raise InvalidFrame(PrefillRow)
      }
    } else if seed != 1UL || sample_index != 0 {
      raise InvalidFrame(PrefillRow)
    }
    let slot = read_i32(bytes, offset + 92)
    if slot < 0 || slot >= worker_limits.max_completion_slots() {
      raise InvalidFrame(CompletionSlot)
    }
    scheduled_tokens = scheduled_tokens + token_length
  }
  for row in 0..<decode_count {
    let offset = decode_offset + row * decode_row_bytes
    if read_u64(bytes, offset) == 0UL || read_u64(bytes, offset + 8) == 0UL {
      raise InvalidFrame(RequestIdentity)
    }
    let position = read_i32(bytes, offset + 16)
    if position < 0 || position >= worker_limits.max_sequence_tokens() {
      raise InvalidFrame(DecodeRow)
    }
    token_cursor = validate_wire_range(
      read_i32(bytes, offset + 20),
      1,
      token_count,
      token_cursor,
      DecodeRow,
    )
    page_cursor = validate_wire_range(
      read_i32(bytes, offset + 24),
      read_i32(bytes, offset + 28),
      page_count,
      page_cursor,
      DecodeRow,
    )
    capability_cursor = validate_wire_range(
      read_i32(bytes, offset + 32),
      read_i32(bytes, offset + 36),
      capability_count,
      capability_cursor,
      DecodeRow,
    )
    validate_wire_sampling(bytes, offset + 40, inference_limits)
    if read_u64(bytes, offset + 72) == 0UL ||
      read_i32(bytes, offset + 80) < 0 ||
      read_i32(bytes, offset + 80) >= worker_limits.max_sequence_tokens() {
      raise InvalidFrame(DecodeRow)
    }
    let slot = read_i32(bytes, offset + 84)
    if slot < 0 || slot >= worker_limits.max_completion_slots() {
      raise InvalidFrame(CompletionSlot)
    }
    scheduled_tokens = scheduled_tokens + 1
  }
  if token_cursor != token_count ||
    page_cursor != page_count ||
    capability_cursor != capability_count ||
    scheduled_tokens != token_budget {
    raise InvalidFrame(Counts)
  }
  validate_unique_wire_rows(
    bytes, prefill_offset, prefill_count, decode_offset, decode_count,
  )
}

///|
/// Validates an untrusted fixed-buffer frame before replacing the current view.
/// Any rejection leaves the prior frame and epoch unchanged.
pub fn PlanFrameBuffer::load(
  self : PlanFrameBuffer,
  source : FixedArray[Byte],
  length : Int,
) -> ValidatedPlanFrame raise WorkerWireError {
  if self.epoch == 0xffffffffffffffffUL {
    raise FrameEpochOverflow
  }
  validate_plan_frame_source(
    source,
    length,
    self.worker_limits,
    self.inference_limits,
    self.storage.length(),
  )
  for index in 0..<length {
    self.storage[index] = source[index]
  }
  self.epoch += 1UL
  self.length = length
  ValidatedPlanFrame(self, self.epoch)
}

///|
/// Copies the authenticated frame to caller-owned transport storage.
pub fn ValidatedPlanFrame::copy_to(
  self : ValidatedPlanFrame,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Unit raise WorkerWireError {
  let owner = self.owner()
  if destination_offset < 0 ||
    destination_offset > destination.length() ||
    owner.length > destination.length() - destination_offset {
    raise DestinationTooSmall
  }
  for index in 0..<owner.length {
    destination[destination_offset + index] = owner.storage[index]
  }
}
