///|
fn write_valid_decode_row(
  bytes : FixedArray[Byte],
  offset : Int,
  request : UInt64,
  token_start : Int,
  page_start : Int,
  capability_start : Int,
  slot : Int,
) -> Unit {
  write_u64(bytes, offset, request)
  write_u64(bytes, offset + 8, request)
  write_i32(bytes, offset + 16, token_start)
  write_i32(bytes, offset + 20, token_start)
  write_i32(bytes, offset + 24, page_start)
  write_i32(bytes, offset + 28, 1)
  write_i32(bytes, offset + 32, capability_start)
  write_i32(bytes, offset + 36, 1)
  write_u32(bytes, offset + 40, 0)
  write_u64(bytes, offset + 44, 1.0.reinterpret_as_uint64())
  write_u32(bytes, offset + 52, 0)
  write_i32(bytes, offset + 56, 0)
  write_u32(bytes, offset + 60, 0)
  write_u64(bytes, offset + 64, 0UL)
  write_u64(bytes, offset + 72, request + 10UL)
  write_i32(bytes, offset + 80, token_start)
  write_i32(bytes, offset + 84, slot)
}

///|
fn valid_two_decode_frame() -> FixedArray[Byte] {
  let length = 272
  let bytes = FixedArray::make(length, b'\x00')
  write_plan_header(bytes, length, 1UL, 1UL, 2, 2, 2, 2, 0, 2)
  write_i32(bytes, 64, 10)
  write_i32(bytes, 68, 11)
  write_i32(bytes, 72, 0)
  write_i32(bytes, 76, 1)
  write_i32(bytes, 80, 1)
  write_i32(bytes, 84, 1)
  write_i32(bytes, 88, 1)
  write_i32(bytes, 92, 2)
  write_valid_decode_row(bytes, 96, 7UL, 0, 0, 0, 0)
  write_valid_decode_row(bytes, 184, 8UL, 1, 1, 1, 1)
  write_u32(bytes, plan_checksum_offset, frame_checksum(bytes, length))
  bytes
}

///|
fn refresh_wire_checksum(bytes : FixedArray[Byte]) -> Unit {
  write_u32(bytes, plan_checksum_offset, frame_checksum(bytes, bytes.length()))
}

///|
test "semantic table and row corruption is rejected after valid checksum" {
  let limits = try! @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=2,
    max_decode_rows=2,
    max_plan_rows=2,
    max_plan_tokens=4,
    max_plan_pages=4,
    max_capabilities=4,
    max_completion_slots=4,
    max_sequence_tokens=16,
    max_token_id=32,
  )
  let inference = try! @inference.InferenceLimits::new(
    max_text_bytes=16,
    max_input_tokens=16,
    max_new_tokens=16,
    max_context_tokens=16,
    max_token_id=32,
    max_stop_token_ids=1,
    max_stop_strings=1,
    max_stop_string_bytes=1,
    max_trace_bytes=1,
    max_cache_scope_bytes=1,
    max_decoded_delta_bytes=1,
    max_deadline_millis=1UL,
    max_top_k=32,
    max_temperature=4.0,
  )
  let owner = PlanFrameBuffer::new(limits, inference)
  let valid = valid_two_decode_frame()
  let prior = owner.load(valid, valid.length())
  let invalid_token = valid_two_decode_frame()
  write_i32(invalid_token, 64, 33)
  refresh_wire_checksum(invalid_token)
  try owner.load(invalid_token, invalid_token.length()) catch {
    InvalidFrame(TokenTable) => ()
    _ => fail("expected token-table rejection")
  } noraise {
    _ => fail("expected token-table rejection")
  }
  assert_eq(prior.token_at(1), 11)
  let invalid_page = valid_two_decode_frame()
  write_i32(invalid_page, 76, 0)
  refresh_wire_checksum(invalid_page)
  try owner.load(invalid_page, invalid_page.length()) catch {
    InvalidFrame(PageTable) => ()
    _ => fail("expected page-table rejection")
  } noraise {
    _ => fail("expected page-table rejection")
  }
  let invalid_capability = valid_two_decode_frame()
  write_i32(invalid_capability, 88, 0)
  refresh_wire_checksum(invalid_capability)
  try owner.load(invalid_capability, invalid_capability.length()) catch {
    InvalidFrame(CapabilityTable) => ()
    _ => fail("expected capability rejection")
  } noraise {
    _ => fail("expected capability rejection")
  }
  let invalid_range = valid_two_decode_frame()
  write_i32(invalid_range, 184 + 20, 0)
  refresh_wire_checksum(invalid_range)
  try owner.load(invalid_range, invalid_range.length()) catch {
    InvalidFrame(DecodeRow) => ()
    _ => fail("expected row-range rejection")
  } noraise {
    _ => fail("expected row-range rejection")
  }
}

///|
test "duplicate request slot and malformed sampling are rejected" {
  let owner = PlanFrameBuffer::new(
    try! @worker_protocol.WorkerProtocolLimits::new(
      max_prefill_rows=2,
      max_decode_rows=2,
      max_plan_rows=2,
      max_plan_tokens=4,
      max_plan_pages=4,
      max_capabilities=4,
      max_completion_slots=4,
      max_sequence_tokens=16,
      max_token_id=32,
    ),
    try! @inference.InferenceLimits::new(
      max_text_bytes=16,
      max_input_tokens=16,
      max_new_tokens=16,
      max_context_tokens=16,
      max_token_id=32,
      max_stop_token_ids=1,
      max_stop_strings=1,
      max_stop_string_bytes=1,
      max_trace_bytes=1,
      max_cache_scope_bytes=1,
      max_decoded_delta_bytes=1,
      max_deadline_millis=1UL,
      max_top_k=32,
      max_temperature=4.0,
    ),
  )
  let duplicate_request = valid_two_decode_frame()
  write_u64(duplicate_request, 184, 7UL)
  refresh_wire_checksum(duplicate_request)
  try owner.load(duplicate_request, duplicate_request.length()) catch {
    InvalidFrame(RequestIdentity) => ()
    _ => fail("expected request rejection")
  } noraise {
    _ => fail("expected request rejection")
  }
  let duplicate_slot = valid_two_decode_frame()
  write_i32(duplicate_slot, 184 + 84, 0)
  refresh_wire_checksum(duplicate_slot)
  try owner.load(duplicate_slot, duplicate_slot.length()) catch {
    InvalidFrame(CompletionSlot) => ()
    _ => fail("expected slot rejection")
  } noraise {
    _ => fail("expected slot rejection")
  }
  let malformed_sampling = valid_two_decode_frame()
  write_u32(malformed_sampling, 96 + 52, 1)
  write_i32(malformed_sampling, 96 + 56, 0)
  refresh_wire_checksum(malformed_sampling)
  try owner.load(malformed_sampling, malformed_sampling.length()) catch {
    InvalidFrame(Sampling) => ()
    _ => fail("expected sampling rejection")
  } noraise {
    _ => fail("expected sampling rejection")
  }
}
