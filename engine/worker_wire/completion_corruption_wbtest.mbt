///|
fn valid_completion_frame() -> FixedArray[Byte] {
  let length = 128
  let bytes = FixedArray::make(length, b'\x00')
  write_completion_header(bytes, length, 1UL, 2UL, 2)
  let first = completion_entry_offset(0)
  write_i32(bytes, first, 0)
  write_u32(bytes, first + 4, 0)
  write_u64(bytes, first + 8, 7UL)
  write_u64(bytes, first + 16, 0UL)
  write_i32(bytes, first + 24, 2)
  let second = completion_entry_offset(1)
  write_i32(bytes, second, 1)
  write_u32(bytes, second + 4, 3)
  write_u64(bytes, second + 8, 8UL)
  write_u64(bytes, second + 16, 4UL)
  write_u32(bytes, second + 32, 1)
  write_u32(
    bytes,
    completion_checksum_offset,
    completion_frame_checksum(bytes, length),
  )
  bytes
}

///|
fn refresh_completion_checksum(bytes : FixedArray[Byte]) -> Unit {
  write_u32(
    bytes,
    completion_checksum_offset,
    completion_frame_checksum(bytes, bytes.length()),
  )
}

///|
fn completion_test_owner() -> CompletionFrameBuffer {
  try! CompletionFrameBuffer::new(
    @worker_protocol.WorkerProtocolLimits::new(
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
    @inference.InferenceLimits::new(
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
}

///|
test "checksum-valid completion payload corruption is rejected" {
  let owner = completion_test_owner()
  let invalid_kind = valid_completion_frame()
  write_u32(invalid_kind, completion_entry_offset(0) + 4, 9)
  refresh_completion_checksum(invalid_kind)
  try owner.load(invalid_kind, invalid_kind.length()) catch {
    InvalidFrame(CompletionEntry) => ()
    _ => fail("expected kind rejection")
  } noraise {
    _ => fail("expected kind rejection")
  }
  let invalid_payload = valid_completion_frame()
  write_i32(invalid_payload, completion_entry_offset(0) + 28, 3)
  refresh_completion_checksum(invalid_payload)
  try owner.load(invalid_payload, invalid_payload.length()) catch {
    InvalidFrame(CompletionEntry) => ()
    _ => fail("expected payload rejection")
  } noraise {
    _ => fail("expected payload rejection")
  }
  let invalid_failure = valid_completion_frame()
  write_u32(invalid_failure, completion_entry_offset(1) + 32, 3)
  refresh_completion_checksum(invalid_failure)
  try owner.load(invalid_failure, invalid_failure.length()) catch {
    InvalidFrame(CompletionEntry) => ()
    _ => fail("expected failure rejection")
  } noraise {
    _ => fail("expected failure rejection")
  }
}

///|
test "duplicate completion identities are rejected" {
  let owner = completion_test_owner()
  let duplicate_slot = valid_completion_frame()
  write_i32(duplicate_slot, completion_entry_offset(1), 0)
  refresh_completion_checksum(duplicate_slot)
  try owner.load(duplicate_slot, duplicate_slot.length()) catch {
    InvalidFrame(CompletionSlot) => ()
    _ => fail("expected slot rejection")
  } noraise {
    _ => fail("expected slot rejection")
  }
  let duplicate_request = valid_completion_frame()
  write_u64(duplicate_request, completion_entry_offset(1) + 8, 7UL)
  refresh_completion_checksum(duplicate_request)
  try owner.load(duplicate_request, duplicate_request.length()) catch {
    InvalidFrame(RequestIdentity) => ()
    _ => fail("expected request rejection")
  } noraise {
    _ => fail("expected request rejection")
  }
}
