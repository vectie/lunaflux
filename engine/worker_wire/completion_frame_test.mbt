///|
test "completion frame round trips exact scalar outcomes" {
  let plan = submitted_wire_fixture()
  let completion = submitted_completion_fixture(plan)
  let sender = @worker_wire.CompletionFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let encoded = sender.encode_completion(completion)
  encoded.validate_for(plan)
  assert_eq(encoded.length(), 128)
  assert_eq(encoded.sequence_value(), 1UL)
  assert_eq(encoded.model_generation_value(), 9UL)
  assert_eq(encoded.entry_count(), 2)
  let prefill = encoded.entry_for_slot(0)
  assert_eq(prefill.kind_value(), 1)
  assert_eq(prefill.request_id_value(), 7UL)
  assert_eq(prefill.request_generation_value(), 0UL)
  assert_eq(prefill.processed_tokens(), 2)
  assert_eq(prefill.token_id(), 17)
  assert_eq(prefill.failure_value(), 0)
  let decode = encoded.entry(1)
  assert_eq(decode.slot(), 1)
  assert_eq(decode.kind_value(), 2)
  assert_eq(decode.request_id_value(), 8UL)
  assert_eq(decode.request_generation_value(), 3UL)
  assert_eq(decode.processed_tokens(), 0)
  assert_eq(decode.token_id(), 18)
  let transport = FixedArray::make(encoded.length(), b'\x00')
  encoded.copy_to(transport, destination_offset=0)
  let receiver = @worker_wire.CompletionFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let received = receiver.load(transport, transport.length())
  assert_eq(received.entry_for_slot(1).token_id(), 18)
}

///|
test "completion frame authenticates worker failure against any exact row" {
  let plan = submitted_wire_fixture()
  let completion_owner = @worker_protocol.CompletionBuffer::new(
    wire_protocol_limits(),
  )
  let prefill = plan.prefill_row(0)
  let decode = plan.decode_row(0)
  completion_owner.append_failed(
    slot=prefill.completion_slot(),
    request_id=prefill.request_id(),
    request_generation=prefill.request_generation(),
    failure=ExecutionFailed,
  )
  completion_owner.append_failed(
    slot=decode.completion_slot(),
    request_id=decode.request_id(),
    request_generation=decode.request_generation(),
    failure=ResourceUnavailable,
  )
  let frame_owner = @worker_wire.CompletionFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  frame_owner
  .encode_completion(completion_owner.submit_for(plan))
  .validate_for(plan)
}

///|
test "invalid completion transport preserves prior frame" {
  let plan = submitted_wire_fixture()
  let sender = @worker_wire.CompletionFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let encoded = sender.encode_completion(submitted_completion_fixture(plan))
  let transport = FixedArray::make(encoded.length(), b'\x00')
  encoded.copy_to(transport, destination_offset=0)
  let receiver = @worker_wire.CompletionFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let prior = receiver.load(transport, transport.length())
  transport[0] = transport[0] ^ b'\x01'
  try receiver.load(transport, transport.length()) catch {
    InvalidFrame(Magic) => ()
    _ => fail("expected magic rejection")
  } noraise {
    _ => fail("expected invalid frame")
  }
  assert_eq(prior.entry_for_slot(0).token_id(), 17)
}

///|
test "completion re-encode invalidates old frame and row views" {
  let plan = submitted_wire_fixture()
  let owner = @worker_wire.CompletionFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let first = owner.encode_completion(submitted_completion_fixture(plan))
  let row = first.entry(0)
  ignore(owner.encode_completion(submitted_completion_fixture(plan)))
  try row.slot() catch {
    StaleFrame => ()
    _ => fail("expected stale row")
  } noraise {
    _ => fail("expected stale row")
  }
}

///|
test "canonical completion encoding is byte-for-byte deterministic" {
  let plan = submitted_wire_fixture()
  let left_owner = @worker_wire.CompletionFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let right_owner = @worker_wire.CompletionFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let left = left_owner.encode_completion(submitted_completion_fixture(plan))
  let right = right_owner.encode_completion(submitted_completion_fixture(plan))
  let left_bytes = FixedArray::make(left.length(), b'\x00')
  let right_bytes = FixedArray::make(right.length(), b'\x00')
  left.copy_to(left_bytes, destination_offset=0)
  right.copy_to(right_bytes, destination_offset=0)
  for index in 0..<left.length() {
    assert_eq(left_bytes[index], right_bytes[index])
  }
}
