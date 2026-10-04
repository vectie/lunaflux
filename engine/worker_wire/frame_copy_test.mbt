///|
test "plan bulk copy preserves unaligned transport envelope and validates before commit" {
  let sender = @worker_wire.PlanFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let frame = sender.encode_plan(submitted_wire_fixture())
  let length = frame.length()
  let transport = FixedArray::make(length + 10, b'\xa5')
  frame.copy_to(transport, destination_offset=3)
  for index in 0..<3 {
    assert_eq(transport[index], b'\xa5')
  }
  for index in (length + 3)..<transport.length() {
    assert_eq(transport[index], b'\xa5')
  }
  let bytes = FixedArray::make(length, b'\x00')
  FixedArray::unsafe_blit(bytes, 0, transport, 3, length)
  let receiver = @worker_wire.PlanFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let received = receiver.load(bytes, length)
  assert_eq(received.token_at(2), 12)
  let restored = FixedArray::make(length, b'\x00')
  received.copy_to(restored, destination_offset=0)
  assert_eq(restored, bytes)
  for offset in [-1, 11] {
    try frame.copy_to(transport, destination_offset=offset) catch {
      DestinationTooSmall => ()
      _ => fail("unexpected bounds error")
    } noraise {
      _ => fail("invalid offset accepted")
    }
  }
  bytes[0] = b'\x00'
  try receiver.load(bytes, length) catch {
    InvalidFrame(Magic) => ()
    _ => fail("unexpected validation error")
  } noraise {
    _ => fail("invalid plan accepted")
  }
  assert_eq(received.token_at(2), 12)
}

///|
test "completion bulk copy preserves envelope and prior valid epoch on corruption" {
  let submitted = submitted_wire_fixture()
  let sender = @worker_wire.CompletionFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let frame = sender.encode_completion(submitted_completion_fixture(submitted))
  let length = frame.length()
  let transport = FixedArray::make(length + 10, b'\xa5')
  frame.copy_to(transport, destination_offset=3)
  for index in 0..<3 {
    assert_eq(transport[index], b'\xa5')
  }
  for index in (length + 3)..<transport.length() {
    assert_eq(transport[index], b'\xa5')
  }
  let bytes = FixedArray::make(length, b'\x00')
  FixedArray::unsafe_blit(bytes, 0, transport, 3, length)
  let receiver = @worker_wire.CompletionFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let received = receiver.load(bytes, length)
  assert_eq(received.entry_for_slot(1).token_id(), 18)
  let restored = FixedArray::make(length, b'\x00')
  received.copy_to(restored, destination_offset=0)
  assert_eq(restored, bytes)
  for offset in [-1, 11] {
    try frame.copy_to(transport, destination_offset=offset) catch {
      DestinationTooSmall => ()
      _ => fail("unexpected bounds error")
    } noraise {
      _ => fail("invalid offset accepted")
    }
  }
  bytes[0] = b'\x00'
  try receiver.load(bytes, length) catch {
    InvalidFrame(Magic) => ()
    _ => fail("unexpected validation error")
  } noraise {
    _ => fail("invalid completion accepted")
  }
  assert_eq(received.entry_for_slot(1).token_id(), 18)
}
