///|
test "rank group wire canonical payload round trip and stale epoch" {
  let binding = fixture_wire_binding(0)
  let limits = fixture_wire_limits()
  let sender = @rank_group_wire.RankGroupWireFrameBuffer::new(limits)
  let receiver = @rank_group_wire.RankGroupWireFrameBuffer::new(limits)
  let payload = FixedArray::makei(4, index => (index + 1).to_byte())
  let frame = try! sender.encode_payload(
    binding,
    Submit,
    ParentToChild,
    A,
    1UL,
    payload,
    4,
  )
  let wire = FixedArray::make(sender.capacity(), b'\x00')
  let length = try! frame.length()
  try! frame.copy_to(wire, destination_offset=0)
  let decoded = try! receiver.load(wire, length)
  assert_eq(try! decoded.kind(), Submit)
  assert_eq(try! decoded.direction(), ParentToChild)
  assert_eq(try! decoded.slot(), Some(A))
  assert_eq(try! decoded.sequence_value(), 1UL)
  assert_eq(try! decoded.payload_length(), 4)
  assert_eq(try! decoded.binding().group_digest(), binding.group_digest())
  assert_eq(
    try! decoded.binding().execution_plan_digest(),
    binding.execution_plan_digest(),
  )
  let copied = FixedArray::make(4, b'\x00')
  try! decoded.copy_payload_to(copied, destination_offset=0)
  assert_eq(copied, payload)
  ignore(try! receiver.load(wire, length))
  try decoded.kind() catch {
    Invalid(FrameEpoch) => ()
    _ => fail("expected stale frame rejection")
  } noraise {
    _ => fail("expected stale frame rejection")
  }
}

///|
test "rank group wire rejects truncation trailing bytes and direction mismatch" {
  let binding = fixture_wire_binding(0)
  let limits = fixture_wire_limits()
  let sender = @rank_group_wire.RankGroupWireFrameBuffer::new(limits)
  let receiver = @rank_group_wire.RankGroupWireFrameBuffer::new(limits)
  let configure_payload = FixedArray::make(1, b'\x01')
  let frame = try! sender.encode_configure_payload(
    binding, configure_payload, 1,
  )
  let wire = FixedArray::make(sender.capacity(), b'\x00')
  let length = try! frame.length()
  try! frame.copy_to(wire, destination_offset=0)
  try receiver.load(wire, length - 1) catch {
    Invalid(Length) => ()
    _ => fail("expected truncated frame rejection")
  } noraise {
    _ => fail("expected truncated frame rejection")
  }
  try receiver.load(wire, length + 1) catch {
    Invalid(Length) => ()
    _ => fail("expected trailing frame rejection")
  } noraise {
    _ => fail("expected trailing frame rejection")
  }
  try
    @rank_group_wire.encode_rank_group_wire_payload_into(
      wire,
      limits,
      binding,
      Configure,
      ChildToParent,
      None,
      0UL,
      configure_payload,
      1,
    )
  catch {
    Invalid(Direction) => ()
    _ => fail("expected direction rejection")
  } noraise {
    _ => fail("expected direction rejection")
  }
}
