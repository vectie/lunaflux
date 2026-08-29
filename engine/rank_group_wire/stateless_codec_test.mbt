///|
test "rank group wire stateless Configure decode copies scalars and binding" {
  let limits = fixture_wire_limits()
  let binding = fixture_wire_binding(0)
  let payload = FixedArray::makei(3, index => (index + 1).to_byte())
  let wire = FixedArray::make(limits.max_frame_bytes(), b'\x00')
  let length = try! @rank_group_wire.encode_rank_group_wire_payload_into(
    wire,
    limits,
    binding,
    Configure,
    ParentToChild,
    None,
    0UL,
    payload,
    3,
  )
  let decoded = try! @rank_group_wire.decode_rank_group_wire_configure(
    wire, length, limits,
  )
  let header = decoded.header()
  assert_eq(header.kind(), Configure)
  assert_eq(header.direction(), ParentToChild)
  assert_eq(header.slot(), None)
  assert_eq(header.sequence_value(), 0UL)
  assert_eq(header.payload_length(), 3)
  assert_eq(header.length(), length)
  assert_eq(decoded.binding().rank(), 0)
  assert_eq(decoded.binding().worker_digest(), binding.worker_digest())
  assert_eq(
    decoded.binding().execution_plan_digest(),
    binding.execution_plan_digest(),
  )

  // The decoded values remain exact after caller-owned source mutation.
  wire[16] = b'\x7f'
  wire[320] = b'0'
  assert_eq(header.kind(), Configure)
  assert_eq(header.payload_length(), 3)
  assert_eq(decoded.binding().worker_digest(), binding.worker_digest())
}

///|
test "embedded payload sealing writes only the header around payload" {
  let limits = fixture_wire_limits()
  let header = rank_group_wire_header_bytes()
  let storage = FixedArray::make(limits.max_frame_bytes(), b'\x00')
  storage[header] = b'p'
  storage[header + 1] = b'l'
  storage[header + 2] = b'a'
  storage[header + 3] = b'n'
  let length = seal_rank_group_wire_embedded_payload(
    storage,
    limits,
    fixture_wire_binding(0),
    Submit,
    ParentToChild,
    Some(A),
    1UL,
    4,
  )
  assert_eq(storage[header], b'p')
  assert_eq(storage[header + 3], b'n')
  assert_eq(
    decode_rank_group_wire_header(storage, length, limits).payload_length(),
    4,
  )
}

///|
test "rank group wire stateless locked binding rejects rank substitution" {
  let limits = fixture_wire_limits()
  let rank_zero = fixture_wire_binding(0)
  let rank_one = fixture_wire_binding(1)
  let payload = FixedArray::make(1, b'\x01')
  let wire = FixedArray::make(limits.max_frame_bytes(), b'\x00')
  let length = try! @rank_group_wire.encode_rank_group_wire_payload_into(
    wire,
    limits,
    rank_zero,
    Submit,
    ParentToChild,
    Some(A),
    1UL,
    payload,
    1,
  )
  let header = try! @rank_group_wire.validate_rank_group_wire_binding(
    wire, length, limits, rank_zero,
  )
  assert_eq(header.kind(), Submit)
  assert_eq(header.slot(), Some(A))
  try
    @rank_group_wire.validate_rank_group_wire_binding(
      wire, length, limits, rank_one,
    )
  catch {
    Invalid(Binding) => ()
    _ => fail("expected exact locked-binding rejection")
  } noraise {
    _ => fail("expected exact locked-binding rejection")
  }
}

///|
test "rank group wire FrameBuffer and stateless encoder share canonical bytes" {
  let limits = fixture_wire_limits()
  let binding = fixture_wire_binding(0)
  let payload = FixedArray::makei(4, index => (index + 1).to_byte())
  let direct = FixedArray::make(limits.max_frame_bytes(), b'\x00')
  let direct_length = try! @rank_group_wire.encode_rank_group_wire_payload_into(
    direct,
    limits,
    binding,
    Submit,
    ParentToChild,
    Some(B),
    7UL,
    payload,
    4,
  )
  let buffer = @rank_group_wire.RankGroupWireFrameBuffer::new(limits)
  let frame = try! buffer.encode_payload(
    binding,
    Submit,
    ParentToChild,
    B,
    7UL,
    payload,
    4,
  )
  let buffered = FixedArray::make(limits.max_frame_bytes(), b'\x00')
  try! frame.copy_to(buffered, destination_offset=0)
  assert_eq(try! frame.length(), direct_length)
  for index in 0..<direct_length {
    assert_eq(buffered[index], direct[index])
  }
}

///|
test "rank group wire stateless payload encoding tolerates destination alias" {
  let limits = fixture_wire_limits()
  let binding = fixture_wire_binding(0)
  let destination = FixedArray::make(limits.max_frame_bytes(), b'\x00')
  destination[0] = b'\x11'
  destination[1] = b'\x22'
  destination[2] = b'\x33'
  let length = try! @rank_group_wire.encode_rank_group_wire_payload_into(
    destination,
    limits,
    binding,
    Submit,
    ParentToChild,
    Some(A),
    1UL,
    destination,
    3,
  )
  let receiver = @rank_group_wire.RankGroupWireFrameBuffer::new(limits)
  let frame = try! receiver.load(destination, length)
  let copied = FixedArray::make(3, b'\x00')
  try! frame.copy_payload_to(copied, destination_offset=0)
  assert_eq(copied[0], b'\x11')
  assert_eq(copied[1], b'\x22')
  assert_eq(copied[2], b'\x33')
}

///|
test "rank group wire Configure requires bounded nonempty opaque payload" {
  let limits = fixture_wire_limits()
  let binding = fixture_wire_binding(0)
  let destination = FixedArray::make(limits.max_frame_bytes(), b'\x00')
  try
    @rank_group_wire.encode_rank_group_wire_control_into(
      destination,
      limits,
      binding,
      Configure,
      ParentToChild,
      None,
      0UL,
    )
  catch {
    Invalid(Payload) => ()
    _ => fail("expected empty Configure rejection")
  } noraise {
    _ => fail("expected empty Configure rejection")
  }

  let payload = FixedArray::make(1, b'\x01')
  try
    @rank_group_wire.encode_rank_group_wire_payload_into(
      destination,
      limits,
      binding,
      Ready,
      ChildToParent,
      None,
      0UL,
      payload,
      1,
    )
  catch {
    Invalid(Payload) => ()
    _ => fail("expected payload-bearing Ready rejection")
  } noraise {
    _ => fail("expected payload-bearing Ready rejection")
  }
}
