///|
fn fixture_private_binding() -> RankGroupWireBinding {
  let content = try! @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let plan = try! @spec.PlanDigest::from_sha256(
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  )
  let identity = try! @spec.ModelIdentity::new(content, plan)
  try! RankGroupWireBinding::new(
    identity~,
    model_generation_value=1UL,
    group_generation_value=1UL,
    rank=0,
    world_size=2,
    group_digest="1111111111111111111111111111111111111111111111111111111111111111",
    topology_digest="2222222222222222222222222222222222222222222222222222222222222222",
    worker_digest="3333333333333333333333333333333333333333333333333333333333333333",
    device_plan_digest="4444444444444444444444444444444444444444444444444444444444444444",
    execution_plan_digest="9999999999999999999999999999999999999999999999999999999999999999",
    collective_digest="5555555555555555555555555555555555555555555555555555555555555555",
    kv_digest="6666666666666666666666666666666666666666666666666666666666666666",
  )
}

///|
fn canonical_private_frame() -> (FixedArray[Byte], Int, RankGroupWireLimits) {
  let limits = try! RankGroupWireLimits::new(max_payload_bytes=8)
  let sender = RankGroupWireFrameBuffer::new(limits)
  let payload = FixedArray::make(1, b'\x01')
  let frame = try! sender.encode_configure_payload(
    fixture_private_binding(),
    payload,
    1,
  )
  let wire = FixedArray::make(sender.capacity(), b'\x00')
  let length = try! frame.length()
  try! frame.copy_to(wire, destination_offset=0)
  (wire, length, limits)
}

///|
fn refresh_private_checksum(bytes : FixedArray[Byte], length : Int) -> Unit {
  wire_write_u32(
    bytes,
    RANK_GROUP_WIRE_CHECKSUM_OFFSET,
    wire_checksum(bytes, length),
  )
}

///|
test "rank group wire hostile header corruption is rejected exactly" {
  let (wire, length, limits) = canonical_private_frame()
  let receiver = RankGroupWireFrameBuffer::new(limits)
  wire[0] = b'X'
  refresh_private_checksum(wire, length)
  try receiver.load(wire, length) catch {
    Invalid(Magic) => ()
    _ => fail("expected magic rejection")
  } noraise {
    _ => fail("expected magic rejection")
  }

  let (reserved, reserved_length, reserved_limits) = canonical_private_frame()
  wire_write_u32(reserved, 640, 1U)
  refresh_private_checksum(reserved, reserved_length)
  try
    RankGroupWireFrameBuffer::new(reserved_limits).load(
      reserved, reserved_length,
    )
  catch {
    Invalid(Reserved) => ()
    _ => fail("expected reserved rejection")
  } noraise {
    _ => fail("expected reserved rejection")
  }

  let (direction, direction_length, direction_limits) = canonical_private_frame()
  wire_write_i32(direction, 60, 1)
  refresh_private_checksum(direction, direction_length)
  try
    RankGroupWireFrameBuffer::new(direction_limits).load(
      direction, direction_length,
    )
  catch {
    Invalid(Direction) => ()
    _ => fail("expected direction rejection")
  } noraise {
    _ => fail("expected direction rejection")
  }
}

///|
test "rank group wire codec fault poisons group transcript" {
  let bindings = [
    fixture_private_binding(),
    try! RankGroupWireBinding::new(
      identity=fixture_private_binding().identity(),
      model_generation_value=1UL,
      group_generation_value=1UL,
      rank=1,
      world_size=2,
      group_digest="1111111111111111111111111111111111111111111111111111111111111111",
      topology_digest="2222222222222222222222222222222222222222222222222222222222222222",
      worker_digest="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      device_plan_digest="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      execution_plan_digest="9999999999999999999999999999999999999999999999999999999999999999",
      collective_digest="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      kv_digest="dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
    ),
  ]
  let transcript = try! RankGroupWireTranscript::new(
    bindings,
    previous_sequence_value=0UL,
  )
  let (wire, length, limits) = canonical_private_frame()
  wire[64] = b'1'
  try
    transcript.load_and_accept(
      RankGroupWireFrameBuffer::new(limits),
      wire,
      length,
    )
  catch {
    ProtocolFault(Checksum) => ()
    _ => fail("expected codec fault")
  } noraise {
    _ => fail("expected codec fault")
  }
  assert_eq(transcript.lifecycle(), Faulted)
  assert_eq(transcript.failure_rule(), Some(Checksum))
}

///|
test "rank group wire locked binding rejects execution-plan substitution" {
  let (wire, length, limits) = canonical_private_frame()
  let binding = fixture_private_binding()
  wire[448] = if wire[448] == b'9' { b'8' } else { b'9' }
  refresh_private_checksum(wire, length)
  try validate_rank_group_wire_binding(wire, length, limits, binding) catch {
    Invalid(Binding) => ()
    _ => fail("unexpected execution-plan substitution rejection")
  } noraise {
    _ => fail("substituted execution-plan digest matched locked binding")
  }
}
