///|
fn fixture_wire_identity() -> @spec.ModelIdentity {
  let content = try! @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let plan = try! @spec.PlanDigest::from_sha256(
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  )
  try! @spec.ModelIdentity::new(content, plan)
}

///|
fn fixture_wire_binding(
  rank : Int,
  group_generation_value? : UInt64 = 1UL,
) -> @rank_group_wire.RankGroupWireBinding {
  try! @rank_group_wire.RankGroupWireBinding::new(
    identity=fixture_wire_identity(),
    model_generation_value=1UL,
    group_generation_value~,
    rank~,
    world_size=2,
    group_digest="1111111111111111111111111111111111111111111111111111111111111111",
    topology_digest="2222222222222222222222222222222222222222222222222222222222222222",
    worker_digest=if rank == 0 {
      "3333333333333333333333333333333333333333333333333333333333333333"
    } else {
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    },
    device_plan_digest=if rank == 0 {
      "4444444444444444444444444444444444444444444444444444444444444444"
    } else {
      "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    },
    execution_plan_digest="9999999999999999999999999999999999999999999999999999999999999999",
    collective_digest=if rank == 0 {
      "5555555555555555555555555555555555555555555555555555555555555555"
    } else {
      "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    },
    kv_digest=if rank == 0 {
      "6666666666666666666666666666666666666666666666666666666666666666"
    } else {
      "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
    },
  )
}

///|
fn fixture_wire_bindings() -> Array[@rank_group_wire.RankGroupWireBinding] {
  [fixture_wire_binding(0), fixture_wire_binding(1)]
}

///|
fn fixture_wire_limits() -> @rank_group_wire.RankGroupWireLimits {
  try! @rank_group_wire.RankGroupWireLimits::new(max_payload_bytes=64)
}

///|
fn accept_wire_control(
  transcript : @rank_group_wire.RankGroupWireTranscript,
  sender : @rank_group_wire.RankGroupWireFrameBuffer,
  binding : @rank_group_wire.RankGroupWireBinding,
  kind : @rank_group_wire.RankGroupWireKind,
  direction : @rank_group_wire.RankGroupWireDirection,
  slot : @rank_group_wire.RankGroupWireSlot?,
  sequence : UInt64,
) -> Unit {
  let frame = if kind == Configure {
    let payload = FixedArray::make(1, b'\x01')
    try! sender.encode_configure_payload(binding, payload, 1)
  } else {
    try! sender.encode_control(binding, kind, direction, slot, sequence)
  }
  try! transcript.accept(frame)
}

///|
fn ready_wire_transcript() -> (
  @rank_group_wire.RankGroupWireTranscript,
  @rank_group_wire.RankGroupWireFrameBuffer,
  Array[@rank_group_wire.RankGroupWireBinding],
) {
  let bindings = fixture_wire_bindings()
  let transcript = try! @rank_group_wire.RankGroupWireTranscript::new(
    bindings,
    previous_sequence_value=0UL,
  )
  let sender = @rank_group_wire.RankGroupWireFrameBuffer::new(
    fixture_wire_limits(),
  )
  for binding in bindings {
    accept_wire_control(
      transcript,
      sender,
      binding,
      Configure,
      ParentToChild,
      None,
      0UL,
    )
  }
  for binding in bindings {
    accept_wire_control(
      transcript,
      sender,
      binding,
      Ready,
      ChildToParent,
      None,
      0UL,
    )
  }
  (transcript, sender, bindings)
}
