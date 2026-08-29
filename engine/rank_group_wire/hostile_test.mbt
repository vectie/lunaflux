///|
test "rank group wire requires every configure before any ready and faults sticky" {
  let bindings = fixture_wire_bindings()
  let transcript = try! @rank_group_wire.RankGroupWireTranscript::new(
    bindings,
    previous_sequence_value=0UL,
  )
  let sender = @rank_group_wire.RankGroupWireFrameBuffer::new(
    fixture_wire_limits(),
  )
  accept_wire_control(
    transcript,
    sender,
    bindings[0],
    Configure,
    ParentToChild,
    None,
    0UL,
  )
  let early_ready = try! sender.encode_control(
    bindings[0],
    Ready,
    ChildToParent,
    None,
    0UL,
  )
  try transcript.accept(early_ready) catch {
    ProtocolFault(Lifecycle) => ()
    _ => fail("expected startup barrier fault")
  } noraise {
    _ => fail("expected startup barrier fault")
  }
  assert_eq(transcript.lifecycle(), Faulted)
  let configure_payload = FixedArray::make(1, b'\x01')
  let later = try! sender.encode_configure_payload(
    bindings[1],
    configure_payload,
    1,
  )
  try transcript.accept(later) catch {
    TranscriptFaulted(Lifecycle) => ()
    _ => fail("expected sticky transcript fault")
  } noraise {
    _ => fail("expected sticky transcript fault")
  }
}

///|
test "rank group wire aborts a partially configured group without waiting on absent ranks" {
  let bindings = fixture_wire_bindings()
  let transcript = try! @rank_group_wire.RankGroupWireTranscript::new(
    bindings,
    previous_sequence_value=0UL,
  )
  let sender = @rank_group_wire.RankGroupWireFrameBuffer::new(
    fixture_wire_limits(),
  )
  accept_wire_control(
    transcript,
    sender,
    bindings[0],
    Configure,
    ParentToChild,
    None,
    0UL,
  )
  accept_wire_control(
    transcript,
    sender,
    bindings[0],
    Abort,
    ParentToChild,
    None,
    0UL,
  )
  accept_wire_control(
    transcript,
    sender,
    bindings[0],
    Aborted,
    ChildToParent,
    None,
    0UL,
  )
  assert_eq(transcript.lifecycle(), AbortedPlan)
}

///|
test "rank group wire rejects exact generation and rank binding substitution" {
  let (transcript, sender, bindings) = ready_wire_transcript()
  let foreign_generation = fixture_wire_binding(0, group_generation_value=2UL)
  let payload = FixedArray::make(1, b'\x01')
  let frame = try! sender.encode_payload(
    foreign_generation,
    Submit,
    ParentToChild,
    A,
    1UL,
    payload,
    1,
  )
  try transcript.accept(frame) catch {
    ProtocolFault(Binding) => ()
    _ => fail("expected generation binding fault")
  } noraise {
    _ => fail("expected generation binding fault")
  }
  assert_eq(transcript.lifecycle(), Faulted)
  assert_eq(bindings[0].rank(), 0)
}

///|
test "rank group wire rejects A B reuse and completion before poll" {
  let (transcript, sender, bindings) = ready_wire_transcript()
  let payload = FixedArray::make(1, b'\x01')
  for binding in bindings {
    let submit = try! sender.encode_payload(
      binding,
      Submit,
      ParentToChild,
      A,
      1UL,
      payload,
      1,
    )
    try! transcript.accept(submit)
  }
  let wrong_slot = try! sender.encode_payload(
    bindings[0],
    Submit,
    ParentToChild,
    B,
    2UL,
    payload,
    1,
  )
  try transcript.accept(wrong_slot) catch {
    ProtocolFault(Sequence) => ()
    _ => fail("expected live-slot reuse fault")
  } noraise {
    _ => fail("expected live-slot reuse fault")
  }

  let (second, second_sender, second_bindings) = ready_wire_transcript()
  for binding in second_bindings {
    let submit = try! second_sender.encode_payload(
      binding,
      Submit,
      ParentToChild,
      A,
      1UL,
      payload,
      1,
    )
    try! second.accept(submit)
  }
  let premature = try! second_sender.encode_payload(
    second_bindings[0],
    Complete,
    ChildToParent,
    A,
    1UL,
    payload,
    1,
  )
  try second.accept(premature) catch {
    ProtocolFault(Lifecycle) => ()
    _ => fail("expected completion-before-poll fault")
  } noraise {
    _ => fail("expected completion-before-poll fault")
  }
}

///|
test "rank group wire duplicate configure poisons the exact transcript" {
  let bindings = fixture_wire_bindings()
  let transcript = try! @rank_group_wire.RankGroupWireTranscript::new(
    bindings,
    previous_sequence_value=0UL,
  )
  let sender = @rank_group_wire.RankGroupWireFrameBuffer::new(
    fixture_wire_limits(),
  )
  accept_wire_control(
    transcript,
    sender,
    bindings[0],
    Configure,
    ParentToChild,
    None,
    0UL,
  )
  let configure_payload = FixedArray::make(1, b'\x01')
  let duplicate = try! sender.encode_configure_payload(
    bindings[0],
    configure_payload,
    1,
  )
  try transcript.accept(duplicate) catch {
    ProtocolFault(Lifecycle) => ()
    _ => fail("expected duplicate configure fault")
  } noraise {
    _ => fail("expected duplicate configure fault")
  }
  assert_eq(transcript.lifecycle(), Faulted)
}

///|
test "rank group wire validates limits and lowercase digests" {
  try @rank_group_wire.RankGroupWireLimits::new(max_payload_bytes=0) catch {
    Invalid(Limit) => ()
    _ => fail("expected zero limit rejection")
  } noraise {
    _ => fail("expected zero limit rejection")
  }
  try
    @rank_group_wire.RankGroupWireBinding::new(
      identity=fixture_wire_identity(),
      model_generation_value=1UL,
      group_generation_value=1UL,
      rank=0,
      world_size=2,
      group_digest="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
      topology_digest="2222222222222222222222222222222222222222222222222222222222222222",
      worker_digest="3333333333333333333333333333333333333333333333333333333333333333",
      device_plan_digest="4444444444444444444444444444444444444444444444444444444444444444",
      execution_plan_digest="9999999999999999999999999999999999999999999999999999999999999999",
      collective_digest="5555555555555555555555555555555555555555555555555555555555555555",
      kv_digest="6666666666666666666666666666666666666666666666666666666666666666",
    )
  catch {
    Invalid(Digest) => ()
    _ => fail("expected noncanonical digest rejection")
  } noraise {
    _ => fail("expected noncanonical digest rejection")
  }
}

///|
test "rank group wire maximum payload fits process whole-frame bound" {
  let limits = try! @rank_group_wire.RankGroupWireLimits::new(
    max_payload_bytes=67108220,
  )
  assert_eq(limits.max_frame_bytes(), 67108864)
  try
    @rank_group_wire.RankGroupWireLimits::new(max_payload_bytes=67108221)
  catch {
    Invalid(Limit) => ()
    _ => fail("unexpected oversized wire limit error")
  } noraise {
    _ => fail("wire payload exceeding process frame bound was admitted")
  }
}
