///|
fn accept_rank_and_group(
  group : @rank_group_wire.RankGroupWireTranscript,
  rank_transcript : @rank_group_wire.RankGroupWireRankTranscript,
  sender : @rank_group_wire.RankGroupWireFrameBuffer,
  limits : @rank_group_wire.RankGroupWireLimits,
  binding : @rank_group_wire.RankGroupWireBinding,
  kind : @rank_group_wire.RankGroupWireKind,
  direction : @rank_group_wire.RankGroupWireDirection,
  slot : @rank_group_wire.RankGroupWireSlot?,
  sequence_value : UInt64,
  payload : FixedArray[Byte]?,
) -> Unit {
  let frame = match (kind, payload) {
    (Configure, Some(bytes)) =>
      try! sender.encode_configure_payload(binding, bytes, bytes.length())
    (_, Some(bytes)) =>
      try! sender.encode_payload(
        binding,
        kind,
        direction,
        slot.unwrap(),
        sequence_value,
        bytes,
        bytes.length(),
      )
    (_, None) =>
      try! sender.encode_control(binding, kind, direction, slot, sequence_value)
  }
  try! group.accept(frame)
  let source = FixedArray::make(sender.capacity(), b'\x00')
  let length = try! frame.length()
  try! frame.copy_to(source, destination_offset=0)
  if kind == Configure {
    ignore(try! rank_transcript.accept_configure(source, length, limits))
  } else if direction == ParentToChild {
    ignore(try! rank_transcript.accept_parent(source, length, limits))
  } else {
    ignore(try! rank_transcript.accept_child(source, length, limits))
  }
}

///|
fn ready_local_rank_fixture() -> (
  @rank_group_wire.RankGroupWireRankTranscript,
  @rank_group_wire.RankGroupWireFrameBuffer,
  @rank_group_wire.RankGroupWireLimits,
  @rank_group_wire.RankGroupWireBinding,
  FixedArray[Byte],
) {
  let limits = fixture_wire_limits()
  let binding = fixture_wire_binding(0)
  let sender = @rank_group_wire.RankGroupWireFrameBuffer::new(limits)
  let source = FixedArray::make(sender.capacity(), b'\x00')
  let payload = FixedArray::make(1, b'\x31')
  let configure = try! sender.encode_configure_payload(binding, payload, 1)
  let configure_length = try! configure.length()
  try! configure.copy_to(source, destination_offset=0)
  let rank_transcript = @rank_group_wire.RankGroupWireRankTranscript::new()
  ignore(
    try! rank_transcript.accept_configure(source, configure_length, limits),
  )
  try! rank_transcript.admit_ready(binding, previous_sequence_value=0UL)
  let ready = try! sender.encode_control(
    binding,
    Ready,
    ChildToParent,
    None,
    0UL,
  )
  let ready_length = try! ready.length()
  try! ready.copy_to(source, destination_offset=0)
  ignore(try! rank_transcript.accept_child(source, ready_length, limits))
  (rank_transcript, sender, limits, binding, source)
}

///|
test "rank-local transcript matches group ordering for one complete plan" {
  let bindings = fixture_wire_bindings()
  let limits = fixture_wire_limits()
  let group = try! @rank_group_wire.RankGroupWireTranscript::new(
    bindings,
    previous_sequence_value=0UL,
  )
  let locals = [
    @rank_group_wire.RankGroupWireRankTranscript::new(),
    @rank_group_wire.RankGroupWireRankTranscript::new(),
  ]
  let sender = @rank_group_wire.RankGroupWireFrameBuffer::new(limits)
  let configure_payload = FixedArray::make(2, b'\x41')
  for rank in 0..<2 {
    accept_rank_and_group(
      group,
      locals[rank],
      sender,
      limits,
      bindings[rank],
      Configure,
      ParentToChild,
      None,
      0UL,
      Some(configure_payload),
    )
    try! locals[rank].admit_ready(bindings[rank], previous_sequence_value=0UL)
  }
  for rank in 0..<2 {
    accept_rank_and_group(
      group,
      locals[rank],
      sender,
      limits,
      bindings[rank],
      Ready,
      ChildToParent,
      None,
      0UL,
      None,
    )
  }
  let submit_payload = FixedArray::make(3, b'\x51')
  for rank in 0..<2 {
    accept_rank_and_group(
      group,
      locals[rank],
      sender,
      limits,
      bindings[rank],
      Submit,
      ParentToChild,
      Some(A),
      1UL,
      Some(submit_payload),
    )
  }
  for rank in 0..<2 {
    accept_rank_and_group(
      group,
      locals[rank],
      sender,
      limits,
      bindings[rank],
      PollComplete,
      ParentToChild,
      Some(A),
      1UL,
      None,
    )
    let completion = if rank == 0 {
      Some(FixedArray::make(2, b'\x61'))
    } else {
      None
    }
    accept_rank_and_group(
      group,
      locals[rank],
      sender,
      limits,
      bindings[rank],
      Complete,
      ChildToParent,
      Some(A),
      1UL,
      completion,
    )
  }
  assert_eq(group.lifecycle(), ReadyGroup)
  for rank_transcript in locals {
    assert_eq(rank_transcript.lifecycle(), ReadyGroup)
    assert_eq(rank_transcript.previous_sequence_value(), 1UL)
  }
}

///|
test "rank-local transcript rejects startup binding and sequence substitution" {
  let limits = fixture_wire_limits()
  let expected = fixture_wire_binding(0)
  let substituted = fixture_wire_binding(0, group_generation_value=2UL)
  let sender = @rank_group_wire.RankGroupWireFrameBuffer::new(limits)
  let payload = FixedArray::make(1, b'\x31')
  let configure = try! sender.encode_configure_payload(expected, payload, 1)
  let source = FixedArray::make(sender.capacity(), b'\x00')
  let length = try! configure.length()
  try! configure.copy_to(source, destination_offset=0)
  let rank_transcript = @rank_group_wire.RankGroupWireRankTranscript::new()
  ignore(try! rank_transcript.accept_configure(source, length, limits))
  try
    rank_transcript.admit_ready(substituted, previous_sequence_value=0UL)
  catch {
    ProtocolFault(Binding) => ()
    _ => fail("unexpected local binding-substitution error")
  } noraise {
    _ => fail("local binding substitution was admitted")
  }

  let ready = @rank_group_wire.RankGroupWireRankTranscript::new()
  ignore(try! ready.accept_configure(source, length, limits))
  try! ready.admit_ready(expected, previous_sequence_value=4UL)
  let ready_frame = try! sender.encode_control(
    expected,
    Ready,
    ChildToParent,
    None,
    0UL,
  )
  let ready_length = try! ready_frame.length()
  try! ready_frame.copy_to(source, destination_offset=0)
  ignore(try! ready.accept_child(source, ready_length, limits))
  let skipped = try! sender.encode_payload(
    expected,
    Submit,
    ParentToChild,
    A,
    6UL,
    payload,
    1,
  )
  let skipped_length = try! skipped.length()
  try! skipped.copy_to(source, destination_offset=0)
  try ready.accept_parent(source, skipped_length, limits) catch {
    ProtocolFault(Sequence) => ()
    _ => fail("unexpected local sequence-skip error")
  } noraise {
    _ => fail("local sequence skip was admitted")
  }
}

///|
test "rank-local transcript rejects reflected responses before state mutation" {
  let (rank_transcript, sender, limits, binding, source) = ready_local_rank_fixture()
  let reflected = try! sender.encode_control(
    binding,
    Pending,
    ChildToParent,
    Some(A),
    1UL,
  )
  let length = try! reflected.length()
  try! reflected.copy_to(source, destination_offset=0)
  try rank_transcript.accept_parent(source, length, limits) catch {
    ProtocolFault(Direction) => ()
    _ => fail("unexpected reflected-response error")
  } noraise {
    _ => fail("reflected child response advanced parent-command state")
  }
  assert_eq(rank_transcript.lifecycle(), Faulted)
}

///|
test "rank-local transcript rejects duplicate and post-abort Abort" {
  let (duplicate, sender, limits, binding, source) = ready_local_rank_fixture()
  let abort_frame = try! sender.encode_control(
    binding,
    Abort,
    ParentToChild,
    None,
    0UL,
  )
  let abort_length = try! abort_frame.length()
  try! abort_frame.copy_to(source, destination_offset=0)
  ignore(try! duplicate.accept_parent(source, abort_length, limits))
  assert_eq(duplicate.lifecycle(), Aborting)
  try duplicate.accept_parent(source, abort_length, limits) catch {
    ProtocolFault(Lifecycle) => ()
    _ => fail("unexpected duplicate Abort error")
  } noraise {
    _ => fail("duplicate Abort was admitted")
  }

  let (post_abort, response_sender, post_limits, post_binding, post_source) = ready_local_rank_fixture()
  let first = try! response_sender.encode_control(
    post_binding,
    Abort,
    ParentToChild,
    None,
    0UL,
  )
  let first_length = try! first.length()
  try! first.copy_to(post_source, destination_offset=0)
  ignore(try! post_abort.accept_parent(post_source, first_length, post_limits))
  let acknowledged = try! response_sender.encode_control(
    post_binding,
    Aborted,
    ChildToParent,
    None,
    0UL,
  )
  let acknowledged_length = try! acknowledged.length()
  try! acknowledged.copy_to(post_source, destination_offset=0)
  ignore(
    try! post_abort.accept_child(post_source, acknowledged_length, post_limits),
  )
  assert_eq(post_abort.lifecycle(), AbortedPlan)
  let repeated = try! response_sender.encode_control(
    post_binding,
    Abort,
    ParentToChild,
    None,
    0UL,
  )
  let repeated_length = try! repeated.length()
  try! repeated.copy_to(post_source, destination_offset=0)
  try
    post_abort.accept_parent(post_source, repeated_length, post_limits)
  catch {
    ProtocolFault(Lifecycle) => ()
    _ => fail("unexpected post-abort Abort error")
  } noraise {
    _ => fail("Abort was admitted after AbortedPlan")
  }
}
