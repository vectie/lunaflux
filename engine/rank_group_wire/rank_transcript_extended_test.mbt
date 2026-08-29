///|
struct DifferentialRankWireRig {
  group : @rank_group_wire.RankGroupWireTranscript
  locals : Array[@rank_group_wire.RankGroupWireRankTranscript]
  sender : @rank_group_wire.RankGroupWireFrameBuffer
  limits : @rank_group_wire.RankGroupWireLimits
  bindings : Array[@rank_group_wire.RankGroupWireBinding]
}

///|
fn configured_differential_rig() -> DifferentialRankWireRig {
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
  }
  { group, locals, sender, limits, bindings }
}

///|
fn ready_differential_rig() -> DifferentialRankWireRig {
  let rig = configured_differential_rig()
  for rank in 0..<2 {
    try! rig.locals[rank].admit_ready(
      rig.bindings[rank],
      previous_sequence_value=0UL,
    )
    accept_rank_and_group(
      rig.group,
      rig.locals[rank],
      rig.sender,
      rig.limits,
      rig.bindings[rank],
      Ready,
      ChildToParent,
      None,
      0UL,
      None,
    )
  }
  rig
}

///|
fn assert_differential_phase(
  rig : DifferentialRankWireRig,
  lifecycle : @rank_group_wire.RankGroupWireLifecycle,
  previous_sequence_value : UInt64,
  active_sequence_value : UInt64,
) -> Unit raise {
  assert_eq(rig.group.lifecycle(), lifecycle)
  assert_eq(rig.group.previous_sequence_value(), previous_sequence_value)
  assert_eq(rig.group.active_sequence_value(), active_sequence_value)
  for rank_transcript in rig.locals {
    assert_eq(rank_transcript.lifecycle(), lifecycle)
    assert_eq(
      rank_transcript.previous_sequence_value(),
      previous_sequence_value,
    )
    assert_eq(rank_transcript.active_sequence_value(), active_sequence_value)
  }
}

///|
fn submit_differential_plan(
  rig : DifferentialRankWireRig,
  sequence : UInt64,
  slot : @rank_group_wire.RankGroupWireSlot,
) -> Unit {
  let payload = FixedArray::make(3, b'\x51')
  for rank in 0..<2 {
    accept_rank_and_group(
      rig.group,
      rig.locals[rank],
      rig.sender,
      rig.limits,
      rig.bindings[rank],
      Submit,
      ParentToChild,
      Some(slot),
      sequence,
      Some(payload),
    )
  }
}

///|
fn abort_differential_plan(
  rig : DifferentialRankWireRig,
  slot : @rank_group_wire.RankGroupWireSlot?,
  sequence : UInt64,
) -> Unit {
  for rank in 0..<2 {
    accept_rank_and_group(
      rig.group,
      rig.locals[rank],
      rig.sender,
      rig.limits,
      rig.bindings[rank],
      Abort,
      ParentToChild,
      slot,
      sequence,
      None,
    )
  }
  for rank in 0..<2 {
    accept_rank_and_group(
      rig.group,
      rig.locals[rank],
      rig.sender,
      rig.limits,
      rig.bindings[rank],
      Aborted,
      ChildToParent,
      slot,
      sequence,
      None,
    )
  }
}

///|
fn terminal_differential_exchange(
  rig : DifferentialRankWireRig,
  command : @rank_group_wire.RankGroupWireKind,
  response : @rank_group_wire.RankGroupWireKind,
) -> Unit {
  for rank in 0..<2 {
    accept_rank_and_group(
      rig.group,
      rig.locals[rank],
      rig.sender,
      rig.limits,
      rig.bindings[rank],
      command,
      ParentToChild,
      None,
      0UL,
      None,
    )
  }
  for rank in 0..<2 {
    accept_rank_and_group(
      rig.group,
      rig.locals[rank],
      rig.sender,
      rig.limits,
      rig.bindings[rank],
      response,
      ChildToParent,
      None,
      0UL,
      None,
    )
  }
}

///|
test "rank-local transcripts match group Pending loops and completion roles" {
  let rig = ready_differential_rig()
  assert_differential_phase(rig, ReadyGroup, 0UL, 0UL)
  submit_differential_plan(rig, 1UL, A)
  assert_differential_phase(rig, Executing, 0UL, 1UL)
  for _round in 0..<2 {
    for rank in 0..<2 {
      accept_rank_and_group(
        rig.group,
        rig.locals[rank],
        rig.sender,
        rig.limits,
        rig.bindings[rank],
        PollComplete,
        ParentToChild,
        Some(A),
        1UL,
        None,
      )
      accept_rank_and_group(
        rig.group,
        rig.locals[rank],
        rig.sender,
        rig.limits,
        rig.bindings[rank],
        Pending,
        ChildToParent,
        Some(A),
        1UL,
        None,
      )
    }
    assert_differential_phase(rig, Executing, 0UL, 1UL)
  }
  for rank in 0..<2 {
    accept_rank_and_group(
      rig.group,
      rig.locals[rank],
      rig.sender,
      rig.limits,
      rig.bindings[rank],
      PollComplete,
      ParentToChild,
      Some(A),
      1UL,
      None,
    )
    let leader_payload = if rank == 0 {
      Some(FixedArray::make(2, b'\x61'))
    } else {
      None
    }
    accept_rank_and_group(
      rig.group,
      rig.locals[rank],
      rig.sender,
      rig.limits,
      rig.bindings[rank],
      Complete,
      ChildToParent,
      Some(A),
      1UL,
      leader_payload,
    )
  }
  assert_differential_phase(rig, ReadyGroup, 1UL, 0UL)
}

///|
test "rank-local transcripts match active and inactive abort terminal flow" {
  let active = ready_differential_rig()
  submit_differential_plan(active, 1UL, B)
  abort_differential_plan(active, Some(B), 1UL)
  assert_differential_phase(active, AbortedPlan, 1UL, 0UL)
  terminal_differential_exchange(active, Drain, Drained)
  assert_differential_phase(active, DrainedGroup, 1UL, 0UL)
  terminal_differential_exchange(active, Close, Closed)
  assert_differential_phase(active, ClosedGroup, 1UL, 0UL)

  let configuring = configured_differential_rig()
  abort_differential_plan(configuring, None, 0UL)
  assert_differential_phase(configuring, AbortedPlan, 0UL, 0UL)

  let ready = ready_differential_rig()
  abort_differential_plan(ready, None, 0UL)
  assert_differential_phase(ready, AbortedPlan, 0UL, 0UL)
}

///|
fn polling_local_rank(
  rank : Int,
) -> (
  @rank_group_wire.RankGroupWireRankTranscript,
  @rank_group_wire.RankGroupWireFrameBuffer,
  @rank_group_wire.RankGroupWireLimits,
  @rank_group_wire.RankGroupWireBinding,
  FixedArray[Byte],
) {
  let limits = fixture_wire_limits()
  let binding = fixture_wire_binding(rank)
  let sender = @rank_group_wire.RankGroupWireFrameBuffer::new(limits)
  let source = FixedArray::make(sender.capacity(), b'\x00')
  let configure_payload = FixedArray::make(1, b'\x31')
  let configure = try! sender.encode_configure_payload(
    binding, configure_payload, 1,
  )
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
  let payload = FixedArray::make(1, b'\x41')
  let submit = try! sender.encode_payload(
    binding,
    Submit,
    ParentToChild,
    A,
    1UL,
    payload,
    1,
  )
  let submit_length = try! submit.length()
  try! submit.copy_to(source, destination_offset=0)
  ignore(try! rank_transcript.accept_parent(source, submit_length, limits))
  let poll = try! sender.encode_control(
    binding,
    PollComplete,
    ParentToChild,
    Some(A),
    1UL,
  )
  let poll_length = try! poll.length()
  try! poll.copy_to(source, destination_offset=0)
  ignore(try! rank_transcript.accept_parent(source, poll_length, limits))
  (rank_transcript, sender, limits, binding, source)
}

///|
test "rank-local completion rejects leader and follower payload substitution" {
  let (leader, leader_sender, leader_limits, leader_binding, leader_source) = polling_local_rank(
    0,
  )
  let empty_leader = try! leader_sender.encode_control(
    leader_binding,
    Complete,
    ChildToParent,
    Some(A),
    1UL,
  )
  let empty_length = try! empty_leader.length()
  try! empty_leader.copy_to(leader_source, destination_offset=0)
  try leader.accept_child(leader_source, empty_length, leader_limits) catch {
    ProtocolFault(Payload) => ()
    _ => fail("unexpected leader empty-completion error")
  } noraise {
    _ => fail("leader zero-payload completion was admitted")
  }

  let (
    follower,
    follower_sender,
    follower_limits,
    follower_binding,
    follower_source,
  ) = polling_local_rank(1)
  let payload = FixedArray::make(1, b'\x71')
  let nonempty_follower = try! follower_sender.encode_payload(
    follower_binding,
    Complete,
    ChildToParent,
    A,
    1UL,
    payload,
    1,
  )
  let nonempty_length = try! nonempty_follower.length()
  try! nonempty_follower.copy_to(follower_source, destination_offset=0)
  try
    follower.accept_child(follower_source, nonempty_length, follower_limits)
  catch {
    ProtocolFault(Payload) => ()
    _ => fail("unexpected follower payload-completion error")
  } noraise {
    _ => fail("follower nonzero-payload completion was admitted")
  }
}

///|
test "rank-local terminal rejection and transport loss remain sticky" {
  let closed = ready_differential_rig()
  abort_differential_plan(closed, None, 0UL)
  terminal_differential_exchange(closed, Drain, Drained)
  terminal_differential_exchange(closed, Close, Closed)
  let extra = try! closed.sender.encode_control(
    closed.bindings[0],
    Drain,
    ParentToChild,
    None,
    0UL,
  )
  let extra_length = try! extra.length()
  let source = FixedArray::make(closed.sender.capacity(), b'\x00')
  try! extra.copy_to(source, destination_offset=0)
  try closed.group.accept(extra) catch {
    ProtocolFault(Lifecycle) => ()
    _ => fail("unexpected closed group rejection")
  } noraise {
    _ => fail("closed group accepted another command")
  }
  try
    closed.locals[0].accept_parent(source, extra_length, closed.limits)
  catch {
    ProtocolFault(Lifecycle) => ()
    _ => fail("unexpected closed local rejection")
  } noraise {
    _ => fail("closed local transcript accepted another command")
  }

  let (lost, lost_sender, lost_limits, lost_binding, lost_source) = ready_local_rank_fixture()
  lost.abandon_on_transport_loss()
  lost.abandon_on_transport_loss()
  assert_eq(lost.lifecycle(), Faulted)
  assert_eq(lost.failure_rule(), Some(TransportLost))
  let submit_payload = FixedArray::make(1, b'\x41')
  let submit = try! lost_sender.encode_payload(
    lost_binding,
    Submit,
    ParentToChild,
    A,
    1UL,
    submit_payload,
    1,
  )
  let submit_length = try! submit.length()
  try! submit.copy_to(lost_source, destination_offset=0)
  try lost.accept_parent(lost_source, submit_length, lost_limits) catch {
    TranscriptFaulted(TransportLost) => ()
    _ => fail("unexpected sticky transport-loss error")
  } noraise {
    _ => fail("transport-lost transcript remained usable")
  }
}
