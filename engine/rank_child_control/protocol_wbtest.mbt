///|
fn ready_control_owner(
  rank? : Int = 0,
) -> (RankChildControl, @rank_group_wire.RankGroupWireBinding) raise {
  let owner = control_test_owner()
  let binding = control_test_binding(rank~)
  let configure = publish_test_configure(owner, binding)
  try! owner.transcript.admit_ready(binding, previous_sequence_value=0UL)
  let ready_length = try! @rank_group_wire.encode_rank_group_wire_control_into(
    owner.tx_storage,
    owner.wire_limits,
    binding,
    Ready,
    ChildToParent,
    None,
    0UL,
  )
  ignore(
    try! owner.transcript.accept_child(
      owner.tx_storage,
      ready_length,
      owner.wire_limits,
    ),
  )
  owner.consume_inbound()
  ignore(configure)
  (owner, binding)
}

///|
fn publish_parent_control(
  owner : RankChildControl,
  binding : @rank_group_wire.RankGroupWireBinding,
  kind : @rank_group_wire.RankGroupWireKind,
  slot : @rank_group_wire.RankGroupWireSlot?,
  sequence_value : UInt64,
) -> RankChildInbound raise {
  let length = try! @rank_group_wire.encode_rank_group_wire_control_into(
    owner.rx_destination,
    owner.wire_limits,
    binding,
    kind,
    ParentToChild,
    slot,
    sequence_value,
  )
  let progress = try! owner.publish_received(length)
  assert_true(progress.has_inbound())
  try! owner.inbound()
}

///|
fn publish_parent_payload(
  owner : RankChildControl,
  binding : @rank_group_wire.RankGroupWireBinding,
  kind : @rank_group_wire.RankGroupWireKind,
  slot : @rank_group_wire.RankGroupWireSlot,
  sequence_value : UInt64,
  payload : FixedArray[Byte],
) -> RankChildInbound raise {
  let length = try! @rank_group_wire.encode_rank_group_wire_payload_into(
    owner.rx_destination,
    owner.wire_limits,
    binding,
    kind,
    ParentToChild,
    Some(slot),
    sequence_value,
    payload,
    payload.length(),
  )
  let progress = try! owner.publish_received(length)
  assert_true(progress.has_inbound())
  try! owner.inbound()
}

///|
fn advance_child_control(
  owner : RankChildControl,
  binding : @rank_group_wire.RankGroupWireBinding,
  kind : @rank_group_wire.RankGroupWireKind,
  slot : @rank_group_wire.RankGroupWireSlot?,
  sequence_value : UInt64,
) -> Unit {
  let length = try! @rank_group_wire.encode_rank_group_wire_control_into(
    owner.tx_storage,
    owner.wire_limits,
    binding,
    kind,
    ChildToParent,
    slot,
    sequence_value,
  )
  ignore(
    try! owner.transcript.accept_child(
      owner.tx_storage,
      length,
      owner.wire_limits,
    ),
  )
}

///|
fn advance_inactive_abort(
  owner : RankChildControl,
  binding : @rank_group_wire.RankGroupWireBinding,
) -> Unit raise {
  let abort_frame = publish_parent_control(owner, binding, Abort, None, 0UL)
  owner.consume_inbound()
  ignore(abort_frame)
  advance_child_control(owner, binding, Aborted, None, 0UL)
}

///|
fn advance_drained(
  owner : RankChildControl,
  binding : @rank_group_wire.RankGroupWireBinding,
) -> Unit raise {
  advance_inactive_abort(owner, binding)
  let drain = publish_parent_control(owner, binding, Drain, None, 0UL)
  owner.consume_inbound()
  ignore(drain)
  advance_child_control(owner, binding, Drained, None, 0UL)
}

///|
fn assert_child_response(
  owner : RankChildControl,
  binding : @rank_group_wire.RankGroupWireBinding,
  kind : @rank_group_wire.RankGroupWireKind,
  slot : @rank_group_wire.RankGroupWireSlot?,
  sequence_value : UInt64,
  payload_length : Int,
) -> Unit raise {
  let length = @rank_group_wire.rank_group_wire_header_bytes() + payload_length
  let header = try! @rank_group_wire.validate_rank_group_wire_binding(
    owner.tx_storage,
    length,
    owner.wire_limits,
    binding,
  )
  assert_eq(header.kind(), kind)
  assert_eq(header.direction(), ChildToParent)
  assert_eq(header.slot(), slot)
  assert_eq(header.sequence_value(), sequence_value)
  assert_eq(header.payload_length(), payload_length)
}

///|
test "rank child white-box covers Submit Poll Pending and leader Complete" {
  let (pending_owner, pending_binding) = ready_control_owner()
  let payload = FixedArray::make(3, b'\x41')
  let submitted = publish_parent_payload(
    pending_owner,
    pending_binding,
    Submit,
    A,
    1UL,
    payload,
  )
  try! pending_owner.accept_submit(submitted)
  let first_poll = publish_parent_control(
    pending_owner,
    pending_binding,
    PollComplete,
    Some(A),
    1UL,
  )
  try! pending_owner.begin_pending(first_poll)
  assert_child_response(
    pending_owner,
    pending_binding,
    Pending,
    Some(A),
    1UL,
    0,
  )
  assert_eq(pending_owner.lifecycle(), Executing)

  let (complete_owner, complete_binding) = ready_control_owner()
  let complete_submit = publish_parent_payload(
    complete_owner,
    complete_binding,
    Submit,
    A,
    1UL,
    payload,
  )
  try! complete_owner.accept_submit(complete_submit)
  let pending_poll = publish_parent_control(
    complete_owner,
    complete_binding,
    PollComplete,
    Some(A),
    1UL,
  )
  complete_owner.consume_inbound()
  ignore(pending_poll)
  advance_child_control(complete_owner, complete_binding, Pending, Some(A), 1UL)
  let final_poll = publish_parent_control(
    complete_owner,
    complete_binding,
    PollComplete,
    Some(A),
    1UL,
  )
  let completion = FixedArray::make(2, b'\x61')
  try! complete_owner.begin_leader_complete(
    final_poll,
    completion,
    payload_length=2,
  )
  assert_child_response(
    complete_owner,
    complete_binding,
    Complete,
    Some(A),
    1UL,
    2,
  )
  assert_eq(complete_owner.lifecycle(), ReadyGroup)
}

///|
test "rank child white-box covers follower completion and rank rejection" {
  let (follower, binding) = ready_control_owner(rank=1)
  let payload = FixedArray::make(1, b'\x41')
  try! follower.accept_submit(
    publish_parent_payload(follower, binding, Submit, A, 1UL, payload),
  )
  let poll = publish_parent_control(
    follower,
    binding,
    PollComplete,
    Some(A),
    1UL,
  )
  try! follower.begin_follower_complete(poll)
  assert_child_response(follower, binding, Complete, Some(A), 1UL, 0)
  assert_eq(follower.lifecycle(), ReadyGroup)

  let (leader, leader_binding) = ready_control_owner()
  try! leader.accept_submit(
    publish_parent_payload(leader, leader_binding, Submit, A, 1UL, payload),
  )
  let leader_poll = publish_parent_control(
    leader,
    leader_binding,
    PollComplete,
    Some(A),
    1UL,
  )
  try leader.begin_follower_complete(leader_poll) catch {
    Invalid(Payload) => ()
    _ => fail("unexpected leader/follower role error")
  } noraise {
    _ => fail("leader used follower completion authority")
  }
  assert_false(leader.send_active)
  assert_true(leader.inbound_active)
}

///|
test "rank child white-box covers Abort Drained and Closed responses" {
  let (aborting, abort_binding) = ready_control_owner()
  let abort_frame = publish_parent_control(
    aborting,
    abort_binding,
    Abort,
    None,
    0UL,
  )
  try! aborting.begin_aborted(abort_frame)
  assert_child_response(aborting, abort_binding, Aborted, None, 0UL, 0)
  assert_eq(aborting.lifecycle(), AbortedPlan)

  let (draining, drain_binding) = ready_control_owner()
  advance_inactive_abort(draining, drain_binding)
  let drain = publish_parent_control(draining, drain_binding, Drain, None, 0UL)
  try! draining.begin_drained(drain)
  assert_child_response(draining, drain_binding, Drained, None, 0UL, 0)
  assert_eq(draining.lifecycle(), DrainedGroup)

  let (closing, close_binding) = ready_control_owner()
  advance_drained(closing, close_binding)
  let close = publish_parent_control(closing, close_binding, Close, None, 0UL)
  try! closing.begin_closed(close)
  assert_child_response(closing, close_binding, Closed, None, 0UL, 0)
  assert_eq(closing.lifecycle(), ClosedGroup)
}

///|
test "rank child held RX publishes only after TX completion" {
  let (owner, binding) = ready_control_owner()
  let payload = FixedArray::make(1, b'\x41')
  let length = try! @rank_group_wire.encode_rank_group_wire_payload_into(
    owner.rx_destination,
    owner.wire_limits,
    binding,
    Submit,
    ParentToChild,
    Some(A),
    1UL,
    payload,
    1,
  )
  owner.receive_held = true
  owner.receive_held_eof = false
  owner.receive_held_length = length
  owner.send_active = true
  let held = try! owner.progress_receive()
  assert_true(held.is_held())
  assert_false(owner.inbound_active)
  owner.send_active = false
  let published = try! owner.progress_receive()
  assert_true(published.has_inbound())
  let inbound = try! owner.inbound()
  assert_eq(try! inbound.kind(), Submit)
  try! owner.accept_submit(inbound)
}

///|
test "rank child held EOF is clean only after Closed and otherwise sticky" {
  let (closed, closed_binding) = ready_control_owner()
  advance_drained(closed, closed_binding)
  let close = publish_parent_control(closed, closed_binding, Close, None, 0UL)
  closed.consume_inbound()
  ignore(close)
  advance_child_control(closed, closed_binding, Closed, None, 0UL)
  closed.receive_held = true
  closed.receive_held_eof = true
  closed.receive_held_length = 0
  let eof = try! closed.progress_receive()
  assert_true(eof.is_clean_eof())
  assert_false(closed.is_failed())

  let (early, _) = ready_control_owner()
  early.receive_held = true
  early.receive_held_eof = true
  early.receive_held_length = 0
  try early.progress_receive() catch {
    Failed(Transport) => ()
    _ => fail("unexpected early EOF error")
  } noraise {
    _ => fail("early EOF was accepted")
  }
  try early.progress_receive() catch {
    Failed(Transport) => ()
    _ => fail("early EOF failure was not sticky")
  } noraise {
    _ => fail("transport-faulted owner remained usable")
  }
}
