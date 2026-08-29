///|
fn polling_control_owner(
  rank? : Int = 0,
) -> (RankChildControl, @rank_group_wire.RankGroupWireBinding, RankChildInbound) raise {
  let (owner, binding) = ready_control_owner(rank~)
  let payload = FixedArray::make(1, b'\x41')
  let submit = publish_parent_payload(owner, binding, Submit, A, 1UL, payload)
  try! owner.accept_submit(submit)
  let poll = publish_parent_control(owner, binding, PollComplete, Some(A), 1UL)
  (owner, binding, poll)
}

///|
test "rank child emits distinct payload-free plan failure categories" {
  let (leader, leader_binding, leader_poll) = polling_control_owner()
  try! leader.begin_collective_failed(leader_poll)
  assert_child_response(
    leader,
    leader_binding,
    PlanFailed(CollectiveFailure),
    Some(A),
    1UL,
    0,
  )
  assert_true(leader.send_active)
  assert_false(leader.inbound_active)
  assert_eq(leader.lifecycle(), AbortedPlan)
  assert_eq(leader.transcript.plan_failure(), Some(CollectiveFailure))
  assert_eq(leader.transcript.previous_sequence_value(), 1UL)
  assert_eq(leader.transcript.active_sequence_value(), 0UL)

  let (follower, follower_binding, follower_poll) = polling_control_owner(
    rank=1,
  )
  try! follower.begin_execution_failed(follower_poll)
  assert_child_response(
    follower,
    follower_binding,
    PlanFailed(RankExecutionFailure),
    Some(A),
    1UL,
    0,
  )
  assert_true(follower.send_active)
  assert_false(follower.inbound_active)
  assert_eq(follower.lifecycle(), AbortedPlan)
  assert_eq(follower.transcript.plan_failure(), Some(RankExecutionFailure))
  assert_eq(follower.transcript.previous_sequence_value(), 1UL)
  assert_eq(follower.transcript.active_sequence_value(), 0UL)
}

///|
test "rank child failure requires the exact live PollComplete capability" {
  let (wrong_state, binding) = ready_control_owner()
  let payload = FixedArray::make(1, b'\x41')
  let submit = publish_parent_payload(
    wrong_state,
    binding,
    Submit,
    A,
    1UL,
    payload,
  )
  try wrong_state.begin_collective_failed(submit) catch {
    Invalid(Lifecycle) => ()
    _ => fail("unexpected wrong-kind failure response error")
  } noraise {
    _ => fail("Submit capability authorized a failure response")
  }
  assert_true(wrong_state.inbound_active)
  assert_false(wrong_state.send_active)
  assert_eq(wrong_state.lifecycle(), Executing)

  let (first, _, poll) = polling_control_owner()
  try! first.begin_execution_failed(poll)
  try first.begin_execution_failed(poll) catch {
    Invalid(Epoch) => ()
    _ => fail("unexpected duplicate failure response error")
  } noraise {
    _ => fail("consumed poll authorized duplicate failure")
  }
  assert_eq(first.lifecycle(), AbortedPlan)
  assert_true(first.send_active)

  let (other, _, other_poll) = polling_control_owner(rank=1)
  let (target, _) = ready_control_owner()
  try target.begin_collective_failed(other_poll) catch {
    Invalid(Lifecycle) => ()
    _ => fail("unexpected foreign-poll failure response error")
  } noraise {
    _ => fail("foreign rank owner authorized a failure response")
  }
  assert_true(other.inbound_active)
  assert_false(target.send_active)
}

///|
test "rank child failed plan cannot be substituted with Complete" {
  let (leader, _, poll) = polling_control_owner()
  try! leader.begin_collective_failed(poll)
  let completion = FixedArray::make(1, b'\x61')
  try leader.begin_leader_complete(poll, completion, payload_length=1) catch {
    Invalid(Epoch) => ()
    _ => fail("unexpected post-failure Complete error")
  } noraise {
    _ => fail("failed plan was reused as leader completion")
  }
  assert_eq(leader.lifecycle(), AbortedPlan)
  assert_eq(leader.transcript.previous_sequence_value(), 1UL)
}

///|
test "unconsumed PollComplete prevents receiving a terminal cut" {
  let (owner, _, poll) = polling_control_owner()
  try owner.begin_receive() catch {
    Invalid(Lifecycle) => ()
    _ => fail("unexpected receive-with-live-poll error")
  } noraise {
    _ => fail("new receive bypassed an unconsumed PollComplete")
  }
  assert_true(owner.inbound_active)
  assert_false(owner.receive_active)
  assert_eq(owner.lifecycle(), Polling)
  assert_eq(owner.transcript.previous_sequence_value(), 0UL)
  assert_eq(try! poll.kind(), PollComplete)
}

///|
test "held Drain waits for TX then advances failed predecessor once" {
  let (owner, binding, poll) = polling_control_owner()
  owner.consume_inbound()
  ignore(poll)
  advance_child_control(
    owner,
    binding,
    PlanFailed(CollectiveFailure),
    Some(A),
    1UL,
  )
  owner.send_active = true
  let drain_length = try! @rank_group_wire.encode_rank_group_wire_control_into(
    owner.rx_destination,
    owner.wire_limits,
    binding,
    Drain,
    ParentToChild,
    None,
    0UL,
  )
  owner.receive_held = true
  owner.receive_held_eof = false
  owner.receive_held_length = drain_length
  let held = try! owner.progress_receive()
  assert_true(held.is_held())
  assert_false(owner.inbound_active)
  assert_eq(owner.lifecycle(), AbortedPlan)
  assert_eq(owner.transcript.previous_sequence_value(), 1UL)

  // This transcript-only white-box fixture has no native transfer. Release its
  // simulated response ownership before the held Drain may publish.
  owner.send_active = false
  let published = try! owner.progress_receive()
  assert_true(published.has_inbound())
  assert_eq(owner.lifecycle(), Draining)
  assert_eq(owner.transcript.previous_sequence_value(), 1UL)
  let drain = try! owner.inbound()
  owner.consume_inbound()
  ignore(drain)
  advance_child_control(owner, binding, Drained, None, 0UL)
  owner.send_active = true
  assert_child_response(owner, binding, Drained, None, 0UL, 0)
  assert_eq(owner.lifecycle(), DrainedGroup)
  assert_eq(owner.transcript.previous_sequence_value(), 1UL)

  owner.send_active = false
  let close = publish_parent_control(owner, binding, Close, None, 0UL)
  owner.consume_inbound()
  ignore(close)
  advance_child_control(owner, binding, Closed, None, 0UL)
  owner.send_active = true
  assert_child_response(owner, binding, Closed, None, 0UL, 0)
  assert_eq(owner.lifecycle(), ClosedGroup)
  assert_eq(owner.transcript.previous_sequence_value(), 1UL)
}

///|
test "held Drain cannot cut an executing survivor before TX completion" {
  let (owner, binding, poll) = polling_control_owner(rank=1)
  try! owner.begin_pending(poll)
  assert_eq(owner.lifecycle(), Executing)
  assert_eq(owner.transcript.previous_sequence_value(), 0UL)
  let drain_length = try! @rank_group_wire.encode_rank_group_wire_control_into(
    owner.rx_destination,
    owner.wire_limits,
    binding,
    Drain,
    ParentToChild,
    None,
    0UL,
  )
  owner.receive_held = true
  owner.receive_held_eof = false
  owner.receive_held_length = drain_length
  let held = try! owner.progress_receive()
  assert_true(held.is_held())
  assert_eq(owner.lifecycle(), Executing)
  assert_eq(owner.transcript.previous_sequence_value(), 0UL)

  owner.send_active = false
  let published = try! owner.progress_receive()
  assert_true(published.has_inbound())
  assert_eq(owner.lifecycle(), Draining)
  assert_eq(owner.transcript.previous_sequence_value(), 1UL)
  assert_eq(owner.transcript.active_sequence_value(), 0UL)
}
