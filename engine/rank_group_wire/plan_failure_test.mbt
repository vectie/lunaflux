///|
fn exercise_differential_plan_failure(
  reporting_rank : Int,
  failure : @rank_group_wire.RankGroupWirePlanFailure,
) -> Unit raise {
  let rig = ready_differential_rig()
  submit_differential_plan(rig, 1UL, A)
  accept_rank_and_group(
    rig.group,
    rig.locals[reporting_rank],
    rig.sender,
    rig.limits,
    rig.bindings[reporting_rank],
    PollComplete,
    ParentToChild,
    Some(A),
    1UL,
    None,
  )
  accept_rank_and_group(
    rig.group,
    rig.locals[reporting_rank],
    rig.sender,
    rig.limits,
    rig.bindings[reporting_rank],
    PlanFailed(failure),
    ChildToParent,
    Some(A),
    1UL,
    None,
  )
  assert_eq(rig.group.lifecycle(), AbortedPlan)
  assert_eq(rig.group.plan_failure(), Some(failure))
  assert_eq(rig.group.plan_failure_rank(), Some(reporting_rank))
  assert_eq(rig.group.previous_sequence_value(), 1UL)
  assert_eq(rig.group.active_sequence_value(), 0UL)
  assert_eq(rig.locals[reporting_rank].lifecycle(), AbortedPlan)
  assert_eq(rig.locals[reporting_rank].plan_failure(), Some(failure))
  assert_eq(rig.locals[reporting_rank].previous_sequence_value(), 1UL)

  // The reporting rank has retired the plan. Surviving ranks learn the same
  // terminal cut from Drain without fabricating an Aborted response.
  let surviving_rank = 1 - reporting_rank
  assert_eq(rig.locals[surviving_rank].lifecycle(), Executing)
  assert_eq(rig.locals[surviving_rank].previous_sequence_value(), 0UL)
  for rank in 0..<2 {
    accept_rank_and_group(
      rig.group,
      rig.locals[rank],
      rig.sender,
      rig.limits,
      rig.bindings[rank],
      Drain,
      ParentToChild,
      None,
      0UL,
      None,
    )
  }
  assert_eq(rig.group.lifecycle(), Draining)
  for rank_transcript in rig.locals {
    assert_eq(rank_transcript.lifecycle(), Draining)
    assert_eq(rank_transcript.previous_sequence_value(), 1UL)
    assert_eq(rank_transcript.active_sequence_value(), 0UL)
  }
  for rank in 0..<2 {
    accept_rank_and_group(
      rig.group,
      rig.locals[rank],
      rig.sender,
      rig.limits,
      rig.bindings[rank],
      Drained,
      ChildToParent,
      None,
      0UL,
      None,
    )
  }
  assert_differential_phase(rig, DrainedGroup, 1UL, 0UL)
  terminal_differential_exchange(rig, Close, Closed)
  assert_differential_phase(rig, ClosedGroup, 1UL, 0UL)
  assert_eq(rig.group.plan_failure(), Some(failure))
  assert_eq(rig.group.plan_failure_rank(), Some(reporting_rank))
}

///|
test "plan failures are differential for leader and follower recovery" {
  exercise_differential_plan_failure(0, CollectiveFailure)
  exercise_differential_plan_failure(1, RankExecutionFailure)
}

///|
fn accept_local_failure(
  rank_transcript : @rank_group_wire.RankGroupWireRankTranscript,
  sender : @rank_group_wire.RankGroupWireFrameBuffer,
  limits : @rank_group_wire.RankGroupWireLimits,
  binding : @rank_group_wire.RankGroupWireBinding,
  source : FixedArray[Byte],
  kind : @rank_group_wire.RankGroupWireKind,
  slot : @rank_group_wire.RankGroupWireSlot,
  sequence_value : UInt64,
) -> Unit raise @rank_group_wire.RankGroupWireError {
  let frame = try! sender.encode_control(
    binding,
    kind,
    ChildToParent,
    Some(slot),
    sequence_value,
  )
  let length = try! frame.length()
  try! frame.copy_to(source, destination_offset=0)
  ignore(rank_transcript.accept_child(source, length, limits))
}

///|
test "plan failure wire shape rejects direction and payload substitution" {
  let limits = fixture_wire_limits()
  let binding = fixture_wire_binding(0)
  let sender = @rank_group_wire.RankGroupWireFrameBuffer::new(limits)
  try
    sender.encode_control(
      binding,
      PlanFailed(CollectiveFailure),
      ParentToChild,
      Some(A),
      1UL,
    )
  catch {
    Invalid(Direction) => ()
    _ => fail("unexpected plan-failure direction error")
  } noraise {
    _ => fail("parent-to-child plan failure was encoded")
  }
  let payload = FixedArray::make(1, b'\x71')
  try
    sender.encode_payload(
      binding,
      PlanFailed(RankExecutionFailure),
      ChildToParent,
      A,
      1UL,
      payload,
      1,
    )
  catch {
    Invalid(Payload) => ()
    _ => fail("unexpected plan-failure payload error")
  } noraise {
    _ => fail("payload-bearing plan failure was encoded")
  }
}

///|
test "plan failure rejects binding rank slot and sequence substitution" {
  let (rank_changed, sender, limits, _, source) = polling_local_rank(0)
  try
    accept_local_failure(
      rank_changed,
      sender,
      limits,
      fixture_wire_binding(1),
      source,
      PlanFailed(CollectiveFailure),
      A,
      1UL,
    )
  catch {
    ProtocolFault(Binding) => ()
    _ => fail("unexpected plan-failure rank substitution error")
  } noraise {
    _ => fail("plan failure from a substituted rank was admitted")
  }

  let (binding_changed, binding_sender, binding_limits, _, binding_source) = polling_local_rank(
    0,
  )
  try
    accept_local_failure(
      binding_changed,
      binding_sender,
      binding_limits,
      fixture_wire_binding(0, group_generation_value=2UL),
      binding_source,
      PlanFailed(CollectiveFailure),
      A,
      1UL,
    )
  catch {
    ProtocolFault(Binding) => ()
    _ => fail("unexpected plan-failure binding substitution error")
  } noraise {
    _ => fail("plan failure from a substituted binding was admitted")
  }

  let (slot_changed, slot_sender, slot_limits, slot_binding, slot_source) = polling_local_rank(
    0,
  )
  try
    accept_local_failure(
      slot_changed,
      slot_sender,
      slot_limits,
      slot_binding,
      slot_source,
      PlanFailed(CollectiveFailure),
      B,
      1UL,
    )
  catch {
    ProtocolFault(Sequence) => ()
    _ => fail("unexpected plan-failure slot substitution error")
  } noraise {
    _ => fail("plan failure for a substituted slot was admitted")
  }

  let (
    sequence_changed,
    sequence_sender,
    sequence_limits,
    sequence_binding,
    sequence_source,
  ) = polling_local_rank(1)
  try
    accept_local_failure(
      sequence_changed,
      sequence_sender,
      sequence_limits,
      sequence_binding,
      sequence_source,
      PlanFailed(RankExecutionFailure),
      A,
      2UL,
    )
  catch {
    ProtocolFault(Sequence) => ()
    _ => fail("unexpected plan-failure sequence substitution error")
  } noraise {
    _ => fail("plan failure for a substituted sequence was admitted")
  }
}

///|
test "plan failure is polling-only fail-stop and cannot become Complete" {
  let (early, sender, limits, binding, source) = ready_local_rank_fixture()
  try
    accept_local_failure(
      early,
      sender,
      limits,
      binding,
      source,
      PlanFailed(CollectiveFailure),
      A,
      1UL,
    )
  catch {
    ProtocolFault(Lifecycle) => ()
    _ => fail("unexpected pre-poll plan-failure error")
  } noraise {
    _ => fail("plan failure was admitted before PollComplete")
  }

  let (failed, failed_sender, failed_limits, failed_binding, failed_source) = polling_local_rank(
    0,
  )
  try! accept_local_failure(
    failed,
    failed_sender,
    failed_limits,
    failed_binding,
    failed_source,
    PlanFailed(RankExecutionFailure),
    A,
    1UL,
  )
  assert_eq(failed.lifecycle(), AbortedPlan)
  assert_eq(failed.plan_failure(), Some(RankExecutionFailure))
  try
    accept_local_failure(
      failed,
      failed_sender,
      failed_limits,
      failed_binding,
      failed_source,
      PlanFailed(RankExecutionFailure),
      A,
      1UL,
    )
  catch {
    ProtocolFault(Lifecycle) => ()
    _ => fail("unexpected duplicate plan-failure error")
  } noraise {
    _ => fail("duplicate plan failure was admitted")
  }

  let (
    substituted,
    complete_sender,
    complete_limits,
    complete_binding,
    complete_source,
  ) = polling_local_rank(0)
  try! accept_local_failure(
    substituted,
    complete_sender,
    complete_limits,
    complete_binding,
    complete_source,
    PlanFailed(CollectiveFailure),
    A,
    1UL,
  )
  let payload = FixedArray::make(1, b'\x61')
  let complete = try! complete_sender.encode_payload(
    complete_binding,
    Complete,
    ChildToParent,
    A,
    1UL,
    payload,
    1,
  )
  let length = try! complete.length()
  try! complete.copy_to(complete_source, destination_offset=0)
  try substituted.accept_child(complete_source, length, complete_limits) catch {
    ProtocolFault(Lifecycle) => ()
    _ => fail("unexpected post-failure Complete substitution error")
  } noraise {
    _ => fail("failed plan was reused as successful completion")
  }
}

///|
fn polling_group_for_failure() -> (
  @rank_group_wire.RankGroupWireTranscript,
  @rank_group_wire.RankGroupWireFrameBuffer,
  Array[@rank_group_wire.RankGroupWireBinding],
) {
  let (group, sender, bindings) = ready_wire_transcript()
  let payload = FixedArray::make(1, b'\x41')
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
    try! group.accept(submit)
  }
  accept_wire_control(
    group,
    sender,
    bindings[0],
    PollComplete,
    ParentToChild,
    Some(A),
    1UL,
  )
  (group, sender, bindings)
}

///|
test "group plan failure is polling-only sticky and excludes Complete" {
  let (early, early_sender, early_bindings) = ready_wire_transcript()
  let early_failure = try! early_sender.encode_control(
    early_bindings[0],
    PlanFailed(CollectiveFailure),
    ChildToParent,
    Some(A),
    1UL,
  )
  try early.accept(early_failure) catch {
    ProtocolFault(Lifecycle) => ()
    _ => fail("unexpected group pre-poll plan-failure error")
  } noraise {
    _ => fail("group admitted plan failure before PollComplete")
  }

  let (duplicate, duplicate_sender, duplicate_bindings) = polling_group_for_failure()
  let failed = try! duplicate_sender.encode_control(
    duplicate_bindings[0],
    PlanFailed(RankExecutionFailure),
    ChildToParent,
    Some(A),
    1UL,
  )
  try! duplicate.accept(failed)
  assert_eq(duplicate.lifecycle(), AbortedPlan)
  assert_eq(duplicate.plan_failure(), Some(RankExecutionFailure))
  try duplicate.accept(failed) catch {
    ProtocolFault(Lifecycle) => ()
    _ => fail("unexpected group duplicate plan-failure error")
  } noraise {
    _ => fail("group admitted duplicate plan failure")
  }
  assert_eq(duplicate.lifecycle(), Faulted)

  let (substituted, complete_sender, complete_bindings) = polling_group_for_failure()
  let plan_failed = try! complete_sender.encode_control(
    complete_bindings[0],
    PlanFailed(CollectiveFailure),
    ChildToParent,
    Some(A),
    1UL,
  )
  try! substituted.accept(plan_failed)
  let payload = FixedArray::make(1, b'\x61')
  let complete = try! complete_sender.encode_payload(
    complete_bindings[0],
    Complete,
    ChildToParent,
    A,
    1UL,
    payload,
    1,
  )
  try substituted.accept(complete) catch {
    ProtocolFault(Lifecycle) => ()
    _ => fail("unexpected group post-failure Complete error")
  } noraise {
    _ => fail("group reused failed plan as Complete")
  }
  assert_eq(substituted.lifecycle(), Faulted)
}
