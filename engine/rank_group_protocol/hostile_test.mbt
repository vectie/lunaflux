///|
test "duplicate follower acknowledgement permanently faults the generation" {
  let (plan_owner, plan) = fixture_plan(1UL, 0UL)
  let group = fixture_group(2, plan_owner)
  let submitted = group.begin(plan, started_tick=1UL, deadline_tick=2UL)
  let ack = submitted.lease(plan, 1).follower_ack()
  group.accept_follower_ack(submitted, ack)
  try group.accept_follower_ack(submitted, ack) catch {
    ProtocolFault(DuplicateFollowerAck) => ()
    _ => fail("expected duplicate ACK protocol fault")
  } noraise {
    _ => fail("duplicate ACK was accepted")
  }
  assert_eq(group.lifecycle(), Faulted)
  assert_eq(
    group.take_failure().reason(),
    ProtocolViolation(DuplicateFollowerAck),
  )
}

///|
test "scalar substitution faults before follower publication" {
  let (plan_owner, plan) = fixture_plan(1UL, 0UL)
  let group = fixture_group(2, plan_owner)
  let submitted = group.begin(plan, started_tick=1UL, deadline_tick=3UL)
  let forged = @rank_group_protocol.RankExecutionAck::from_wire(
    group_generation_value=1UL,
    plan_sequence_value=2UL,
    model_plan_generation_value=1UL,
    rank=1,
    world_size=2,
  )
  try group.accept_follower_ack(submitted, forged) catch {
    ProtocolFault(PlanSequence) => ()
    _ => fail("expected plan sequence substitution fault")
  } noraise {
    _ => fail("substituted ACK was accepted")
  }
  assert_eq(group.take_failure().reason(), ProtocolViolation(PlanSequence))
}

///|
test "rank loss excludes only the lost rank from drain" {
  let (plan_owner, plan) = fixture_plan(1UL, 0UL)
  let group = fixture_group(3, plan_owner)
  let submitted = group.begin(plan, started_tick=1UL, deadline_tick=3UL)
  group.report_rank_loss(submitted, 1)
  let report = group.take_failure()
  assert_eq(report.reason(), RankLost)
  assert_eq(report.failed_rank(), 1)
  try group.drain_lease(1) catch {
    Invalid(LostRankDrain) => ()
    _ => fail("expected lost-rank drain rejection")
  } noraise {
    _ => fail("lost rank received drain authority")
  }
  for rank in [0, 2] {
    let lease = group.drain_lease(rank)
    group.accept_drain_ack(lease.ack())
  }
  assert_true(group.drain_complete())
}

///|
test "additional rank loss cannot strand a faulted group drain" {
  let (plan_owner, plan) = fixture_plan(1UL, 0UL)
  let group = fixture_group(3, plan_owner)
  let submitted = group.begin(plan, started_tick=1UL, deadline_tick=3UL)
  group.report_collective_failure(submitted, 1)
  let report = group.take_failure()
  assert_eq(report.reason(), CollectiveFailure)
  assert_eq(report.failed_rank(), 1)
  group.mark_drain_rank_lost(0)
  group.mark_drain_rank_lost(1)
  let survivor = group.drain_lease(2)
  group.accept_drain_ack(survivor.ack())
  assert_true(group.drain_complete())
  try group.mark_drain_rank_lost(0) catch {
    Invalid(LostRankDrain) => ()
    _ => fail("expected already-lost rank rejection")
  } noraise {
    _ => fail("lost rank was marked twice")
  }
  group.close_faulted()
  assert_eq(group.lifecycle(), Closed)
}

///|
test "skipped first plan sequence faults a fresh group" {
  let (plan_owner, plan) = fixture_plan(2UL, 1UL)
  let group = fixture_group(2, plan_owner)
  try group.begin(plan, started_tick=1UL, deadline_tick=3UL) catch {
    ProtocolFault(PlanSequence) => ()
    _ => fail("expected exact next sequence rejection")
  } noraise {
    _ => fail("skipped sequence was accepted")
  }
  assert_eq(group.take_failure().reason(), ProtocolViolation(PlanSequence))
}

///|
test "rank-group owner rejects the single-rank worker domain" {
  let (plan_owner, _) = fixture_plan(1UL, 0UL)
  try
    @rank_group_protocol.RankGroupOwner::new(
      identity=fixture_identity(),
      model_plan_generation=@worker_protocol.ModelPlanGeneration::first(),
      generation=@rank_group_protocol.RankGroupGeneration::first(),
      world_size=1,
      previous_plan_sequence_value=0UL,
      plan_buffer_identities=[plan_owner.identity()],
    )
  catch {
    Invalid(WorldSize) => ()
    _ => fail("unexpected single-rank rejection")
  } noraise {
    _ => fail("rank group admitted the single-rank worker domain")
  }
}

///|
test "foreign plan owner is rejected even when scalar identities match" {
  let (admitted_owner, admitted) = fixture_plan(1UL, 0UL)
  let (_, foreign) = fixture_plan(1UL, 0UL)
  let group = fixture_group(2, admitted_owner)
  let submitted = group.begin(admitted, started_tick=1UL, deadline_tick=3UL)
  try submitted.lease(foreign, 1) catch {
    ProtocolFault(Submission) => ()
    _ => fail("expected exact plan-buffer identity rejection")
  } noraise {
    _ => fail("foreign plan owner was accepted")
  }
}

///|
test "idle rank loss publishes exact sequence-zero evidence once" {
  let (plan_owner, plan) = fixture_plan(1UL, 0UL)
  let ready = fixture_group(2, plan_owner)
  ready.report_idle_rank_loss(1)
  let report = ready.take_failure()
  assert_eq(report.reason(), RankLost)
  assert_eq(report.failed_rank(), 1)
  assert_eq(report.plan_sequence_value(), 0UL)
  try ready.take_failure() catch {
    Invalid(FailurePublication) => ()
    _ => fail("unexpected duplicate idle publication error")
  } noraise {
    _ => fail("idle failure published twice")
  }

  let executing = fixture_group(2, plan_owner)
  ignore(executing.begin(plan, started_tick=1UL, deadline_tick=3UL))
  try executing.report_idle_rank_loss(0) catch {
    InvalidLifecycle(expected=Ready, actual=Executing) => ()
    _ => fail("unexpected executing idle-loss error")
  } noraise {
    _ => fail("idle loss was admitted during execution")
  }

  let faulted = fixture_group(2, plan_owner)
  faulted.report_idle_rank_loss(0)
  try faulted.report_idle_rank_loss(1) catch {
    InvalidLifecycle(expected=Ready, actual=Faulted) => ()
    _ => fail("unexpected faulted idle-loss error")
  } noraise {
    _ => fail("idle loss was admitted after fault")
  }

  let closed = fixture_group(2, plan_owner)
  closed.close()
  try closed.report_idle_rank_loss(0) catch {
    InvalidLifecycle(expected=Ready, actual=Closed) => ()
    _ => fail("unexpected closed idle-loss error")
  } noraise {
    _ => fail("idle loss was admitted after close")
  }
}
