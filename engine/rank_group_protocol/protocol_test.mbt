///|
test "minimum rank group preserves one leader and canonical completion" {
  let (plan_owner, plan) = fixture_plan(1UL, 0UL)
  let group = fixture_group(2, plan_owner)
  let submitted = group.begin(plan, started_tick=10UL, deadline_tick=20UL)
  group.accept_follower_ack(submitted, submitted.lease(plan, 1).follower_ack())
  assert_true(group.followers_complete(submitted))
  let (_, completion) = fixture_completion(plan)
  let completed = group.complete_leader(
    submitted.lease(plan, 0),
    plan,
    completion,
  )
  completed.validate_completion(plan, completion)
  assert_eq(completion.entry_count(), 1)
  group.retire_completed(completed)
  assert_eq(group.lifecycle(), Ready)
  assert_eq(group.previous_plan_sequence_value(), 1UL)
}

///|
test "world two publishes only after the follower acknowledgement" {
  let (plan_owner, plan) = fixture_plan(1UL, 0UL)
  let group = fixture_group(2, plan_owner)
  let submitted = group.begin(plan, started_tick=10UL, deadline_tick=20UL)
  let (_, completion) = fixture_completion(plan)
  try group.complete_leader(submitted.lease(plan, 0), plan, completion) catch {
    NotReady(FollowersPending) => ()
    _ => fail("expected follower barrier")
  } noraise {
    _ => fail("leader completed before follower")
  }
  let follower = submitted.lease(plan, 1)
  group.accept_follower_ack(submitted, follower.follower_ack())
  assert_true(group.followers_complete(submitted))
  let completed = group.complete_leader(
    submitted.lease(plan, 0),
    plan,
    completion,
  )
  completed.validate_completion(plan, completion)
  assert_eq(completion.sequence_wire(), 1UL)
}

///|
test "completed capability retires once and rejects foreign owners" {
  let (first_plan_owner, first_plan) = fixture_plan(1UL, 0UL)
  let first_group = fixture_group(2, first_plan_owner)
  let first_submitted = first_group.begin(
    first_plan,
    started_tick=10UL,
    deadline_tick=20UL,
  )
  let (_, first_completion) = fixture_completion(first_plan)
  first_group.accept_follower_ack(
    first_submitted,
    first_submitted.lease(first_plan, 1).follower_ack(),
  )
  let completed = first_group.complete_leader(
    first_submitted.lease(first_plan, 0),
    first_plan,
    first_completion,
  )

  let (foreign_plan_owner, _) = fixture_plan(1UL, 0UL)
  let foreign_group = fixture_group(2, foreign_plan_owner)
  try foreign_group.retire_completed(completed) catch {
    Invalid(Submission) => ()
    _ => fail("unexpected foreign retirement error")
  } noraise {
    _ => fail("foreign group retired another owner's completion")
  }
  assert_eq(foreign_group.lifecycle(), Ready)

  first_group.retire_completed(completed)
  assert_eq(first_group.lifecycle(), Ready)
  try first_group.retire_completed(completed) catch {
    Invalid(Submission) => ()
    _ => fail("unexpected duplicate retirement error")
  } noraise {
    _ => fail("completed capability retired twice")
  }
}

///|
test "deadline failure publishes once and drains every survivor" {
  let (plan_owner, plan) = fixture_plan(1UL, 0UL)
  let group = fixture_group(2, plan_owner)
  let submitted = group.begin(plan, started_tick=10UL, deadline_tick=20UL)
  assert_false(group.expire_if_due(submitted, 19UL))
  assert_true(group.expire_if_due(submitted, 20UL))
  let report = group.take_failure()
  assert_eq(report.reason(), DeadlineExpired)
  assert_eq(report.failed_rank(), -1)
  try group.take_failure() catch {
    Invalid(FailurePublication) => ()
    _ => fail("expected one-shot failure publication")
  } noraise {
    _ => fail("failure published twice")
  }
  for rank in 0..<2 {
    let lease = group.drain_lease(rank)
    group.accept_drain_ack(lease.ack())
  }
  assert_true(group.drain_complete())
  group.close_faulted()
  assert_eq(group.lifecycle(), Closed)
}

///|
test "begin host failure binds the exact intended plan sequence" {
  let (plan_owner, plan) = fixture_plan(1UL, 0UL)
  let group = fixture_group(2, plan_owner)
  group.report_begin_host_failure(plan)
  let report = group.take_failure()
  assert_eq(report.reason(), HostFailure)
  assert_eq(report.plan_sequence_value(), 1UL)
  assert_eq(report.failed_rank(), -1)
  assert_eq(group.lifecycle(), Faulted)
}

///|
test "active host failure retains the active plan sequence" {
  let (plan_owner, plan) = fixture_plan(1UL, 0UL)
  let group = fixture_group(2, plan_owner)
  ignore(group.begin(plan, started_tick=10UL, deadline_tick=20UL))
  group.report_active_host_failure()
  let report = group.take_failure()
  assert_eq(report.reason(), HostFailure)
  assert_eq(report.plan_sequence_value(), 1UL)
  assert_eq(report.failed_rank(), -1)
}

///|
test "two admitted reusable plan buffers preserve one exact sequence" {
  let (first_owner, first_plan) = fixture_plan(1UL, 0UL)
  let (second_owner, second_plan) = fixture_plan(2UL, 1UL)
  let group = @rank_group_protocol.RankGroupOwner::new(
    identity=fixture_identity(),
    model_plan_generation=@worker_protocol.ModelPlanGeneration::first(),
    generation=@rank_group_protocol.RankGroupGeneration::first(),
    world_size=2,
    previous_plan_sequence_value=0UL,
    plan_buffer_identities=[first_owner.identity(), second_owner.identity()],
  )
  let first = group.begin(first_plan, started_tick=1UL, deadline_tick=3UL)
  group.accept_follower_ack(first, first.lease(first_plan, 1).follower_ack())
  let (_, first_completion) = fixture_completion(first_plan)
  let first_done = group.complete_leader(
    first.lease(first_plan, 0),
    first_plan,
    first_completion,
  )
  group.retire_completed(first_done)
  let second = group.begin(second_plan, started_tick=4UL, deadline_tick=6UL)
  assert_eq(second.sequence_value(), 2UL)
  assert_eq(second.lease(second_plan, 0).rank(), 0)
}
