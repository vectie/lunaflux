///|
test "submitted capability is rederived only for the exact live plan" {
  let (plan_owner, plan) = fixture_plan(1UL, 0UL)
  let group = fixture_group(2, plan_owner)
  ignore(group.begin(plan, started_tick=10UL, deadline_tick=20UL))
  let rederived = group.reauthenticate_submitted(plan)
  assert_eq(rederived.sequence_value(), 1UL)

  let (_, foreign_plan) = fixture_plan(1UL, 0UL)
  try group.reauthenticate_submitted(foreign_plan) catch {
    ProtocolFault(Submission) => ()
    _ => fail("unexpected foreign reauthentication error")
  } noraise {
    _ => fail("foreign plan rederived a submitted capability")
  }
}

///|
test "completed capability is rederived from exact accepted epoch evidence" {
  let (plan_owner, plan) = fixture_plan(1UL, 0UL)
  let group = fixture_group(2, plan_owner)
  let submitted = group.begin(plan, started_tick=10UL, deadline_tick=20UL)
  group.accept_follower_ack(submitted, submitted.lease(plan, 1).follower_ack())
  let (_, completion) = fixture_completion(plan)
  ignore(group.complete_leader(submitted.lease(plan, 0), plan, completion))
  try group.reauthenticate_completed(2UL) catch {
    Invalid(Submission) => ()
    _ => fail("unexpected stale completed epoch error")
  } noraise {
    _ => fail("stale epoch rederived a completed capability")
  }
  let rederived = group.reauthenticate_completed(1UL)
  group.retire_completed(rederived)
  assert_eq(group.lifecycle(), Ready)
}
