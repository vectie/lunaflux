///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "progress rejects a different admitted A B plan without changing scalars" {
  let plan_a_owner = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let plan_b_owner = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let PreparedRankGroupProcess(supervisor) = ownership_prepared_for(
    plan_a_owner, plan_b_owner,
  )
  let plan_a = ownership_plan(plan_a_owner)
  let plan_b = ownership_plan(plan_b_owner)
  ignore(ownership_prime_active(supervisor, plan_a, 0x7ffffffffffff000UL))
  supervisor.lifecycle_value = Polling
  try supervisor.progress(plan_b) catch {
    Invalid(Submission) => ()
    _ => fail("unexpected foreign active plan rejection")
  } noraise {
    _ => fail("different A B plan progressed the live exchange")
  }
  assert_eq(supervisor.lifecycle(), Polling)
  assert_eq(supervisor.protocol.lifecycle(), Executing)
  assert_eq(supervisor.active_slot_code, 0)
  assert_eq(supervisor.active_sequence_value, 1UL)
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "active poll frame deadline is the exact earlier plan cut" {
  let plan_a_owner = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let plan_b_owner = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let PreparedRankGroupProcess(supervisor) = ownership_prepared_for(
    plan_a_owner, plan_b_owner,
  )
  let plan = ownership_plan(plan_a_owner)
  let now = supervisor.clock.now_millis()
  let deadline = now + 500UL
  ignore(ownership_prime_active(supervisor, plan, deadline))
  supervisor.lifecycle_value = Polling
  assert_true(supervisor.prepare_active_frame_deadline(plan))
  assert_eq(
    supervisor.active_frame_deadline_millis,
    deadline.reinterpret_as_int64(),
  )
  assert_eq(supervisor.protocol.lifecycle(), Executing)
}

///|
fn ownership_prime_accepted_completion(
  supervisor : RankGroupProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> AcceptedRankGroupCompletion raise {
  let submitted = ownership_prime_active(supervisor, plan, 10UL)
  supervisor.protocol.accept_follower_ack(
    submitted,
    submitted.lease(plan, 1).follower_ack(),
  )
  supervisor.lifecycle_value = CompletionReady
  supervisor.accept_completion(
    supervisor.received_completion(),
    plan,
    ownership_completion(plan),
  )
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "accepted completion scalar reauthentication rejects every stale state" {
  let pre_owner = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let pre_b = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let PreparedRankGroupProcess(pre_accept) = ownership_prepared_for(
    pre_owner, pre_b,
  )
  let pre_plan = ownership_plan(pre_owner)
  ignore(ownership_prime_active(pre_accept, pre_plan, 10UL))
  pre_accept.lifecycle_value = CompletionReady
  try pre_accept.accepted_completion_for_sequence(1UL) catch {
    Invalid(Lifecycle) => ()
    _ => fail("unexpected pre-accept scalar rejection")
  } noraise {
    _ => fail("pre-accept scalar minted an accepted capability")
  }

  let exact_owner = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let exact_b = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let PreparedRankGroupProcess(exact) = ownership_prepared_for(
    exact_owner, exact_b,
  )
  let exact_plan = ownership_plan(exact_owner)
  ignore(ownership_prime_accepted_completion(exact, exact_plan))
  let reauthenticated = exact.accepted_completion_for_sequence(1UL)
  for wrong in [0UL, 2UL] {
    try exact.accepted_completion_for_sequence(wrong) catch {
      Invalid(Owner) => ()
      _ => fail("unexpected substituted-sequence rejection")
    } noraise {
      _ => fail("substituted sequence minted an accepted capability")
    }
  }
  exact.retire_after_scheduler_commit(reauthenticated)
  try exact.accepted_completion_for_sequence(1UL) catch {
    Invalid(Lifecycle) => ()
    _ => fail("unexpected post-retire scalar rejection")
  } noraise {
    _ => fail("post-retire scalar minted an accepted capability")
  }

  let stale_owner = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let stale_b = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let PreparedRankGroupProcess(stale) = ownership_prepared_for(
    stale_owner, stale_b,
  )
  let stale_plan = ownership_plan(stale_owner)
  ignore(ownership_prime_accepted_completion(stale, stale_plan))
  stale.exchange_epoch += 1UL
  try stale.accepted_completion_for_sequence(1UL) catch {
    Invalid(Owner) => ()
    _ => fail("unexpected stale-epoch rejection")
  } noraise {
    _ => fail("stale accepted epoch minted a capability")
  }
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "accepted completion scalar requires completed protocol and live process" {
  let protocol_owner = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let protocol_b = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let PreparedRankGroupProcess(protocol_stale) = ownership_prepared_for(
    protocol_owner, protocol_b,
  )
  let protocol_plan = ownership_plan(protocol_owner)
  ignore(ownership_prime_accepted_completion(protocol_stale, protocol_plan))
  let completed = protocol_stale.protocol.reauthenticate_completed(
    protocol_stale.accepted_epoch,
  )
  protocol_stale.protocol.retire_completed(completed)
  try protocol_stale.accepted_completion_for_sequence(1UL) catch {
    Failed(Protocol(_)) => ()
    _ => fail("unexpected stale protocol rejection")
  } noraise {
    _ => fail("ready protocol minted an accepted capability")
  }
  assert_eq(protocol_stale.lifecycle(), Faulted)

  let terminal_owner = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let terminal_b = @worker_protocol.SchedulePlanBuffer::new(
    ownership_worker_limits(),
  )
  let PreparedRankGroupProcess(terminal) = ownership_prepared_for(
    terminal_owner, terminal_b,
  )
  let terminal_plan = ownership_plan(terminal_owner)
  ignore(ownership_prime_accepted_completion(terminal, terminal_plan))
  terminal.failure_value = Some(TransportLost(rank=0))
  terminal.lifecycle_value = Faulted
  try terminal.accepted_completion_for_sequence(1UL) catch {
    Failed(TransportLost(rank=0)) => ()
    _ => fail("unexpected faulted-process rejection")
  } noraise {
    _ => fail("faulted process minted an accepted capability")
  }
  terminal.lifecycle_value = Cleaning
  try terminal.accepted_completion_for_sequence(1UL) catch {
    Failed(TransportLost(rank=0)) => ()
    _ => fail("unexpected cleaning-process rejection")
  } noraise {
    _ => fail("cleaning process minted an accepted capability")
  }
}
