///|
fn drive_worker_maintenance(
  owner : WorkerProcessSupervisor,
) -> WorkerProcessMaintenanceProgress raise {
  for _ in 0..<1000000 {
    let progress = owner.progress_shutdown_maintenance()
    if progress.is_complete() ||
      progress.is_cleanup_required() ||
      progress.is_cleanup_stuck() {
      return progress
    }
  }
  fail("worker maintenance did not reach a terminal result")
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "generic maintenance preserves exact recovery obligation" {
  let owner = test_supervisor()
  owner.state_a = 1
  owner.sequence_a = 1UL
  owner.last_submitted = 1UL
  let retained = WorkerSubmission(owner, 0, 1UL)
  owner.begin_shutdown_maintenance()
  assert_eq(owner.shutdown_maintenance_remaining_millis(), 0)
  assert_eq(owner.lifecycle(), ClosingFailed)
  assert_eq(owner.state_a, 1)
  let retryable = owner.apply_maintenance_progress(true, false, false, false)
  assert_true(retryable.is_cleanup_required())
  assert_eq(owner.maintenance_phase, 1)
  let stuck = owner.apply_maintenance_progress(false, true, false, false)
  assert_true(stuck.is_cleanup_stuck())
  assert_eq(owner.lifecycle(), ClosingFailed)
  assert_eq(owner.maintenance_phase, 1)
  try owner.close() catch {
    Invalid(Lifecycle) => ()
    _ => fail("unexpected active-maintenance close error")
  } noraise {
    _ => fail("blocking close substituted for cooperative maintenance")
  }
  let first = owner.progress_shutdown_maintenance()
  assert_true(first.is_advanced())
  let remaining = owner.shutdown_maintenance_remaining_millis()
  assert_true(remaining >= 0 && remaining <= 1000)
  let result = drive_worker_maintenance(owner)
  assert_true(result.is_complete())
  assert_eq(owner.lifecycle(), Closed)
  let exit = owner.shutdown_maintenance_exit()
  assert_eq(exit.kind(), 0)
  assert_eq(exit.code(), 0)
  owner.validate_abandonment(retained)
  owner.abandon_submission(retained)
  assert_eq(owner.outstanding_count(), 0)
  try owner.validate_abandonment(retained) catch {
    Invalid(Submission) => ()
    _ => fail("unexpected retired alias error")
  } noraise {
    _ => fail("retired A/B alias remained live")
  }
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "generic maintenance escalates a non-EOF child through TERM" {
  let owner = test_supervisor(
    executable=b"/usr/bin/yes",
    shutdown_timeout_millis=1,
  )
  owner.begin_shutdown_maintenance()
  let result = drive_worker_maintenance(owner)
  assert_true(result.is_complete())
  let exit = owner.shutdown_maintenance_exit()
  assert_eq(exit.kind(), 1)
  assert_eq(exit.code(), 15)
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "root recovery retains roots and invalidates transport progress" {
  let owner = test_root_bound_supervisor()
  owner.transport_mode = 2
  owner.active_exchange_epoch = 1UL
  owner.exchange_phase = 0
  owner.exchange_sequence = 1UL
  owner.begin_recovery_maintenance()
  assert_eq(owner.maintenance_remaining_millis(), 0)
  assert_eq(owner.active_exchange_epoch, 0UL)
  let retained_stuck = owner.apply_retained_child_progress(
    worker_maintenance_cleanup_stuck,
  )
  assert_true(retained_stuck.is_cleanup_stuck())
  assert_eq(owner.maintenance_mode, 1)
  assert_eq(owner.lifecycle(), RootBoundRecoveryRequired)
  assert_true(roots_still_live(owner.roots))
  try owner.progress_exchange() catch {
    RootBoundInvalidLifecycle => ()
    _ => fail("unexpected stale exchange-progress error")
  } noraise {
    _ => fail("stale exchange progress survived recovery start")
  }
  let mut complete = false
  for _ in 0..<1000000 {
    if !complete {
      let progress = owner.progress_maintenance()
      assert_false(progress.is_cleanup_required())
      assert_false(progress.is_cleanup_stuck())
      if progress.is_complete() {
        complete = true
      }
    }
  }
  assert_true(complete)
  assert_eq(owner.lifecycle(), RootBoundRecovering)
  assert_true(roots_still_live(owner.roots))
  let exit = owner.maintenance_exit()
  assert_eq(exit.kind(), 0)
  assert_eq(exit.code(), 0)
  owner.close()
  assert_true(owner.is_closed())
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "root terminal maintenance closes roots only after child reap" {
  let owner = test_root_bound_supervisor()
  owner.begin_close_maintenance()
  assert_true(roots_still_live(owner.roots))
  let retained_stuck = owner.apply_retained_child_progress(
    worker_maintenance_cleanup_stuck,
  )
  assert_true(retained_stuck.is_cleanup_stuck())
  assert_eq(owner.maintenance_mode, 2)
  assert_eq(owner.lifecycle(), RootBoundClosingFailed)
  assert_true(roots_still_live(owner.roots))
  let mut child_complete = false
  for _ in 0..<1000000 {
    if !child_complete {
      let progress = owner.progress_maintenance()
      assert_false(progress.is_cleanup_required())
      assert_false(progress.is_cleanup_stuck())
      if owner.process.lifecycle() == Closed {
        child_complete = true
        assert_true(progress.is_advanced())
        assert_true(roots_still_live(owner.roots))
        assert_eq(owner.maintenance_mode, 3)
      }
    }
  }
  assert_true(child_complete)
  let busy = owner.apply_root_close_progress(7, true)
  assert_true(busy.is_cleanup_required())
  assert_eq(owner.maintenance_mode, 3)
  assert_eq(owner.roots_cleanup_failure(), Some(RootsBusy))
  assert_true(roots_still_live(owner.roots))
  let root_result = owner.progress_maintenance()
  assert_true(root_result.is_complete())
  assert_true(owner.is_closed())
  assert_false(roots_still_live(owner.roots))
  assert_true(owner.progress_maintenance().is_complete())
}
