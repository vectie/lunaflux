///|
fn maintenance_test_limits() -> ProcessLimits raise ProcessError {
  ProcessLimits::new(
    max_frame_bytes=4096,
    io_timeout_millis=1000,
    shutdown_timeout_millis=1000,
  )
}

///|
test "cooperative child shutdown retains one owner and exact exit" {
  let child = spawn(b"/bin/cat", maintenance_test_limits())
  child.begin_shutdown_maintenance()
  assert_eq(child.maintenance_phase, 1)
  try child.begin_shutdown_maintenance() catch {
    InvalidState(Close) => ()
    _ => fail("unexpected duplicate maintenance error")
  } noraise {
    _ => fail("duplicate maintenance begin was accepted")
  }
  try child.wait() catch {
    InvalidState(Wait) => ()
    _ => fail("unexpected compatibility wait substitution error")
  } noraise {
    _ => fail("blocking wait substituted for active maintenance")
  }
  try child.close() catch {
    InvalidState(Close) => ()
    _ => fail("unexpected compatibility close substitution error")
  } noraise {
    _ => fail("blocking close substituted for active maintenance")
  }
  let first = child.progress_shutdown_maintenance()
  assert_true(first.is_advanced())
  assert_eq(child.maintenance_phase, 2)
  let mut complete = false
  for _ in 0..<100000 {
    if !complete {
      let progress = child.progress_shutdown_maintenance()
      assert_false(progress.is_cleanup_required())
      if progress.is_complete() {
        complete = true
      }
    }
  }
  assert_true(complete)
  assert_true(child.is_closed())
  let exit = child.maintenance_exit()
  assert_eq(exit.kind(), 0)
  assert_eq(exit.code(), 0)
  assert_true(child.progress_shutdown_maintenance().is_complete())
}

///|
test "maintenance begin revokes both pending fixed-buffer directions" {
  let child = spawn(b"/bin/cat", maintenance_test_limits())
  let source = FixedArray::make(8, b'x')
  let destination = FixedArray::make(8, b'y')
  let staging = FixedArray::make(8, b'z')
  child.begin_write_frame(source, offset=0, byte_count=8)
  child.begin_read_frame(destination, staging~)
  assert_true(child.write_transfer_active)
  assert_true(child.read_transfer_active)
  child.begin_shutdown_maintenance()
  assert_false(child.write_transfer_active)
  assert_false(child.read_transfer_active)
  assert_true(
    physical_equal(child.write_transfer_source, child.empty_transfer_buffer),
  )
  assert_true(
    physical_equal(child.read_transfer_destination, child.empty_transfer_buffer),
  )
  assert_true(
    physical_equal(child.read_transfer_staging, child.empty_transfer_buffer),
  )
  let mut complete = false
  for _ in 0..<100000 {
    if !complete && child.progress_shutdown_maintenance().is_complete() {
      complete = true
    }
  }
  assert_true(complete)
}

///|
test "backward monotonic samples cannot extend a maintenance phase" {
  let child = spawn(b"/bin/cat", maintenance_test_limits())
  child.begin_shutdown_maintenance()
  assert_true(child.progress_shutdown_maintenance().is_advanced())
  let now = raw_monotonic_now()
  child.maintenance_phase_started_millis = now + 1000000L
  child.maintenance_deadline_millis = child.maintenance_phase_started_millis +
    child.limits.shutdown_timeout_millis.to_int64()
  try child.maintenance_remaining_millis() catch {
    OperationFailed(Wait) => ()
    _ => fail("unexpected backward-clock query error")
  } noraise {
    _ => fail("backward clock extended remaining maintenance grace")
  }
  let blocked = child.progress_shutdown_maintenance()
  assert_true(blocked.is_cleanup_required())
  assert_eq(child.maintenance_phase, 11)
  assert_eq(child.maintenance_resume_phase, 2)
  assert_true(child.progress_shutdown_maintenance().is_cleanup_required())
  let recovered_now = raw_monotonic_now()
  child.maintenance_phase_started_millis = recovered_now
  child.maintenance_deadline_millis = recovered_now +
    child.limits.shutdown_timeout_millis.to_int64()
  assert_true(child.progress_shutdown_maintenance().is_advanced())
  assert_eq(child.maintenance_phase, 2)
  let mut complete = false
  for _ in 0..<100000 {
    if !complete && child.progress_shutdown_maintenance().is_complete() {
      complete = true
    }
  }
  assert_true(complete)
}

///|
test "bounded final reap converts endless pending into retained cleanup stuck" {
  let child = fixture_child_owner(maintenance_test_limits())
  child.maintenance_phase = 10
  let started = child.begin_final_reap_interval()
  assert_true(started.is_advanced())
  assert_eq(child.maintenance_phase, 12)
  assert_eq(
    child.maintenance_deadline_millis - child.maintenance_phase_started_millis,
    1000L,
  )
  for _ in 0..<1000 {
    let pending = child.apply_final_reap_result(5L, maintenance_pending)
    assert_true(pending.is_pending())
    assert_eq(child.maintenance_phase, 12)
  }
  let stuck = child.expire_final_reap_interval()
  assert_true(stuck.is_cleanup_stuck())
  assert_eq(child.maintenance_phase, 13)
  assert_false(child.maintenance_has_exit)
  for _ in 0..<1000 {
    let retained = child.apply_final_reap_result(5L, maintenance_cleanup_stuck)
    assert_true(retained.is_cleanup_stuck())
    assert_false(retained.is_pending())
    assert_eq(child.maintenance_phase, 13)
    assert_false(child.maintenance_has_exit)
  }
  let reaped = child.apply_final_reap_result(
    (9L << 16) | (1L << 8),
    maintenance_cleanup_stuck,
  )
  assert_true(reaped.is_advanced())
  assert_eq(child.maintenance_phase, 7)
  let exit = child.maintenance_exit()
  assert_eq(exit.kind(), 1)
  assert_eq(exit.code(), 9)
}

///|
test "backward clock in final reap retains recoverable phase" {
  let child = spawn(b"/bin/cat", maintenance_test_limits())
  child.begin_shutdown_maintenance()
  child.maintenance_phase = 12
  let now = raw_monotonic_now()
  child.maintenance_phase_started_millis = now + 1000000L
  child.maintenance_deadline_millis = child.maintenance_phase_started_millis +
    child.limits.shutdown_timeout_millis.to_int64()
  let blocked = child.progress_final_reap_interval()
  assert_true(blocked.is_cleanup_required())
  assert_false(blocked.is_cleanup_stuck())
  assert_eq(child.maintenance_phase, 11)
  assert_eq(child.maintenance_resume_phase, 12)
  let recovered_now = raw_monotonic_now()
  child.maintenance_phase_started_millis = recovered_now
  child.maintenance_deadline_millis = recovered_now +
    child.limits.shutdown_timeout_millis.to_int64()
  assert_true(child.progress_shutdown_maintenance().is_advanced())
  assert_eq(child.maintenance_phase, 12)
  child.maintenance_phase = 0
  child.close()
  assert_true(child.is_closed())
}
