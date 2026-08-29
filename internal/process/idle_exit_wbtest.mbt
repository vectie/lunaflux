///|
fn idle_exit_test_limits() -> ProcessLimits raise ProcessError {
  ProcessLimits::new(
    max_frame_bytes=4096,
    io_timeout_millis=1000,
    shutdown_timeout_millis=1000,
  )
}

///|
test "idle liveness observation leaves a live child usable" {
  let child = spawn(b"/bin/cat", idle_exit_test_limits())
  let observed = child.observe_idle_exit()
  assert_true(observed.is_alive())
  assert_false(observed.has_exited())
  assert_eq(child.maintenance_phase, 0)
  assert_false(child.is_failed())
  try ignore(observed.exit()) catch {
    InvalidState(Wait) => ()
    _ => fail("unexpected live-observation exit error")
  } noraise {
    _ => fail("live observation fabricated exit evidence")
  }
  child.begin_shutdown_maintenance()
  let mut complete = false
  for _ in 0..<100000 {
    if !complete && child.progress_shutdown_maintenance().is_complete() {
      complete = true
    }
  }
  assert_true(complete)
}

///|
test "idle exit is recorded once and closes through maintenance" {
  let child = spawn(b"/usr/bin/true", idle_exit_test_limits())
  let mut observed = idle_child_alive
  for _ in 0..<100000 {
    if !observed.has_exited() {
      observed = child.observe_idle_exit()
    }
  }
  assert_true(observed.has_exited())
  let exit = observed.exit()
  assert_eq(exit.kind(), 0)
  assert_eq(exit.code(), 0)
  assert_true(child.is_failed())
  assert_eq(child.maintenance_phase, 7)
  let retained = child.maintenance_exit()
  assert_eq(retained.kind(), exit.kind())
  assert_eq(retained.code(), exit.code())
  try ignore(child.observe_idle_exit()) catch {
    InvalidState(Wait) => ()
    _ => fail("unexpected duplicate idle-observation error")
  } noraise {
    _ => fail("reaped child was observed twice")
  }
  let closed = child.progress_shutdown_maintenance()
  assert_true(closed.is_complete())
  assert_true(child.is_closed())
  let closed_exit = child.maintenance_exit()
  assert_eq(closed_exit.kind(), exit.kind())
  assert_eq(closed_exit.code(), exit.code())
}

///|
test "idle observation rejects either active transfer without revocation" {
  let child = spawn(b"/bin/cat", idle_exit_test_limits())
  let source = FixedArray::make(8, b'x')
  let destination = FixedArray::make(8, b'y')
  let staging = FixedArray::make(8, b'z')
  child.begin_write_frame(source, offset=0, byte_count=8)
  child.begin_read_frame(destination, staging~)
  try ignore(child.observe_idle_exit()) catch {
    InvalidState(Wait) => ()
    _ => fail("unexpected active-transfer observation error")
  } noraise {
    _ => fail("idle observation overlapped an active transfer")
  }
  assert_true(child.write_transfer_active)
  assert_true(child.read_transfer_active)
  assert_false(child.is_failed())
  child.begin_shutdown_maintenance()
  let mut complete = false
  for _ in 0..<100000 {
    if !complete && child.progress_shutdown_maintenance().is_complete() {
      complete = true
    }
  }
  assert_true(complete)
}
