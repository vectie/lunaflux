///|
fn public_idle_exit_limits() -> ProcessLimits raise ProcessError {
  ProcessLimits::new(
    max_frame_bytes=4096,
    io_timeout_millis=1000,
    shutdown_timeout_millis=1000,
  )
}

///|
test "public idle observation publishes exact exit then maintenance closes" {
  let child = spawn(b"/usr/bin/true", public_idle_exit_limits())
  let mut exited = false
  let mut exit_kind = -1
  let mut exit_code = -1
  for _ in 0..<100000 {
    if !exited {
      let observed = child.observe_idle_exit()
      if observed.has_exited() {
        let exit = observed.exit()
        exited = true
        exit_kind = exit.kind()
        exit_code = exit.code()
      } else {
        assert_true(observed.is_alive())
      }
    }
  }
  assert_true(exited)
  assert_eq(exit_kind, 0)
  assert_eq(exit_code, 0)
  assert_true(child.is_failed())
  assert_true(child.progress_shutdown_maintenance().is_complete())
  assert_true(child.is_closed())
  let retained = child.maintenance_exit()
  assert_eq(retained.kind(), 0)
  assert_eq(retained.code(), 0)
}
