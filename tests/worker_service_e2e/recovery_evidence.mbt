///|
fn run_owned_startup_failure_evidence() -> Unit raise {
  let model = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  try
    @worker_service.prepare_owned(
      scheduler_blueprint(),
      service_binding(),
      b"/private/tmp/lunaflux-worker-does-not-exist",
      startup(),
      bootstrap_source(),
      process_limits(),
      model,
      kernel,
    )
  catch {
    _ => ()
  } noraise {
    _ => raise Failure("missing executable published an owned service")
  }
  model.close()
  kernel.close()
}

///|
fn run_terminal_drain_evidence(executable : Bytes) -> Unit raise {
  let terminal_service = @worker_service.new_fixture(
    scheduler_owner(),
    service_binding(),
    process_owner(executable),
  )
  ignore(terminal_service.admit(request(201UL), 0UL))
  terminal_service.begin_recovery()
  assert_eq(terminal_service.lifecycle(), ServiceDeviceInvalidationRequired)
  assert_eq(terminal_service.invalidate_device_state(), 0)
  terminal_service.abandon_restart()
  assert_eq(terminal_service.lifecycle(), ServiceRestartForbidden)
  assert_eq(terminal_service.drain_restart_forbidden(), Complete)
  assert_eq(terminal_service.lifecycle(), ServiceTerminalCloseReady)
  assert_eq(terminal_service.take_terminal_notice().reason(), WorkerFailed)
  terminal_service.close()
  assert_eq(terminal_service.lifecycle(), ServiceClosed)

  let pressured = @worker_service.new_fixture(
    scheduler_owner(),
    service_binding(),
    process_owner(executable),
  )
  for id in 301UL..<307UL {
    pressured.cancel(pressured.admit(request(id), 0UL))
  }
  ignore(pressured.admit(request(307UL), 0UL))
  pressured.begin_recovery()
  assert_eq(pressured.invalidate_device_state(), 0)
  pressured.abandon_restart()
  try pressured.drain_restart_forbidden() catch {
    Scheduler(Invalid(field=TerminalNoticeCapacity, issue=Backpressured)) => ()
    _ => raise Failure("unexpected terminal-drain backpressure")
  } noraise {
    _ => raise Failure("full terminal ring accepted service drain")
  }
  assert_eq(pressured.lifecycle(), ServiceRestartForbidden)
  assert_eq(pressured.live_request_count(), 1)
  try pressured.close() catch {
    Invalid(Lifecycle) => ()
    _ => raise Failure("unexpected live terminal-close rejection")
  } noraise {
    _ => raise Failure("terminal close discarded a waiting request")
  }
  ignore(pressured.take_terminal_notice())
  assert_eq(pressured.drain_restart_forbidden(), Complete)
  pressured.close()
}

///|
fn run_reservation_classification_evidence(executable : Bytes) -> Unit raise {
  // These retained aliases deliberately violate exclusive-owner discipline to
  // inject deterministic child-loss races into the production state machine.
  let reserved_root = process_owner(executable)
  let reserved = @worker_service.new_fixture(
    scheduler_owner(),
    service_binding(),
    reserved_root,
  )
  ignore(reserved.admit(request(401UL), 0UL))
  guard reserved.progress() is ServiceStarted(1UL) else {
    raise Failure("reserved fault fixture did not start")
  }
  reserved_root.begin_recovery()
  try reserved.progress() catch {
    RootBoundProcess(_) => ()
    _ => raise Failure("unexpected reserved-flight loss error")
  } noraise {
    _ => raise Failure("lost reserved child remained service-ready")
  }
  reserved.begin_recovery()
  assert_eq(reserved.recover_flight(ExecutionFailed), 1UL)
  assert_eq(reserved.lifecycle(), ServiceDeviceInvalidationRequired)
  ignore(reserved.take_terminal_notice())
  assert_eq(reserved.invalidate_device_state(), 0)
  reserved.abandon_restart()
  assert_eq(reserved.drain_restart_forbidden(), Complete)
  reserved.close()

  let absent_root = process_owner(executable)
  let absent = @worker_service.new_fixture(
    scheduler_owner(),
    service_binding(),
    absent_root,
  )
  let foreign_scheduler = scheduler_owner()
  ignore(foreign_scheduler.admit(request(402UL), 0UL))
  let foreign_plan = foreign_scheduler.submitted_plan(
    foreign_scheduler.build_next(),
  )
  assert_eq(absent_root.begin_exchange(foreign_plan), 1UL)
  complete_root_exchange(absent_root)
  absent_root.retire_received(absent_root.received_for_sequence(1UL))
  ignore(absent.admit(request(403UL), 0UL))
  try absent.progress() catch {
    RootBoundProcess(_) => ()
    _ => raise Failure("unexpected absent-reservation begin error")
  } noraise {
    _ => raise Failure("pre-reservation sequence mismatch started")
  }
  assert_eq(absent.lifecycle(), ServiceRecovering)
  absent.begin_recovery()
  assert_eq(absent.recover_flight(ExecutionFailed), 1UL)
  assert_eq(absent.lifecycle(), ServiceRestartForbidden)
  ignore(absent.take_terminal_notice())
  assert_eq(absent.drain_restart_forbidden(), Complete)
  absent.close()
}
