///|
fn overlap_service(executable : Bytes) -> @worker_service.WorkerService raise {
  @worker_service.new_fixture(
    scheduler_owner(),
    service_binding(),
    process_owner(executable),
  )
}

///|
fn drain_overlap_publications(
  service : @worker_service.WorkerService,
) -> Unit raise {
  while service.generated_token_count() + service.terminal_notice_count() > 0 {
    ignore(service.take_next_publication())
  }
}

///|
fn run_aba_overlap(executable : Bytes) -> Unit raise {
  let service = overlap_service(executable)
  ignore(service.admit(request(801UL), 0UL))
  ignore(service.admit(request(802UL), 0UL))
  ignore(service.admit(request(803UL), 0UL))

  progress_to_ready(service, 1UL)
  assert_eq(service.outstanding_count(), 2)
  commit_ready(service, 1UL)
  assert_eq(service.outstanding_count(), 1)

  progress_to_ready(service, 2UL)
  assert_eq(service.outstanding_count(), 2)
  guard service.progress() is ServiceBackpressured else {
    raise Failure("pending A plan bypassed completion backpressure")
  }
  assert_eq(service.outstanding_count(), 2)
  drain_overlap_publications(service)
  commit_ready(service, 2UL)
  assert_eq(service.outstanding_count(), 1)
  drain_overlap_publications(service)

  progress_to_ready(service, 3UL)
  commit_ready(service, 3UL)
  drain_overlap_publications(service)
  assert_eq(service.outstanding_count(), 0)
  assert_eq(service.live_request_count(), 0)
  service.shutdown_clean()
}

///|
fn run_cancelled_pending(executable : Bytes) -> Unit raise {
  let service = overlap_service(executable)
  ignore(service.admit(request(811UL), 0UL))
  let pending = service.admit(request(812UL), 0UL)
  progress_to_ready(service, 1UL)
  assert_eq(service.outstanding_count(), 2)
  service.cancel(pending)
  commit_ready(service, 1UL)
  assert_eq(
    service.take_next_publication().terminal_notice().reason(),
    CancelledByCaller,
  )
  drain_overlap_publications(service)
  progress_to_ready(service, 2UL)
  commit_ready(service, 2UL)
  assert_eq(service.outstanding_count(), 0)
  assert_eq(service.live_request_count(), 0)
  service.shutdown_clean()
}

///|
fn run_expired_pending(executable : Bytes) -> Unit raise {
  let service = overlap_service(executable)
  ignore(service.admit(request(821UL, deadline_budget=1000UL), 0UL))
  ignore(service.admit(request(822UL, deadline_budget=1UL), 0UL))
  progress_to_ready(service, 1UL)
  assert_eq(service.outstanding_count(), 2)
  assert_eq(service.expire_deadlines(1UL), 1)
  commit_ready(service, 1UL)
  assert_eq(
    service.take_next_publication().terminal_notice().reason(),
    DeadlineReached,
  )
  drain_overlap_publications(service)
  progress_to_ready(service, 2UL)
  commit_ready(service, 2UL)
  assert_eq(service.outstanding_count(), 0)
  assert_eq(service.live_request_count(), 0)
  service.shutdown_clean()
}

///|
fn run_lost_flight_with_pending(executable : Bytes) -> Unit raise {
  let root = process_owner(executable)
  let service = @worker_service.new_fixture(
    scheduler_owner(),
    service_binding(),
    root,
  )
  ignore(service.admit(request(831UL), 0UL))
  ignore(service.admit(request(832UL), 0UL))
  guard service.progress() is ServiceStarted(1UL) else {
    raise Failure("overlap loss fixture did not start A")
  }
  guard service.progress() is ServicePlanPending(2UL) else {
    raise Failure("overlap loss fixture did not retain B")
  }
  root.begin_recovery()
  try service.progress() catch {
    PhysicalTransport(_) => ()
    _ => raise Failure("unexpected overlap loss progress error")
  } noraise {
    _ => raise Failure("lost physical flight remained ready")
  }
  service.begin_recovery()
  assert_eq(service.recover_flight(ExecutionFailed), 1UL)
  assert_eq(service.outstanding_count(), 0)
  assert_eq(service.lifecycle(), ServiceDeviceInvalidationRequired)
  assert_eq(service.take_terminal_notice().reason(), WorkerFailed)
  assert_eq(service.take_terminal_notice().reason(), WorkerFailed)
  assert_eq(service.invalidate_device_state(), 0)
  service.abandon_restart()
  assert_eq(service.drain_restart_forbidden(), Complete)
  service.close()
}

///|
fn run_worker_plan_overlap_evidence(executable : Bytes) -> Unit raise {
  run_aba_overlap(executable)
  run_cancelled_pending(executable)
  run_expired_pending(executable)
  run_lost_flight_with_pending(executable)
}
