///|
/// Closes and reaps the current child before any recovery retirement.
pub fn WorkerService::begin_recovery(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  if self.lifecycle != ServiceReady &&
    self.lifecycle != ServiceRecovering &&
    self.lifecycle != ServiceClosingFailed {
    raise Invalid(Lifecycle)
  }
  self.lifecycle = ServiceRecovering
  self.process.close() catch {
    error => {
      self.lifecycle = ServiceClosingFailed
      raise process_error(error)
    }
  }
}

///|
/// Retires the oldest closed-child obligation. A pinned valid completion is
/// committed normally; an unreturned plan is committed as the given failure.
pub fn WorkerService::recover_oldest(
  self : WorkerService,
  failure : @worker_protocol.WorkerFailure,
) -> UInt64 raise WorkerServiceError {
  self.require_lifecycle(ServiceRecovering)
  if self.process.lifecycle() != Closed {
    raise Invalid(Recovery)
  }
  let slot = self.oldest_slot()
  if slot < 0 {
    raise Invalid(Sequence)
  }
  let sequence = self.flight_sequences[slot]
  if self.flight_states[slot] == FLIGHT_RECEIVED {
    return self.commit_received_slot(slot)
  }
  let plan = self.scheduler.submitted_plan_for_sequence(sequence) catch {
    error => raise scheduler_error(error)
  }
  if self.flight_states[slot] == FLIGHT_SUBMITTED {
    let submission = self.process.submission_for_sequence(sequence) catch {
      error => raise process_error(error)
    }
    self.process.validate_abandonment(submission) catch {
      error => raise process_error(error)
    }
    self.scheduler.fail_plan(plan, failure) catch {
      error => raise scheduler_error(error)
    }
    self.process.abandon_submission(submission) catch {
      error => raise process_error(error)
    }
  } else if self.flight_states[slot] == FLIGHT_PLANNED {
    self.scheduler.fail_plan(plan, failure) catch {
      error => raise scheduler_error(error)
    }
  } else {
    raise Invalid(Recovery)
  }
  self.clear_slot(slot)
  sequence
}

///|
pub fn WorkerService::restart(
  self : WorkerService,
  executable : Bytes,
  limits : @worker_process.WorkerProcessLimits,
) -> WorkerServiceRestart raise WorkerServiceError {
  self.require_lifecycle(ServiceRecovering)
  if self.outstanding_count() != 0 {
    raise Invalid(Recovery)
  }
  let startup = self.process.recovery_startup_contract() catch {
    error => raise process_error(error)
  }
  let result = @worker_process.prepare(executable, startup, limits) catch {
    error => raise process_error(error)
  }
  match result {
    ReadyProcess(replacement) => {
      self.process = replacement
      self.lifecycle = ServiceReady
      Restarted
    }
    CleanupRequired(cleanup) => RestartCleanupRequired(cleanup)
  }
}

///|
pub fn WorkerService::shutdown_clean(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_lifecycle(ServiceReady)
  if self.outstanding_count() != 0 {
    raise Invalid(Lifecycle)
  }
  self.process.shutdown_clean() catch {
    error => {
      self.lifecycle = ServiceRecovering
      raise process_error(error)
    }
  }
  self.process.close() catch {
    error => {
      self.lifecycle = ServiceClosingFailed
      raise process_error(error)
    }
  }
  self.lifecycle = ServiceClosed
}
