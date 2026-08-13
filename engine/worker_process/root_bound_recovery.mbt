///|
/// Closes only the current child. The exact root pair remains live for the
/// eventual zero-argument replacement. A close failure has its own retryable
/// phase and cannot be confused with terminal retirement.
pub fn RootBoundWorkerProcessSupervisor::begin_recovery(
  self : RootBoundWorkerProcessSupervisor,
) -> Unit raise RootBoundWorkerProcessError {
  if self.lifecycle != RootBoundReady &&
    self.lifecycle != RootBoundRecoveryRequired &&
    self.lifecycle != RootBoundRecoveryClosingFailed {
    raise RootBoundInvalidLifecycle
  }
  self.clear_exchange()
  let attempt = attempt_root_bound_cleanup(
    true,
    false,
    () => {
      try self.process.close() catch {
        error => Some(Process(error))
      } noraise {
        _ => None
      }
    },
    () => None,
    () => self.process.lifecycle() != Closed,
    () => false,
  )
  self.child_cleanup_failure = attempt.child_failure
  if attempt.child_live {
    self.cause = attempt.child_failure
    self.lifecycle = RootBoundRecoveryClosingFailed
    raise_cleanup_attempt(attempt)
  }
  self.lifecycle = RootBoundRecovering
}

///|
/// Spawns the only permitted replacement: same executable, limits, canonical
/// source bytes, and retained root pair, with predecessor derived from the
/// fully retired old supervisor.
pub fn RootBoundWorkerProcessSupervisor::restart(
  self : RootBoundWorkerProcessSupervisor,
) -> RootBoundWorkerProcessRestart raise RootBoundWorkerProcessError {
  if self.lifecycle != RootBoundRecovering || self.failed_replacement is Some(_) {
    raise RootBoundInvalidLifecycle
  }
  let startup = self.process.recovery_startup_contract() catch {
    error => raise RootBoundFailure(Process(error))
  }
  let prepared = preflight_prepare(
    self.executable,
    startup,
    self.process.bootstrap_source(),
    self.limits,
  ) catch {
    error => raise RootBoundFailure(Process(error))
  }
  let result = prepare_preflighted(
    self.process.bootstrap_source(),
    self.limits,
    prepared,
    self.require_roots(),
  )
  match result {
    ReadyProcess(replacement) => {
      self.process = replacement
      self.clear_exchange()
      self.cause = None
      self.child_cleanup_failure = None
      self.lifecycle = RootBoundReady
      RootBoundRestarted
    }
    CleanupRequired(failed) => {
      if failed.is_closed() {
        let primary = Process(failed.cause())
        if failed.cleanup_status != 0 {
          raise RootBoundCompoundFailure(
            primary,
            Process(failed.cleanup_failure()),
          )
        }
        raise RootBoundFailure(primary)
      }
      self.failed_replacement = Some(failed)
      self.cause = Some(Process(failed.cause()))
      self.child_cleanup_failure = if failed.cleanup_status == 0 {
        None
      } else {
        Some(Process(failed.cleanup_failure()))
      }
      self.lifecycle = RootBoundReplacementCleanup
      RootBoundRestartCleanupRequired
    }
  }
}

///|
pub fn RootBoundWorkerProcessSupervisor::retry_replacement_cleanup(
  self : RootBoundWorkerProcessSupervisor,
) -> Unit raise RootBoundWorkerProcessError {
  if self.lifecycle != RootBoundReplacementCleanup {
    raise RootBoundInvalidLifecycle
  }
  guard self.failed_replacement is Some(failed) else {
    raise RootBoundInvalidLifecycle
  }
  let attempt = attempt_root_bound_cleanup(
    true,
    false,
    () => {
      try failed.retry_close() catch {
        error => Some(Process(error))
      } noraise {
        _ => None
      }
    },
    () => None,
    () => !failed.is_closed(),
    () => false,
  )
  self.child_cleanup_failure = attempt.child_failure
  if attempt.child_live {
    raise_cleanup_attempt(attempt)
  }
  self.failed_replacement = None
  self.lifecycle = RootBoundRecovering
}

///|
/// Retires the instance. Current/failed child cleanup and root-pair cleanup are
/// both attempted. Busy roots preserve exact retry authority; a consumed
/// close failure remains diagnostic evidence without a false live lease.
pub fn RootBoundWorkerProcessSupervisor::close(
  self : RootBoundWorkerProcessSupervisor,
) -> Unit raise RootBoundWorkerProcessError {
  self.clear_exchange()
  if self.lifecycle == RootBoundClosed {
    return
  }
  let child_live = self.failed_replacement is Some(_) ||
    self.process.lifecycle() != Closed
  let attempt = attempt_root_bound_cleanup(
    child_live,
    roots_still_live(self.require_roots()),
    () => {
      let mut failure = None
      if self.failed_replacement is Some(failed) {
        try failed.retry_close() catch {
          error => failure = Some(Process(error))
        } noraise {
          _ => self.failed_replacement = None
        }
      }
      if self.process.lifecycle() != Closed {
        try self.process.close() catch {
          error => if failure is None { failure = Some(Process(error)) }
        } noraise {
          _ => ()
        }
      }
      if failure is Some(_) {
        failure
      } else {
        None
      }
    },
    () => close_root_pair(self.require_roots()),
    () => {
      match self.failed_replacement {
        Some(failed) => !failed.is_closed()
        None => self.process.lifecycle() != Closed
      }
    },
    () => roots_still_live(self.require_roots()),
  )
  self.child_cleanup_failure = attempt.child_failure
  self.roots_cleanup_failure = attempt.roots_failure
  self.lifecycle = if !attempt.child_live && !attempt.roots_live {
    RootBoundClosed
  } else {
    RootBoundClosingFailed
  }
  raise_cleanup_attempt(attempt)
}
