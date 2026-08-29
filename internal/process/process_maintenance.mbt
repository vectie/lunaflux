///|
let maintenance_pending : ChildProcessMaintenanceProgress = { kind: 0 }

///|
let maintenance_advanced : ChildProcessMaintenanceProgress = { kind: 1 }

///|
let maintenance_complete : ChildProcessMaintenanceProgress = { kind: 2 }

///|
let maintenance_cleanup_required : ChildProcessMaintenanceProgress = { kind: 3 }

///|
let maintenance_cleanup_stuck : ChildProcessMaintenanceProgress = { kind: 4 }

///|
fn maintenance_deadline(timeout_millis : Int) -> Int64 raise ProcessError {
  capture_deadline(timeout_millis, Wait)
}

///|
fn ChildProcess::maintenance_clock_status_at(
  self : ChildProcess,
  now : Int64,
) -> Int raise ProcessError {
  let phase_timeout = self.limits.shutdown_timeout_millis.to_int64()
  if now < 0L ||
    now < self.maintenance_phase_started_millis ||
    self.maintenance_deadline_millis - now > phase_timeout {
    raise OperationFailed(Wait)
  }
  if now >= self.maintenance_deadline_millis {
    1
  } else {
    0
  }
}

///|
fn ChildProcess::maintenance_clock_status(
  self : ChildProcess,
) -> Int raise ProcessError {
  self.maintenance_clock_status_at(raw_monotonic_now())
}

///|
fn ChildProcess::record_maintenance_exit(
  self : ChildProcess,
  kind : Int,
  code : Int,
) -> Unit {
  self.maintenance_exit_kind = kind
  self.maintenance_exit_code = code
  self.maintenance_has_exit = true
  self.maintenance_phase = 7
}

///|
fn maintenance_wait_status(result : Int64) -> Int {
  (result & 0xffL).to_int()
}

///|
fn maintenance_wait_kind(result : Int64) -> Int {
  ((result >> 8) & 0xffL).to_int()
}

///|
fn maintenance_wait_code(result : Int64) -> Int {
  ((result >> 16) & 0xffL).to_int()
}

///|
fn ChildProcess::progress_maintenance_poll(
  self : ChildProcess,
  next_phase : Int,
) -> ChildProcessMaintenanceProgress {
  let clock_status = self.maintenance_clock_status() catch {
    _ => {
      self.failed = true
      self.maintenance_resume_phase = self.maintenance_phase
      self.maintenance_phase = 11
      return maintenance_cleanup_required
    }
  }
  if clock_status == 1 {
    self.maintenance_phase = next_phase
    return maintenance_advanced
  }
  let result = raw_try_wait(self.handle)
  let status = maintenance_wait_status(result)
  if status == 0 {
    self.record_maintenance_exit(
      maintenance_wait_kind(result),
      maintenance_wait_code(result),
    )
    return maintenance_advanced
  }
  if status != 5 {
    self.failed = true
    return maintenance_cleanup_required
  }
  maintenance_pending
}

///|
fn ChildProcess::apply_final_reap_result(
  self : ChildProcess,
  result : Int64,
  pending : ChildProcessMaintenanceProgress,
) -> ChildProcessMaintenanceProgress {
  let status = maintenance_wait_status(result)
  if status == 0 {
    self.record_maintenance_exit(
      maintenance_wait_kind(result),
      maintenance_wait_code(result),
    )
    return maintenance_advanced
  }
  if status == 5 {
    return pending
  }
  self.failed = true
  maintenance_cleanup_required
}

///|
fn ChildProcess::begin_final_reap_interval(
  self : ChildProcess,
) -> ChildProcessMaintenanceProgress {
  let deadline = maintenance_deadline(self.limits.shutdown_timeout_millis) catch {
    _ => {
      self.failed = true
      return maintenance_cleanup_required
    }
  }
  self.maintenance_phase_started_millis = deadline -
    self.limits.shutdown_timeout_millis.to_int64()
  self.maintenance_deadline_millis = deadline
  self.maintenance_phase = 12
  maintenance_advanced
}

///|
fn ChildProcess::expire_final_reap_interval(
  self : ChildProcess,
) -> ChildProcessMaintenanceProgress {
  self.failed = true
  self.maintenance_phase = 13
  maintenance_cleanup_stuck
}

///|
fn ChildProcess::progress_final_reap_interval(
  self : ChildProcess,
) -> ChildProcessMaintenanceProgress {
  let clock_status = self.maintenance_clock_status() catch {
    _ => {
      self.failed = true
      self.maintenance_resume_phase = 12
      self.maintenance_phase = 11
      return maintenance_cleanup_required
    }
  }
  if clock_status == 1 {
    return self.expire_final_reap_interval()
  }
  self.apply_final_reap_result(raw_try_wait(self.handle), maintenance_pending)
}

///|
fn ChildProcess::progress_stuck_reap(
  self : ChildProcess,
) -> ChildProcessMaintenanceProgress {
  self.apply_final_reap_result(
    raw_try_wait(self.handle),
    maintenance_cleanup_stuck,
  )
}

///|
/// Starts owner-resident cooperative shutdown without performing a native
/// process transition. Any retained pending-frame references are revoked.
pub fn ChildProcess::begin_shutdown_maintenance(
  self : ChildProcess,
) -> Unit raise ProcessError {
  if self.closed || self.maintenance_phase != 0 {
    raise InvalidState(Close)
  }
  let deadline = maintenance_deadline(self.limits.shutdown_timeout_millis)
  self.clear_pending_transfers()
  self.maintenance_phase_started_millis = deadline -
    self.limits.shutdown_timeout_millis.to_int64()
  self.maintenance_deadline_millis = deadline
  self.maintenance_resume_phase = 0
  self.maintenance_exit_kind = 0
  self.maintenance_exit_code = 0
  self.maintenance_has_exit = false
  self.maintenance_phase = 1
}

///|
/// Performs at most one native ownership transition: half-close, one
/// WNOHANG wait, TERM, KILL, or reaped-descriptor close.
pub fn ChildProcess::progress_shutdown_maintenance(
  self : ChildProcess,
) -> ChildProcessMaintenanceProgress raise ProcessError {
  match self.maintenance_phase {
    0 => raise InvalidState(Close)
    1 => {
      let status = raw_shutdown_write(self.handle)
      if status != 0 {
        self.failed = true
        return maintenance_cleanup_required
      }
      self.maintenance_phase = 2
      maintenance_advanced
    }
    2 => self.progress_maintenance_poll(3)
    3 => {
      let deadline = maintenance_deadline(self.limits.shutdown_timeout_millis) catch {
        _ => {
          self.failed = true
          return maintenance_cleanup_required
        }
      }
      if raw_maintenance_signal(self.handle, 0) != 0 {
        self.failed = true
        return maintenance_cleanup_required
      }
      self.failed = true
      self.maintenance_phase_started_millis = deadline -
        self.limits.shutdown_timeout_millis.to_int64()
      self.maintenance_deadline_millis = deadline
      self.maintenance_phase = 4
      maintenance_advanced
    }
    4 => self.progress_maintenance_poll(5)
    5 => {
      let deadline = maintenance_deadline(self.limits.shutdown_timeout_millis) catch {
        _ => {
          self.failed = true
          return maintenance_cleanup_required
        }
      }
      if raw_maintenance_signal(self.handle, 1) != 0 {
        self.failed = true
        return maintenance_cleanup_required
      }
      self.failed = true
      self.maintenance_phase_started_millis = deadline -
        self.limits.shutdown_timeout_millis.to_int64()
      self.maintenance_deadline_millis = deadline
      self.maintenance_phase = 6
      maintenance_advanced
    }
    6 => self.progress_maintenance_poll(10)
    7 | 9 => {
      let status = raw_close_reaped(self.handle)
      self.closed = raw_is_closed(self.handle)
      if status != 0 {
        self.failed = true
        self.maintenance_phase = 9
        return maintenance_cleanup_required
      }
      self.maintenance_phase = 8
      maintenance_complete
    }
    8 => maintenance_complete
    10 => self.begin_final_reap_interval()
    11 => {
      let now = raw_monotonic_now()
      let phase_timeout = self.limits.shutdown_timeout_millis.to_int64()
      if now < 0L ||
        now < self.maintenance_phase_started_millis ||
        self.maintenance_deadline_millis - now > phase_timeout {
        return maintenance_cleanup_required
      }
      self.maintenance_phase = self.maintenance_resume_phase
      self.maintenance_resume_phase = 0
      maintenance_advanced
    }
    12 => self.progress_final_reap_interval()
    13 => self.progress_stuck_reap()
    _ => {
      self.failed = true
      maintenance_cleanup_required
    }
  }
}

///|
/// Returns the current monotonic grace interval, clamped to zero. Transition
/// phases have no sleep interval and therefore also return zero.
pub fn ChildProcess::maintenance_remaining_millis(
  self : ChildProcess,
) -> Int raise ProcessError {
  if self.maintenance_phase == 0 {
    raise InvalidState(Wait)
  }
  if self.maintenance_phase != 2 &&
    self.maintenance_phase != 4 &&
    self.maintenance_phase != 6 &&
    self.maintenance_phase != 12 {
    return 0
  }
  let now = raw_monotonic_now()
  let phase_timeout = self.limits.shutdown_timeout_millis.to_int64()
  if now < 0L ||
    now < self.maintenance_phase_started_millis ||
    self.maintenance_deadline_millis - now > phase_timeout {
    raise OperationFailed(Wait)
  }
  let remaining = self.maintenance_deadline_millis - now
  if remaining <= 0L {
    0
  } else if remaining > 0x7fffffffL {
    0x7fffffff
  } else {
    remaining.to_int()
  }
}

///|
/// Returns exact reaped exit evidence while the same owner remains in cleanup
/// or complete state.
pub fn ChildProcess::maintenance_exit(
  self : ChildProcess,
) -> ChildExit raise ProcessError {
  if !self.maintenance_has_exit {
    raise InvalidState(Wait)
  }
  { kind: self.maintenance_exit_kind, code: self.maintenance_exit_code }
}
