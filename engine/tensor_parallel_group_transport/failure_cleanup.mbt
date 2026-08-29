///|
pub fn TensorParallelGroupTransport::probe_idle_rank(
  self : TensorParallelGroupTransport,
) -> Unit raise TensorParallelGroupTransportError {
  let supervisor = self.published_process()
  supervisor.probe_idle_rank() catch {
    _ => self.process_error(supervisor)
  }
}

///|
pub fn TensorParallelGroupTransport::take_group_failure(
  self : TensorParallelGroupTransport,
) -> @rank_group_protocol.RankGroupFailureReport raise TensorParallelGroupTransportError {
  if self.failure_report_taken || self.invalid_failure_report {
    rejected(Lifecycle)
  }
  self.capture_group_failure()
  match self.retained_failure_report {
    Some(report) => {
      self.failure_report_taken = true
      report
    }
    None => rejected(Process)
  }
}

///|
pub fn TensorParallelGroupTransport::failure_report_invalid(
  self : TensorParallelGroupTransport,
) -> Bool {
  self.invalid_failure_report
}

///|
pub fn TensorParallelGroupTransport::has_failure_publication_obligation(
  self : TensorParallelGroupTransport,
) -> Bool {
  self.invalid_failure_report ||
  (self.retained_failure_report is Some(_) && !self.failure_report_taken)
}

///|
/// Returns the exact already-consumed failure publication for service-local
/// sanitization. It grants no cleanup or publication authority.
pub fn TensorParallelGroupTransport::taken_group_failure(
  self : TensorParallelGroupTransport,
) -> @rank_group_protocol.RankGroupFailureReport raise TensorParallelGroupTransportError {
  if !self.failure_report_taken {
    rejected(Lifecycle)
  }
  match self.retained_failure_report {
    Some(report) => report
    None => rejected(Process)
  }
}

///|
fn TensorParallelGroupTransport::capture_group_failure(
  self : TensorParallelGroupTransport,
) -> Unit raise TensorParallelGroupTransportError {
  if self.retained_failure_report is Some(_) {
    return
  }
  let supervisor = match self.state {
    Failed(owner) => owner
    Published(owner) if owner.lifecycle() == Faulted => {
      self.state = Failed(owner)
      owner
    }
    _ => rejected(Lifecycle)
  }
  let inspected = supervisor.inspect_group_failure() catch {
    _ => {
      self.invalid_failure_report = true
      self.replacement_required = true
      rejected(Process)
    }
  }
  self.validate_or_poison_group_failure_report(
    inspected,
    supervisor.group_generation_value(),
    supervisor.world_size(),
  )
  let taken = supervisor.take_group_failure() catch { _ => rejected(Process) }
  if !same_group_failure_report(inspected, taken) {
    self.invalid_failure_report = true
    self.replacement_required = true
    rejected(Process)
  }
  self.retained_failure_report = Some(taken)
}

///|
fn TensorParallelGroupTransport::retain_failure_before_cleanup(
  self : TensorParallelGroupTransport,
  supervisor : @rank_group_process.RankGroupProcessSupervisor,
) -> Unit {
  self.state = Failed(supervisor)
  try self.capture_group_failure() catch {
    _ => {
      self.invalid_failure_report = true
      self.replacement_required = true
    }
  } noraise {
    _ => ()
  }
}

///|
fn TensorParallelGroupTransport::validate_group_failure_report(
  self : TensorParallelGroupTransport,
  report : @rank_group_protocol.RankGroupFailureReport,
  group_generation_value : UInt64,
  world_size : Int,
) -> Unit raise TensorParallelGroupTransportError {
  if report.group_generation_value() != group_generation_value ||
    report.plan_sequence_value() != self.active_sequence_value ||
    report.world_size() != world_size ||
    report.failed_rank() < -1 ||
    report.failed_rank() >= world_size ||
    !report.has_canonical_failure_rank() {
    rejected(Process)
  }
}

///|
fn TensorParallelGroupTransport::validate_or_poison_group_failure_report(
  self : TensorParallelGroupTransport,
  report : @rank_group_protocol.RankGroupFailureReport,
  group_generation_value : UInt64,
  world_size : Int,
) -> Unit raise TensorParallelGroupTransportError {
  if self.invalid_failure_report {
    rejected(Process)
  }
  let mut report_valid = false
  defer (if !report_valid {
    self.invalid_failure_report = true
    self.replacement_required = true
  })
  self.validate_group_failure_report(report, group_generation_value, world_size)
  report_valid = true
}

///|
fn same_group_failure_report(
  left : @rank_group_protocol.RankGroupFailureReport,
  right : @rank_group_protocol.RankGroupFailureReport,
) -> Bool {
  left.group_generation_value() == right.group_generation_value() &&
  left.plan_sequence_value() == right.plan_sequence_value() &&
  left.reason() == right.reason() &&
  left.failed_rank() == right.failed_rank() &&
  left.world_size() == right.world_size()
}

///|
pub fn TensorParallelGroupTransport::begin_recovery(
  self : TensorParallelGroupTransport,
) -> Unit raise TensorParallelGroupTransportError {
  let supervisor = match self.state {
    Failed(owner) => owner
    Published(owner) if owner.lifecycle() == Faulted => owner
    _ => rejected(Lifecycle)
  }
  if !self.invalid_failure_report {
    self.capture_group_failure()
  }
  self.replacement_required = true
  supervisor.begin_cleanup() catch {
    _ => {
      self.state = Failed(supervisor)
      rejected(Cleanup)
    }
  }
  self.state = Cleaning(supervisor, replace_after_cleanup=true)
}

///|
fn TensorParallelGroupTransport::finish_root_close(
  self : TensorParallelGroupTransport,
) -> Unit raise TensorParallelGroupTransportError {
  if !@approved_fs.worker_approved_roots_is_closed(self.roots) {
    @approved_fs.close_worker_approved_roots(self.roots) catch {
      _ => {
        self.state = ClosingRoots
        rejected(Cleanup)
      }
    }
  }
  if self.approved_executable.is_activation_live() {
    self.approved_executable.close() catch {
      _ => {
        self.state = ClosingRoots
        rejected(Cleanup)
      }
    }
  }
  self.state = OwnerClosed
}

///|
pub fn TensorParallelGroupTransport::begin_close(
  self : TensorParallelGroupTransport,
) -> Unit raise TensorParallelGroupTransportError {
  if self.active_sequence_value != 0UL && self.lifecycle() != Faulted {
    rejected(Lifecycle)
  }
  self.replacement_required = false
  match self.state {
    NoGroup => {
      self.state = ClosingRoots
      self.finish_root_close()
    }
    Published(supervisor) => {
      if supervisor.lifecycle() == Faulted {
        self.retain_failure_before_cleanup(supervisor)
        supervisor.begin_cleanup() catch {
          _ => self.process_error(supervisor, rule=Cleanup)
        }
      } else {
        supervisor.begin_close() catch {
          _ => {
            if supervisor.lifecycle() != Faulted {
              self.process_error(supervisor, rule=Cleanup)
            }
            self.retain_failure_before_cleanup(supervisor)
            supervisor.begin_cleanup() catch {
              _ => self.process_error(supervisor, rule=Cleanup)
            }
          }
        }
      }
      self.state = Cleaning(supervisor, replace_after_cleanup=false)
    }
    Failed(supervisor) => {
      self.retain_failure_before_cleanup(supervisor)
      supervisor.begin_cleanup() catch {
        _ => self.process_error(supervisor, rule=Cleanup)
      }
      self.state = Cleaning(supervisor, replace_after_cleanup=false)
    }
    OwnerClosed => ()
    _ => rejected(Lifecycle)
  }
}

///|
pub fn TensorParallelGroupTransport::progress_cleanup(
  self : TensorParallelGroupTransport,
) -> TensorParallelGroupCleanupProgress raise TensorParallelGroupTransportError {
  match self.state {
    ClosingRoots => {
      self.finish_root_close()
      return GroupCloseComplete
    }
    Cleaning(supervisor, replace_after_cleanup~) => {
      let progress : @rank_group_process.RankGroupProcessCleanupProgress = if supervisor.lifecycle() ==
        Draining ||
        supervisor.lifecycle() == Closing {
        let terminal = try supervisor.progress_close() catch {
          _ => {
            if supervisor.lifecycle() == Faulted {
              self.retain_failure_before_cleanup(supervisor)
              supervisor.begin_cleanup() catch {
                _ => rejected(Cleanup)
              }
              self.state = Cleaning(supervisor, replace_after_cleanup~)
              return GroupCleanupAdvanced
            }
            rejected(Cleanup)
          }
        } noraise {
          value => value
        }
        match terminal {
          ProcessPending => CleanupPending
          _ => CleanupAdvanced
        }
      } else {
        supervisor.progress_cleanup() catch {
          _ => rejected(Cleanup)
        }
      }
      match progress {
        CleanupAdvanced => GroupCleanupAdvanced
        CleanupPending => GroupCleanupPending
        CleanupStuck(rank~) => GroupCleanupStuck(rank~)
        CleanupComplete =>
          if replace_after_cleanup {
            self.state = NoGroup
            GroupReplacementReady
          } else {
            self.state = ClosingRoots
            self.finish_root_close()
            GroupCloseComplete
          }
      }
    }
    _ => rejected(Lifecycle)
  }
}
