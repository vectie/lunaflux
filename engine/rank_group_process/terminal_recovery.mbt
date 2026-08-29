///|
fn RankGroupProcessSupervisor::stage_terminal_encoded_write(
  self : RankGroupProcessSupervisor,
  rank : Int,
  length : Int,
) -> Unit raise RankGroupProcessError {
  ignore(
    self.wire_transcript.accept_bytes(self.tx, length, self.wire_limits) catch {
      error => self.fail(Wire(error))
    },
  )
  self.children[rank].begin_write(
    self.tx,
    length,
    self.limits.io_timeout_millis,
  ) catch {
    _ => self.terminal_transport_lost(rank)
  }
  self.io_rank = rank
  self.io_phase = 1
}

///|
fn RankGroupProcessSupervisor::stage_terminal(
  self : RankGroupProcessSupervisor,
  rank : Int,
  kind : @rank_group_wire.RankGroupWireKind,
) -> Unit raise RankGroupProcessError {
  let length = @rank_group_wire.encode_rank_group_wire_control_into(
    self.tx,
    self.wire_limits,
    self.binding(rank),
    kind,
    ParentToChild,
    None,
    0UL,
  ) catch {
    error => self.fail(Wire(error))
  }
  self.stage_terminal_encoded_write(rank, length)
}

///|
fn RankGroupProcessSupervisor::stage_abort(
  self : RankGroupProcessSupervisor,
  rank : Int,
) -> Unit raise RankGroupProcessError {
  let slot = self.active_slot()
  let length = @rank_group_wire.encode_rank_group_wire_control_into(
    self.tx,
    self.wire_limits,
    self.binding(rank),
    Abort,
    ParentToChild,
    Some(slot),
    self.active_sequence_value,
  ) catch {
    error => self.fail(Wire(error))
  }
  self.stage_terminal_encoded_write(rank, length)
}

///|
fn RankGroupProcessSupervisor::admit_cooperative_failure(
  self : RankGroupProcessSupervisor,
  failure : RankGroupProcessFailure,
) -> Unit raise RankGroupProcessError {
  if self.failure_value is None {
    self.failure_value = Some(failure)
  }
  try self.protocol.take_failure() catch {
    error => self.fail(Protocol(error))
  } noraise {
    report => self.group_failure = Some(report)
  }
  self.io_rank = -1
  self.io_phase = 0
  self.lifecycle_value = Draining
}

///|
fn RankGroupProcessSupervisor::admit_deadline_failure(
  self : RankGroupProcessSupervisor,
) -> Unit raise RankGroupProcessError {
  if self.failure_value is None {
    self.failure_value = Some(DeadlineExpired)
  }
  try self.protocol.take_failure() catch {
    error => self.fail(Protocol(error))
  } noraise {
    report => self.group_failure = Some(report)
  }
  self.io_rank = -1
  self.io_phase = 0
  self.lifecycle_value = Aborting
}

///|
fn RankGroupProcessSupervisor::begin_deadline_abort(
  self : RankGroupProcessSupervisor,
) -> Unit raise RankGroupProcessError {
  self.admit_deadline_failure()
  self.stage_abort(0)
}

///|
fn RankGroupProcessSupervisor::begin_cooperative_failure(
  self : RankGroupProcessSupervisor,
  failure : RankGroupProcessFailure,
) -> Unit raise RankGroupProcessError {
  self.admit_cooperative_failure(failure)
  self.stage_terminal(0, Drain)
}

///|
fn RankGroupProcessSupervisor::accept_terminal_response(
  self : RankGroupProcessSupervisor,
  rank : Int,
  length : Int,
) -> Unit raise RankGroupProcessError {
  let header = self.wire_transcript.accept_bytes(
    self.rx,
    length,
    self.wire_limits,
  ) catch {
    error => self.fail(Wire(error))
  }
  if self.lifecycle_value == Aborting {
    if header.kind() != Aborted {
      self.fail(Wire(ProtocolFault(Kind)))
    }
    if rank + 1 < self.children.length() {
      self.stage_abort(rank + 1)
    } else {
      self.lifecycle_value = Draining
      self.stage_terminal(0, Drain)
    }
    return
  }
  if self.lifecycle_value == Draining {
    if header.kind() != Drained {
      self.fail(Wire(ProtocolFault(Kind)))
    }
    if !self.healthy_terminal {
      let lease = self.protocol.drain_lease(rank) catch {
        error => self.fail(Protocol(error))
      }
      let ack = lease.ack() catch { error => self.fail(Protocol(error)) }
      self.protocol.accept_drain_ack(ack) catch {
        error => self.fail(Protocol(error))
      }
    }
    if rank + 1 < self.children.length() {
      self.stage_terminal(rank + 1, Drain)
    } else {
      self.lifecycle_value = Closing
      self.stage_terminal(0, Close)
    }
    return
  }
  if self.lifecycle_value != Closing || header.kind() != Closed {
    self.fail(Wire(ProtocolFault(Kind)))
  }
  if rank + 1 < self.children.length() {
    self.stage_terminal(rank + 1, Close)
  } else if self.healthy_terminal {
    self.protocol.close() catch {
      error => self.fail(Protocol(error))
    }
    self.healthy_terminal = false
    self.io_rank = -1
    self.io_phase = 0
    self.begin_child_cleanup()
  } else {
    self.protocol.close_faulted() catch {
      error => self.fail(Protocol(error))
    }
    self.io_rank = -1
    self.io_phase = 0
    self.lifecycle_value = Faulted
  }
}

///|
/// Advances one healthy terminal wire transition without requiring an active
/// scheduler plan. This is valid only after `begin_close` from idle Ready.
pub fn RankGroupProcessSupervisor::progress_close(
  self : RankGroupProcessSupervisor,
) -> RankGroupProcessProgress raise RankGroupProcessError {
  if !self.healthy_terminal ||
    (self.lifecycle_value != Draining && self.lifecycle_value != Closing) {
    self.reject(Lifecycle)
  }
  let rank = self.io_rank
  if rank < 0 || self.io_phase == 0 {
    self.reject(Lifecycle)
  }
  if self.io_phase == 1 {
    let written = self.children[rank].progress_write() catch {
      _ => self.terminal_transport_lost(rank)
    }
    if !written {
      return ProcessPending
    }
    self.children[rank].begin_read(
      self.rx,
      self.rx_staging,
      self.limits.io_timeout_millis,
    ) catch {
      _ => self.terminal_transport_lost(rank)
    }
    self.io_phase = 2
    return ProcessAdvanced
  }
  let received = self.children[rank].progress_read() catch {
    _ => self.terminal_transport_lost(rank)
  }
  match received {
    None => ProcessPending
    Some(length) => {
      self.io_phase = 0
      self.accept_terminal_response(rank, length)
      ProcessAdvanced
    }
  }
}
