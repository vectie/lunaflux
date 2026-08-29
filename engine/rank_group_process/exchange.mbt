///|
fn[T] RankGroupProcessSupervisor::fail_active(
  self : RankGroupProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
  rank : Int,
  failure : RankGroupProcessFailure,
) -> T raise RankGroupProcessError {
  self.protocol.report_supervisor_failure(plan, rank) catch {
    _ => ()
  }
  self.fail(failure)
}

///|
fn[T] RankGroupProcessSupervisor::active_transport_lost(
  self : RankGroupProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
  rank : Int,
) -> T raise RankGroupProcessError {
  self.protocol.report_supervisor_rank_loss(plan, rank) catch {
    _ => ()
  }
  self.fail(TransportLost(rank~))
}

///|
fn[T] RankGroupProcessSupervisor::terminal_transport_lost(
  self : RankGroupProcessSupervisor,
  rank : Int,
) -> T raise RankGroupProcessError {
  self.fail(TransportLost(rank~))
}

///|
fn RankGroupProcessSupervisor::stage_active_encoded_write(
  self : RankGroupProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
  rank : Int,
  length : Int,
) -> Unit raise RankGroupProcessError {
  ignore(
    self.wire_transcript.accept_bytes(self.tx, length, self.wire_limits) catch {
      error => self.fail_active(plan, rank, Wire(error))
    },
  )
  self.children[rank].begin_write_until(
    self.tx,
    length,
    self.active_frame_deadline_millis,
  ) catch {
    _ => self.active_transport_lost(plan, rank)
  }
  self.io_rank = rank
  self.io_phase = 1
}

///|
fn RankGroupProcessSupervisor::prepare_active_frame_deadline(
  self : RankGroupProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> Bool raise RankGroupProcessError {
  let submitted = self.reauthenticate_plan(plan)
  let now = self.clock.now_millis() catch { _ => self.fail(Clock) }
  if now >= self.active_deadline_millis {
    let expired = self.protocol.expire_if_due(submitted, now) catch {
      error => self.fail(Protocol(error))
    }
    if !expired {
      self.fail(Clock)
    }
    self.begin_deadline_abort()
    return false
  }
  if now > 0x7fffffffffffffffUL - self.limits.io_timeout_millis.to_uint64() {
    self.fail(Clock)
  }
  let io_deadline = now + self.limits.io_timeout_millis.to_uint64()
  let frame_deadline = if io_deadline < self.active_deadline_millis {
    io_deadline
  } else {
    self.active_deadline_millis
  }
  if frame_deadline > 0x7fffffffffffffffUL {
    self.fail(Clock)
  }
  self.active_frame_deadline_millis = frame_deadline.reinterpret_as_int64()
  true
}

///|
fn RankGroupProcessSupervisor::stage_submit(
  self : RankGroupProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
  rank : Int,
) -> Unit raise RankGroupProcessError {
  if !self.prepare_active_frame_deadline(plan) {
    return
  }
  let frame = self.plan_frame.current() catch {
    error => self.fail_active(plan, rank, WorkerWire(error))
  }
  let payload_length = frame.length() catch {
    error => self.fail_active(plan, rank, WorkerWire(error))
  }
  if payload_length != self.encoded_plan_length {
    self.reject(Submission)
  }
  let slot = self.active_slot()
  let length = @rank_group_wire.seal_rank_group_wire_embedded_payload(
    self.tx,
    self.wire_limits,
    self.binding(rank),
    Submit,
    ParentToChild,
    Some(slot),
    self.active_sequence_value,
    payload_length,
  ) catch {
    error => self.fail_active(plan, rank, Wire(error))
  }
  self.stage_active_encoded_write(plan, rank, length)
}

///|
fn RankGroupProcessSupervisor::stage_poll(
  self : RankGroupProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
  rank : Int,
) -> Unit raise RankGroupProcessError {
  if !self.prepare_active_frame_deadline(plan) {
    return
  }
  let slot = self.active_slot()
  let length = @rank_group_wire.encode_rank_group_wire_control_into(
    self.tx,
    self.wire_limits,
    self.binding(rank),
    PollComplete,
    ParentToChild,
    Some(slot),
    self.active_sequence_value,
  ) catch {
    error => self.fail_active(plan, rank, Wire(error))
  }
  self.stage_active_encoded_write(plan, rank, length)
}

///|
/// Authenticates the exact scheduler plan owner through RankGroupOwner and
/// starts one monotonic-deadline group exchange.
pub fn RankGroupProcessSupervisor::begin(
  self : RankGroupProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> Unit raise RankGroupProcessError {
  self.require_lifecycle(Ready)
  let started = self.clock.now_millis() catch {
    _ => {
      self.protocol.report_begin_host_failure(plan) catch {
        _ => ()
      }
      self.fail(Clock)
    }
  }
  let duration = self.limits.collective_exchange_timeout_millis.to_uint64()
  if started > 0xffffffffffffffffUL - duration {
    self.protocol.report_begin_host_failure(plan) catch {
      _ => ()
    }
    self.fail(Clock)
  }
  let deadline_millis = started + duration
  if self.exchange_epoch == 0xffffffffffffffffUL {
    raise Invalid(Sequence)
  }
  let sequence = plan.sequence_wire() catch { _ => self.reject(Submission) }
  let slot = self.slot_for_plan(plan)
  ignore(
    self.protocol.begin(
      plan,
      started_tick=started,
      deadline_tick=deadline_millis,
    ) catch {
      error => self.fail(Protocol(error))
    },
  )
  let encoded = self.plan_frame.encode_plan(plan) catch {
    error => self.fail_active(plan, 0, WorkerWire(error))
  }
  let encoded_length = encoded.length() catch {
    error => self.fail_active(plan, 0, WorkerWire(error))
  }
  self.exchange_epoch += 1UL
  self.active_slot_code = if slot == A { 0 } else { 1 }
  self.active_sequence_value = sequence
  self.encoded_plan_length = encoded_length
  self.active_deadline_millis = deadline_millis
  self.active_frame_deadline_millis = 0L
  self.leader_received = false
  self.graph_telemetry.clear()
  self.next_rank = 0
  for rank in 0..<self.rank_done.length() {
    self.rank_done[rank] = false
    self.rank_sidecar_pending[rank] = false
  }
  self.lifecycle_value = Submitting
  self.stage_submit(plan, 0)
}

///|
fn RankGroupProcessSupervisor::next_unfinished_rank(
  self : RankGroupProcessSupervisor,
) -> Int {
  for offset in 0..<self.rank_done.length() {
    let rank = (self.next_rank + offset) % self.rank_done.length()
    if !self.rank_done[rank] {
      self.next_rank = (rank + 1) % self.rank_done.length()
      return rank
    }
  }
  -1
}

///|
fn RankGroupProcessSupervisor::accept_response(
  self : RankGroupProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
  rank : Int,
  length : Int,
) -> Bool raise RankGroupProcessError {
  let submitted = self.reauthenticate_plan(plan)
  let header = self.wire_transcript.accept_bytes(
    self.rx,
    length,
    self.wire_limits,
  ) catch {
    error => self.fail_active(plan, rank, Wire(error))
  }
  match header.kind() {
    Pending => return false
    PlanFailed(failure) => {
      match failure {
        CollectiveFailure => {
          self.protocol.report_collective_failure(submitted, rank) catch {
            error => self.fail(Protocol(error))
          }
          self.begin_cooperative_failure(CollectiveFailed(rank~))
        }
        RankExecutionFailure => {
          self.protocol.report_rank_execution_failure(submitted, rank) catch {
            error => self.fail(Protocol(error))
          }
          self.begin_cooperative_failure(RankExecutionFailed(rank~))
        }
      }
      return false
    }
    Complete => ()
    _ => self.fail_active(plan, rank, Wire(ProtocolFault(Kind)))
  }
  if rank == 0 {
    let payload_length = header.payload_length()
    let header_bytes = @rank_group_wire.rank_group_wire_header_bytes()
    for index in 0..<payload_length {
      self.rx[index] = self.rx[header_bytes + index]
    }
    let completion = self.completion_frame.load(self.rx, payload_length) catch {
      error => self.fail_active(plan, rank, WorkerWire(error))
    }
    completion.validate_for(plan) catch {
      error => self.fail_active(plan, rank, WorkerWire(error))
    }
    self.leader_received = true
  } else {
    let ack = submitted.lease(plan, rank).follower_ack() catch {
        error => self.fail(Protocol(error))
      }
    self.protocol.accept_follower_ack(submitted, ack) catch {
      error => self.fail(Protocol(error))
    }
  }
  self.rank_sidecar_pending[rank] = true
  true
}

///|
fn RankGroupProcessSupervisor::all_rank_results_ready(
  self : RankGroupProcessSupervisor,
) -> Bool {
  if !self.leader_received {
    return false
  }
  for done in self.rank_done {
    if !done {
      return false
    }
  }
  true
}

///|
/// Advances at most one native rank write/read transition globally. The exact
/// scheduler plan must be reauthenticated on every turn.
pub fn RankGroupProcessSupervisor::progress(
  self : RankGroupProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> RankGroupProcessProgress raise RankGroupProcessError {
  let terminal = self.lifecycle_value == Aborting ||
    self.lifecycle_value == Draining ||
    self.lifecycle_value == Closing
  if self.lifecycle_value != Submitting &&
    self.lifecycle_value != Polling &&
    !terminal {
    self.reject(Lifecycle)
  }
  if terminal {
    self.authenticate_active_scalars(plan)
  } else {
    ignore(self.reauthenticate_plan(plan))
  }
  if self.io_phase == 0 {
    if self.lifecycle_value == Polling {
      let rank = self.next_unfinished_rank()
      if rank < 0 {
        if self.all_rank_results_ready() {
          self.lifecycle_value = CompletionReady
          return ProcessCompletionReady
        }
        self.reject(Completion)
      }
      self.stage_poll(plan, rank)
      return ProcessAdvanced
    }
    self.reject(Lifecycle)
  }
  let rank = self.io_rank
  if self.io_phase == 1 {
    let written = self.children[rank].progress_write() catch {
      _ =>
        if terminal {
          self.terminal_transport_lost(rank)
        } else {
          self.active_transport_lost(plan, rank)
        }
    }
    if !written {
      return ProcessPending
    }
    if self.lifecycle_value == Submitting {
      self.io_phase = 0
      self.io_rank = -1
      if rank + 1 < self.children.length() {
        self.stage_submit(plan, rank + 1)
      } else {
        self.lifecycle_value = Polling
      }
      return ProcessAdvanced
    }
    if terminal {
      self.children[rank].begin_read(
        self.rx,
        self.rx_staging,
        self.limits.io_timeout_millis,
      ) catch {
        _ => self.terminal_transport_lost(rank)
      }
    } else {
      self.children[rank].begin_read_until(
        self.rx,
        self.rx_staging,
        self.active_frame_deadline_millis,
      ) catch {
        _ => self.active_transport_lost(plan, rank)
      }
    }
    self.io_phase = 2
    return ProcessAdvanced
  }
  let read_phase = self.io_phase
  let received = self.children[rank].progress_read() catch {
    _ =>
      if terminal {
        self.terminal_transport_lost(rank)
      } else {
        self.active_transport_lost(plan, rank)
      }
  }
  match received {
    None => ProcessPending
    Some(length) => {
      if read_phase == 3 {
        self.io_phase = 0
        self.accept_graph_telemetry_bytes_in_place(plan, rank, length)
        self.io_rank = -1
        if self.all_rank_results_ready() {
          self.lifecycle_value = CompletionReady
          return ProcessCompletionReady
        }
        return ProcessAdvanced
      }
      self.io_phase = 0
      if terminal {
        self.accept_terminal_response(rank, length)
        return ProcessAdvanced
      }
      let sidecar_required = self.accept_response(plan, rank, length)
      if self.lifecycle_value == Draining {
        return ProcessAdvanced
      }
      if sidecar_required {
        self.children[rank].begin_read_until(
          self.rx,
          self.rx_staging,
          self.active_frame_deadline_millis,
        ) catch {
          _ => self.active_transport_lost(plan, rank)
        }
        self.io_phase = 3
        return ProcessAdvanced
      }
      self.io_rank = -1
      if self.all_rank_results_ready() {
        self.lifecycle_value = CompletionReady
        ProcessCompletionReady
      } else {
        ProcessAdvanced
      }
    }
  }
}
