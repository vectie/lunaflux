///|
pub fn RemoteGroup::begin(
  self : RemoteGroup,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> Unit raise {
  if self.lifecycle != Ready {
    raise State
  }
  let identity = plan.buffer_identity()
  let slot : @wire.RankGroupWireSlot = if identity.same_owner(
      self.identities[0],
    ) {
    A
  } else if identity.same_owner(self.identities[1]) {
    B
  } else {
    raise Binding
  }
  errdefer {
    self.protocol.report_begin_host_failure(plan) catch {
      _ => ()
    }
    self.poison()
  }
  self.probe_idle_health()
  self.start_deadline()
  ignore(
    self.protocol.begin(
      plan,
      started_tick=self.last_tick,
      deadline_tick=self.deadline,
    ),
  )
  let frame = self.plan_frame.encode_plan(plan)
  let length = frame.length()
  frame.copy_to(self.payload, destination_offset=0)
  self.sequence = plan.sequence_wire()
  self.slot = slot
  if self.epoch == 0xffffffffffffffffUL {
    raise State
  }
  self.epoch += 1UL
  for rank in 0..<2 {
    let length = @wire.encode_rank_group_wire_payload_into(
      self.tx,
      self.limits,
      self.bindings[rank],
      Submit,
      ParentToChild,
      Some(slot),
      self.sequence,
      self.payload,
      length,
    )
    self.send(rank, length)
    self.phases[rank] = 0
    self.reports[rank].clear()
  }
  self.cursor = 0
  self.lifecycle = Executing
}

///|
fn RemoteGroup::stage_poll(self : RemoteGroup, rank : Int) -> Unit raise {
  let length = @wire.encode_rank_group_wire_control_into(
    self.tx,
    self.limits,
    self.bindings[rank],
    PollComplete,
    ParentToChild,
    Some(self.slot),
    self.sequence,
  )
  self.send(rank, length)
  self.phases[rank] = 2
}

///|
fn RemoteGroup::response(
  self : RemoteGroup,
  plan : @worker_protocol.SubmittedSchedulePlan,
  rank : Int,
  length : Int,
) -> Unit raise {
  let submitted = self.protocol.reauthenticate_submitted(plan)
  let header = self.receive(rank, length)
  match header.kind() {
    Pending => {
      self.phases[rank] = 1
      return
    }
    PlanFailed(reason) => {
      match reason {
        CollectiveFailure =>
          self.protocol.report_collective_failure(submitted, rank)
        RankExecutionFailure =>
          self.protocol.report_rank_execution_failure(submitted, rank)
      }
      raise Protocol
    }
    Complete => ()
    _ => raise Protocol
  }
  if rank == 0 {
    let count = header.payload_length()
    let offset = @wire.rank_group_wire_header_bytes()
    for i in 0..<count {
      self.payload[i] = self.rx[offset + i]
    }
    self.completion_frame.load(self.payload, count).validate_for(plan)
  } else {
    self.protocol.accept_follower_ack(
      submitted,
      submitted.lease(plan, rank).follower_ack(),
    )
  }
  self.mailboxes[rank].begin_read()
  self.phases[rank] = 4
}

///|
/// At most one mailbox transition per turn. Any rank fault revokes both lanes.
pub fn RemoteGroup::progress(
  self : RemoteGroup,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> Bool raise {
  if self.lifecycle != Executing {
    raise State
  }
  let submitted = self.protocol.reauthenticate_submitted(plan)
  let rank = self.cursor
  self.cursor = 1 - rank
  errdefer {
    self.protocol.report_supervisor_failure(plan, rank) catch {
      _ => ()
    }
    self.poison()
  }
  for i in 0..<2 {
    if self.mailboxes[i].is_failed() {
      self.protocol.report_rank_loss(submitted, i)
      raise RankLost(i)
    }
  }
  let now = self.now()
  if self.protocol.expire_if_due(submitted, now) {
    raise Deadline
  }
  match self.phases[rank] {
    0 => if self.mailboxes[rank].poll_write() { self.phases[rank] = 1 }
    1 => self.stage_poll(rank)
    2 =>
      if self.mailboxes[rank].poll_write() {
        self.mailboxes[rank].begin_read()
        self.phases[rank] = 3
      }
    3 =>
      match self.mailboxes[rank].poll_read(self.rx) {
        Some(length) => self.response(plan, rank, length)
        None => ()
      }
    4 =>
      match self.mailboxes[rank].poll_read(self.rx) {
        Some(length) => {
          let report = @worker_wire.decode_worker_graph_telemetry(
            self.rx,
            length,
          )
          if report.sequence_value() != self.sequence {
            raise Protocol
          }
          self.reports[rank].store(report)
          self.phases[rank] = 5
        }
        None => ()
      }
    _ => ()
  }
  if self.phases[0] != 5 || self.phases[1] != 5 {
    return false
  }
  let left = self.reports[0].current()
  let right = self.reports[1].current()
  if left.selected_path() != right.selected_path() ||
    left.graph_hits() != right.graph_hits() ||
    left.graph_misses() != right.graph_misses() ||
    left.has_selected_shape_class() != right.has_selected_shape_class() {
    raise Protocol
  }
  if left.has_selected_shape_class() {
    let a = left.selected_shape_class()
    let b = right.selected_shape_class()
    if a.phase() != b.phase() ||
      a.batch_rows() != b.batch_rows() ||
      a.query_tokens_per_row() != b.query_tokens_per_row() ||
      a.context_tokens() != b.context_tokens() ||
      a.token_rows() != b.token_rows() {
      raise Protocol
    }
  }
  self.lifecycle = CompletionReady
  true
}
