///|
pub fn RemoteGroup::completion_frame(
  self : RemoteGroup,
  sequence : UInt64,
) -> @worker_wire.ValidatedCompletionFrame raise {
  if self.lifecycle != CompletionReady || sequence != self.sequence {
    raise State
  }
  self.completion_frame.current()
}

///|
pub fn RemoteGroup::graph_telemetry(
  self : RemoteGroup,
  sequence : UInt64,
) -> @worker_wire.WorkerGraphTelemetry raise {
  if self.lifecycle != CompletionReady || sequence != self.sequence {
    raise State
  }
  self.reports[0].current()
}

///|
pub fn RemoteGroup::accept_completion(
  self : RemoteGroup,
  plan : @worker_protocol.SubmittedSchedulePlan,
  completion : @worker_protocol.SubmittedCompletion,
) -> Unit raise {
  if self.lifecycle != CompletionReady {
    raise State
  }
  let submitted = self.protocol.reauthenticate_submitted(plan)
  ignore(
    self.protocol.complete_leader(submitted.lease(plan, 0), plan, completion),
  )
  self.lifecycle = Accepted
}

///|
/// Only after the scheduler has committed its accepted result. Never rereads
/// the scheduler plan after that owner has advanced its epoch.
pub fn RemoteGroup::retire_after_scheduler_commit(
  self : RemoteGroup,
  sequence : UInt64,
) -> Unit raise {
  if self.lifecycle != Accepted || sequence != self.sequence {
    raise State
  }
  self.protocol.retire_completed(
    self.protocol.reauthenticate_completed(self.epoch),
  )
  self.sequence = 0UL
  self.lifecycle = Ready
  if self.mailboxes[0].is_failed() || self.mailboxes[1].is_failed() {
    self.transport_lost()
  }
}

///|
pub fn RemoteGroup::has_submission(
  self : RemoteGroup,
  sequence : UInt64,
) -> Bool {
  sequence != 0UL &&
  sequence == self.sequence &&
  self.lifecycle != Ready &&
  self.lifecycle != Closed
}

///|
pub fn RemoteGroup::has_received(self : RemoteGroup, sequence : UInt64) -> Bool {
  self.lifecycle == CompletionReady && sequence == self.sequence
}

///|
/// Graceful drain/close of both ranks. Closed is published only after both
/// authenticated workers report deterministic resource release.
pub fn RemoteGroup::begin_close(self : RemoteGroup) -> Unit raise {
  if self.lifecycle != Ready {
    raise State
  }
  errdefer self.poison()
  self.start_deadline()
  for rank in 0..<2 {
    let length = @wire.encode_rank_group_wire_control_into(
      self.tx,
      self.limits,
      self.bindings[rank],
      Drain,
      ParentToChild,
      None,
      0UL,
    )
    self.send(rank, length)
    self.phases[rank] = 0
  }
  self.lifecycle = Closing
}

///|
pub fn RemoteGroup::progress_close(self : RemoteGroup) -> Bool raise {
  if self.lifecycle != Closing {
    raise State
  }
  errdefer self.poison()
  self.check_live()
  let rank = self.cursor
  self.cursor = 1 - rank
  match self.phases[rank] {
    0 | 3 =>
      if self.mailboxes[rank].poll_write() {
        self.mailboxes[rank].begin_read()
        self.phases[rank] += 1
      }
    1 | 4 =>
      match self.mailboxes[rank].poll_read(self.rx) {
        Some(length) => {
          let kind = self.receive(rank, length).kind()
          if self.phases[rank] == 1 {
            if kind != Drained {
              raise Protocol
            }
            self.phases[rank] = 2
          } else {
            if kind != Closed {
              raise Protocol
            }
            self.phases[rank] = 5
          }
        }
        None => ()
      }
    _ => ()
  }
  if self.phases[0] == 2 && self.phases[1] == 2 {
    for rank in 0..<2 {
      let length = @wire.encode_rank_group_wire_control_into(
        self.tx,
        self.limits,
        self.bindings[rank],
        Close,
        ParentToChild,
        None,
        0UL,
      )
      self.send(rank, length)
      self.phases[rank] = 3
    }
  }
  if self.phases[0] == 5 && self.phases[1] == 5 {
    self.protocol.close()
    self.lifecycle = Closed
    true
  } else {
    false
  }
}
