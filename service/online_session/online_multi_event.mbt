///|
fn LunaOnlineInstance::ack_multi_event_credit(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
  event_epoch : UInt64,
) -> Unit raise LunaOnlineInstanceError {
  let lane = self.require_active_multi_ticket(ticket)
  self.require_owner()
  if self.multi.event_lane != ticket.lane ||
    !self.event_taken ||
    event_epoch != self.event_epoch {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
  }
  let view = self.events.view() catch {
    _ =>
      raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
  }
  match self.event_kind {
    AcceptedEvent => {
      if lane.accepted_acked {
        raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
      }
      self.events.retire(view) catch {
        _ =>
          raise LunaOnlineInstanceFailed(
            LunaOnlineInvalid(LunaOnlineEventCredit),
          )
      }
      self.event_taken = false
      self.multi.event_lane = -1
      lane.accepted_acked = true
    }
    TokenEvent => {
      self.events.retire(view) catch {
        _ =>
          raise LunaOnlineInstanceFailed(
            LunaOnlineInvalid(LunaOnlineEventCredit),
          )
      }
      self.event_taken = false
      self.multi.event_lane = -1
    }
    CompletionUsageEvent => {
      let usage = view.usage() catch {
        _ =>
          raise LunaOnlineInstanceFailed(
            LunaOnlineInvalid(LunaOnlineEventCredit),
          )
      }
      ignore(
        try! self.events.replace_usage_with_completed(
          usage,
          match lane.finish_kind {
            FinishStopSequence => StopSequence
            FinishTokenLimit => TokenLimit
            FinishCancelled => Cancelled
            FinishPending => abort("Luna online multi finish invariant")
          },
          lane.output,
          tail_offset=0,
          tail_length=lane.final_tail_length,
        ),
      )
      self.mark_event_published(CompletedEvent)
    }
    CompletedEvent => {
      self.events.retire(view) catch {
        _ =>
          raise LunaOnlineInstanceFailed(
            LunaOnlineInvalid(LunaOnlineEventCredit),
          )
      }
      self.event_taken = false
      self.multi.event_lane = -1
      lane.phase = LunaRequestReleaseReadyPhase
      self.multi.retirement_lane = ticket.lane
    }
    FailureUsageEvent => {
      let usage = view.usage() catch {
        _ =>
          raise LunaOnlineInstanceFailed(
            LunaOnlineInvalid(LunaOnlineEventCredit),
          )
      }
      let code = match lane.failure_kind {
        DeadlineLatchedFailure => self.deadline_code
        OutputLatchedFailure => self.output_code
        WorkerLatchedFailure => self.worker_code
        NoLatchedFailure => abort("Luna online multi failure invariant")
      }
      ignore(
        try! self.events.replace_usage_with_failed(
          usage,
          code,
          code_offset=0,
          code_length=code.length(),
          retryable=lane.failure_kind == WorkerLatchedFailure,
        ),
      )
      self.mark_event_published(FailedEvent)
    }
    FailedEvent => {
      self.events.retire(view) catch {
        _ =>
          raise LunaOnlineInstanceFailed(
            LunaOnlineInvalid(LunaOnlineEventCredit),
          )
      }
      self.event_taken = false
      self.multi.event_lane = -1
      lane.phase = LunaRequestReleaseReadyPhase
      self.multi.retirement_lane = ticket.lane
    }
    NoSessionEvent => abort("Luna online multi event invariant")
  }
}
