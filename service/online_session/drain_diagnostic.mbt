///|
/// Cold teardown diagnostics only. Does not advance owners, read device state,
/// render request contents, or enter the token-step execution path.
pub fn LunaOnlineFramedService::drain_diagnostic(self : Self) -> String {
  let i = self.instance
  let mut lanes = ""
  for index, lane in i.multi.lanes {
    if lane.phase != LunaInstanceIdlePhase {
      lanes += " lane=\{index}:\{Repr(lane.phase)}:cut=\{Repr(lane.terminal_cut)}:failure=\{Repr(lane.failure_kind)}"
    }
  }
  "stream=\{self.stream_phase} drain=\{self.drain_requested} disconnect=\{self.disconnect_requested} maintenance=\{self.maintenance_kind} tickets=\{self.tickets.length()} queue=\{self.queue_count} prepared=\{self.prepared.length()} event_credits=\{self.event_credits.length()} event_pending=\{self.semantic_event_pending} event_taken=\{self.semantic_event_taken} ack_pending=\{self.event_ack_pending} observation=\{self.observation_kind} rejection=\{self.rejection_epoch} instance_phase=\{Repr(i.phase)} instance_maintenance=\{i.maintenance_state} recovery=\{i.recovery_close} failure_mark=\{i.multi.failure_mark_active}:\{i.multi.failure_mark_cursor} retirement_lane=\{i.multi.retirement_lane} event=\{Repr(i.event_kind)} event_lane=\{i.multi.event_lane} has_event=\{i.events.has_event()}" +
  lanes
}
