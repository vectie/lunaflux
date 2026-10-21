///|
/// Suspend transport only when no GPU/state transaction needs progress. The
/// prepared/executing phases retain their existing bounded polling contract.
pub async fn DecoderStageServer::wait_idle_receive(self : Self) -> Bool {
  guard !self.busy else { raise StageControlError::InvalidState }
  if !(self.state is Receiving && self.phase is (Idle | Released)) {
    return false
  }
  self.busy = true
  defer {
    self.busy = false
  }
  errdefer {
    self.state = FailedState
  }
  let waited = self.channel.wait_idle_receive(
    allow_idle_eof=self.phase is Released,
  )
  if !self.channel.is_open() {
    self.state = Terminal
  }
  waited
}
