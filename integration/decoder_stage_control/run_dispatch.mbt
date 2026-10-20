///|
/// Flushes the prepared run command without waiting for device completion.
/// This is a separate effect frontier: a coordinator may enqueue its local
/// collective frame once the peer command has left this bounded send lane.
/// Both input-readiness receipts are still required by the caller. Delivery
/// is not acknowledgement, retirement, or permission to publish any output.
pub async fn DecoderStageClient::dispatch_run(
  self : Self,
  local_now_millis~ : UInt64,
) -> Bool {
  guard !self.busy && self.state is Sending(kind) && kind == result_kind else {
    raise StageControlError::InvalidState
  }
  self.busy = true
  defer {
    self.busy = false
  }
  errdefer {
    self.state = FailedState
  }
  self.flush_command(result_kind, local_now_millis)
}

///|
// Shared by ordinary exchange polling and the explicit RUN dispatch frontier.
async fn DecoderStageClient::flush_command(
  self : Self,
  reply : Int,
  now : UInt64,
) -> Bool {
  if !self.channel.poll_send(now) {
    return false
  }
  self.channel.begin_receive(now)
  self.state = Receiving(reply)
  true
}
