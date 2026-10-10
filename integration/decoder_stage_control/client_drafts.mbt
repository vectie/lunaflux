///|
/// Request prediction only after the preceding state commit/effect retired.
/// This does not prepare or run the target model and does not advance KV.
pub fn DecoderStageClient::predict(
  self : Self,
  local_now_millis~ : UInt64,
  committed_seed? : Int,
) -> Unit raise {
  guard !self.busy &&
    self.state is Idle &&
    self.epoch > 0UL &&
    self.maximum_draft_tokens > 0 else {
    raise StageControlError::InvalidState
  }
  match committed_seed {
    None => self.command(predict_kind, drafts_kind, 0, local_now_millis)
    Some(seed) => {
      guard seed >= 0 && seed <= self.maximum_token else {
        raise StageControlError::InvalidMessage
      }
      // This is an execution input, not a digest or a diagnostic exchange.
      header(self.message, predict_kind, self.epoch, 1, seed)
      self.state = FailedState
      self.channel.begin_send(
        self.message,
        control_header_bytes,
        local_now_millis,
      )
      self.state = Sending(drafts_kind)
    }
  }
}

///|
pub fn DecoderStageClient::draft_count(self : Self) -> Int raise {
  guard self.state is DraftsReady else { raise StageControlError::InvalidState }
  self.draft_rows
}

///|
/// Borrowed receipt storage stays valid until prepare/release/close; no second
/// exchange, checksum or device readback is performed by these accessors.
pub fn DecoderStageClient::draft_token_at(self : Self, row : Int) -> Int raise {
  guard self.state is DraftsReady && row >= 0 && row < self.draft_rows else {
    raise StageControlError::InvalidState
  }
  read_i32(self.message, control_header_bytes + row * 4)
}
