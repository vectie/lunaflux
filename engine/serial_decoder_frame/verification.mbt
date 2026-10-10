///|
/// Startup output envelope. Ordinary generation retains one last-row sample;
/// all-position verification reserves one result for every input row.
pub fn SerialDecoderFrame::output_capacity(self : Self) -> Int {
  if self.all_position_outputs {
    self.maximum_rows
  } else {
    1
  }
}

///|
pub fn SerialDecoderFrame::output_rows(
  self : Self,
) -> Int raise SerialDecoderError {
  guard self.state is Staged else { raise Busy }
  if !self.pending_output {
    0
  } else if self.all_position_outputs {
    self.pending_rows
  } else {
    1
  }
}

///|
/// Discard successful tentative execution after the enclosing owner restores
/// its device snapshot and retires that restoration. No token is published and
/// no committed request identity/frontier advances. Real execution failures
/// must use retire_stage(execution_failed=true), not this rollback path.
pub fn SerialDecoderFrame::discard_stage(self : Self) -> Unit raise {
  guard self.state is Staged else { raise SerialDecoderError::Busy }
  self.state = Failed
  self.completion_slot.close()
  self.state = Idle
}
