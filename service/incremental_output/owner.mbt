///|
pub fn IncrementalOutput::new(
  tokenizer : @tokenizer.TokenizerSpec,
  stops : @inference.StopConditions,
  inference : @inference.InferenceLimits,
  special_policy~ : @tokenizer.DecodeSpecialPolicy,
) -> IncrementalOutput raise IncrementalOutputError {
  let stop_count = stops.strings().length()
  let stop_bytes = exact_max_stop_bytes(stops)
  let workspace = new_luna_incremental_output_workspace(
    tokenizer,
    inference,
    LunaIncrementalOutputStepBudget::new(
      LUNA_INCREMENTAL_OUTPUT_MAX_STEP_WORK_UNITS,
    ),
    stop_count,
    stop_bytes,
  )
  let work = workspace.begin(stops, special_policy~)
  while work.progress() is LunaIncrementalOutputPending {

  }
  { lease: work.take_lease() }
}

///|
pub fn IncrementalOutput::is_stopped(self : IncrementalOutput) -> Bool {
  try! self.lease.is_stopped()
}

///|
pub fn IncrementalOutput::is_finished(self : IncrementalOutput) -> Bool {
  try! self.lease.is_finished()
}

///|
pub fn IncrementalOutput::push_token_into(
  self : IncrementalOutput,
  token_id : Int,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> IncrementalOutputStep raise IncrementalOutputError {
  let workspace = self.lease.require_workspace()
  if workspace.lifecycle != OUTPUT_ACTIVE {
    raise invalid(Lifecycle, Terminal)
  }
  let status = self.lease.push_token_status(
    token_id,
    destination,
    destination_offset~,
  )
  if !status.is_ready() {
    match status.0 {
      1 => raise invalid(Lifecycle, Terminal)
      2 => raise invalid(Token, Invalid)
      3 => raise invalid(Utf8, Invalid)
      4 => raise invalid(Destination, TooSmall)
      _ => abort("incremental output scalar status invariant")
    }
  }
  IncrementalOutputStep(status.copied_bytes(), status.stop_matched())
}

///|
pub fn IncrementalOutput::finish_into(
  self : IncrementalOutput,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Int raise IncrementalOutputError {
  let workspace = self.lease.require_workspace()
  if workspace.lifecycle != OUTPUT_ACTIVE {
    raise invalid(Lifecycle, Terminal)
  }
  if workspace.utf8_remaining != 0 {
    raise invalid(Utf8, Invalid)
  }
  destination_available(
    destination,
    destination_offset,
    workspace.pending_count,
  )
  let status = self.lease.finish_into_status(destination, destination_offset~)
  if !status.is_ready() {
    abort("incremental output finish preflight invariant")
  }
  status.copied_bytes()
}

///|
pub fn IncrementalOutput::push_token_without_stops_status(
  self : IncrementalOutput,
  token_id : Int,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> IncrementalOutputStatus {
  self.lease.push_token_without_stops_status(
    token_id,
    destination,
    destination_offset~,
  )
}

///|
pub fn IncrementalOutput::finish_into_status(
  self : IncrementalOutput,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> IncrementalOutputStatus {
  self.lease.finish_into_status(destination, destination_offset~)
}

///|
pub fn IncrementalOutput::push_token_status(
  self : IncrementalOutput,
  token_id : Int,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> IncrementalOutputStatus {
  self.lease.push_token_status(token_id, destination, destination_offset~)
}
