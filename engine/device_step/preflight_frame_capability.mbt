///|
fn DeviceStepOwner::preflight_validated_frame(
  self : DeviceStepOwner,
  frame : @worker_wire.ValidatedPlanFrame,
  greedy_only? : Bool = false,
) -> DeviceStepSummary raise DeviceStepError {
  require_lifecycle(self.lifecycle, Ready)
  if self.stage_epoch == 18446744073709551615UL {
    raise StageEpochOverflow
  }
  let sequence_value = frame.sequence_value() catch {
    error => raise Wire(error)
  }
  validate_next_sequence(
    self.has_accepted_predecessor,
    self.accepted_predecessor_value,
    sequence_value,
  )
  preflight_frame_and_write(
    frame,
    self.model_generation_value,
    self.identity,
    self.layout,
    self.limits,
    self.host,
    self.required_capabilities,
    true,
    greedy_only~,
  )
}

///|
/// Consumes one exact preflight summary whose single validation scan already
/// populated reusable host staging. No frame field is rescanned before upload.
fn DeviceStepOwner::stage_preflighted_frame(
  self : DeviceStepOwner,
  frame : @worker_wire.ValidatedPlanFrame,
  summary : DeviceStepSummary,
) -> StagedDeviceStep raise DeviceStepError {
  require_lifecycle(self.lifecycle, Ready)
  let wire_owner = frame.owner_identity() catch { error => raise Wire(error) }
  let wire_epoch = frame.epoch_value() catch { error => raise Wire(error) }
  validate_next_sequence(
    self.has_accepted_predecessor,
    self.accepted_predecessor_value,
    summary.sequence_value,
  )
  if self.stage_epoch == 18446744073709551615UL {
    raise StageEpochOverflow
  }
  upload_owner_transaction(self, summary) catch {
    error => poison_stage_owner(self, error)
  }
  self.stage_epoch += 1UL
  self.staged_summary = summary
  self.staged_plan_owner = None
  self.staged_wire_owner = Some(wire_owner)
  self.staged_wire_epoch = wire_epoch
  self.lifecycle = Staged
  StagedDeviceStep(self, self.stage_epoch)
}
