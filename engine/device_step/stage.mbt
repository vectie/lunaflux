///|
fn require_lifecycle(
  actual : DeviceStepLifecycle,
  expected : DeviceStepLifecycle,
) -> Unit raise DeviceStepError {
  if actual != expected {
    raise InvalidLifecycle(expected~, actual~)
  }
}

///|
fn poison_stage_owner(
  owner : DeviceStepOwner,
  error : DeviceStepError,
) -> Unit raise DeviceStepError {
  owner.lifecycle = Poisoned
  raise error
}

///|
fn validate_next_sequence(
  has_accepted_predecessor : Bool,
  accepted_predecessor_value : UInt64,
  sequence_value : UInt64,
) -> Unit raise DeviceStepError {
  if has_accepted_predecessor {
    if accepted_predecessor_value == 18446744073709551615UL ||
      sequence_value != accepted_predecessor_value + 1UL {
      raise InvalidPlan(PlanSequence)
    }
  } else if sequence_value != 1UL {
    raise InvalidPlan(PlanSequence)
  }
}

///|
fn preflight_stage_then_write(
  plan : @worker_protocol.SubmittedSchedulePlan,
  expected_generation : UInt64,
  expected_identity : @spec.ModelIdentity,
  layout : @device_layout.DeviceKvLayout,
  limits : DeviceStepLimits,
  host : DeviceStepHostBuffers,
  required_capabilities : FixedArray[@plan.KernelCapabilityId],
  has_accepted_predecessor : Bool,
  accepted_predecessor_value : UInt64,
  stage_epoch : UInt64,
  greedy_only? : Bool = false,
) -> DeviceStepSummary raise DeviceStepError {
  if stage_epoch == 18446744073709551615UL {
    raise StageEpochOverflow
  }
  let sequence_value = plan.sequence_wire() catch {
    error => raise protocol_error(error)
  }
  validate_next_sequence(
    has_accepted_predecessor, accepted_predecessor_value, sequence_value,
  )
  preflight_and_write(
    plan,
    expected_generation,
    expected_identity,
    layout,
    limits,
    host,
    required_capabilities,
    true,
    greedy_only~,
  )
}

///|
fn publish_counts(
  host : DeviceStepHostBuffers,
  summary : DeviceStepSummary,
) -> Unit {
  write_i32(host.counts, 0, summary.prefill_count)
  write_i32(host.counts, 1, summary.decode_count)
  write_i32(host.counts, 2, summary.row_count)
  write_i32(host.counts, 3, summary.token_count)
  write_i32(host.counts, 4, summary.page_count)
}

///|
fn copy_operand(
  owner : DeviceStepOwner,
  stage : DeviceStepStage,
  allocation : @device.Allocation,
  host : FixedArray[Byte],
  cells : Int,
) -> Unit raise DeviceStepError {
  owner.context.copy_from_fixed_host(
    allocation,
    host,
    0L,
    0L,
    byte_capacity(cells),
  ) catch {
    error => raise Device(stage, error)
  }
}

///|
fn upload_owner_transaction(
  owner : DeviceStepOwner,
  summary : DeviceStepSummary,
) -> Unit raise DeviceStepError {
  copy_operand(
    owner,
    TokenIds,
    owner.resources.token_ids,
    owner.host.token_ids,
    summary.token_count,
  )
  copy_operand(
    owner,
    TokenPositions,
    owner.resources.token_positions,
    owner.host.token_positions,
    summary.token_count,
  )
  copy_operand(
    owner,
    QueryRowOffsets,
    owner.resources.query_row_offsets,
    owner.host.query_row_offsets,
    summary.row_count + 1,
  )
  copy_operand(
    owner,
    SequenceLengths,
    owner.resources.sequence_lengths,
    owner.host.sequence_lengths,
    summary.row_count,
  )
  copy_operand(
    owner,
    PageOffsets,
    owner.resources.page_offsets,
    owner.host.page_offsets,
    summary.row_count + 1,
  )
  copy_operand(
    owner,
    PhysicalPageIndices,
    owner.resources.physical_page_indices,
    owner.host.physical_page_indices,
    summary.page_count,
  )
  publish_counts(owner.host, summary)
  copy_operand(
    owner,
    Counts,
    owner.resources.counts,
    owner.host.counts,
    COUNTS_LENGTH,
  )
}

///|
fn DeviceStepOwner::stage_with_sampling_mode(
  self : DeviceStepOwner,
  plan : @worker_protocol.SubmittedSchedulePlan,
  greedy_only : Bool,
) -> StagedDeviceStep raise DeviceStepError {
  require_lifecycle(self.lifecycle, Ready)
  let plan_owner = plan.buffer_identity() catch {
    error => raise protocol_error(error)
  }
  let summary = preflight_stage_then_write(
    plan,
    self.model_generation_value,
    self.identity,
    self.layout,
    self.limits,
    self.host,
    self.required_capabilities,
    self.has_accepted_predecessor,
    self.accepted_predecessor_value,
    self.stage_epoch,
    greedy_only~,
  )
  upload_owner_transaction(self, summary) catch {
    error => poison_stage_owner(self, error)
  }
  self.stage_epoch = self.stage_epoch + 1UL
  self.staged_summary = summary
  self.staged_plan_owner = Some(plan_owner)
  self.staged_wire_owner = None
  self.staged_wire_epoch = 0UL
  self.lifecycle = Staged
  StagedDeviceStep(self, self.stage_epoch)
}

///|
/// Validates and uploads one submitted plan into reusable worker-owned storage.
/// Validation and host staging share one bounded scan; no device write occurs
/// until that scan succeeds. Payload operands upload first and real counts
/// publish last. Any copy failure poisons the owner, so a stale prior count
/// publication can never be dispatched through this capability.
pub fn DeviceStepOwner::stage(
  self : DeviceStepOwner,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> StagedDeviceStep raise DeviceStepError {
  self.stage_with_sampling_mode(plan, false)
}

///|
/// Validates and uploads one canonical received plan frame without requiring
/// any scheduler heap-owner capability in the worker process. The same
/// transaction and poison rules as `stage` apply.
fn DeviceStepOwner::stage_frame_with_sampling_mode(
  self : DeviceStepOwner,
  frame : @worker_wire.ValidatedPlanFrame,
  greedy_only : Bool,
) -> StagedDeviceStep raise DeviceStepError {
  self.stage_preflighted_frame(
    frame,
    self.preflight_validated_frame(frame, greedy_only~),
  )
}

///|
pub fn DeviceStepOwner::stage_frame(
  self : DeviceStepOwner,
  frame : @worker_wire.ValidatedPlanFrame,
) -> StagedDeviceStep raise DeviceStepError {
  self.stage_frame_with_sampling_mode(frame, false)
}

///|
fn StagedDeviceStep::authenticate(
  self : StagedDeviceStep,
  owner : DeviceStepOwner,
) -> Unit raise DeviceStepError {
  if !physical_equal(self.0, owner) {
    raise StageOwnerMismatch
  }
  require_lifecycle(owner.lifecycle, Staged)
  if self.1 != owner.stage_epoch {
    raise StaleStage
  }
}

///|
/// Retires one staged publication after its synchronous execution has ended.
pub fn DeviceStepOwner::finish(
  self : DeviceStepOwner,
  staged : StagedDeviceStep,
) -> Unit raise DeviceStepError {
  staged.authenticate(self)
  self.has_accepted_predecessor = true
  self.accepted_predecessor_value = self.staged_summary.sequence_value
  self.staged_summary = empty_device_step_summary()
  self.staged_plan_owner = None
  self.staged_wire_owner = None
  self.staged_wire_epoch = 0UL
  self.lifecycle = Ready
}
