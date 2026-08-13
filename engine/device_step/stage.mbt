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
) -> DeviceStepSummary raise DeviceStepError {
  let summary = preflight_and_write(
    plan, expected_generation, expected_identity, layout, limits, host, required_capabilities,
    false,
  )
  validate_next_sequence(
    has_accepted_predecessor,
    accepted_predecessor_value,
    summary.sequence_value,
  )
  if stage_epoch == 18446744073709551615UL {
    raise StageEpochOverflow
  }
  preflight_and_write(
    plan, expected_generation, expected_identity, layout, limits, host, required_capabilities,
    true,
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
fn invalidate_counts(host : DeviceStepHostBuffers) -> Unit {
  for index in 0..<COUNTS_LENGTH {
    write_i32(host.counts, index, 0)
  }
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
  invalidate_counts(owner.host)
  copy_operand(
    owner,
    Counts,
    owner.resources.counts,
    owner.host.counts,
    COUNTS_LENGTH,
  )
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
/// Validates and uploads one submitted plan into reusable worker-owned storage.
/// Validation is complete before host writes. Device counts are invalidated
/// first, payload operands upload next, and real counts publish last. Any copy
/// failure poisons the owner so stale prior counts can never be dispatched.
pub fn DeviceStepOwner::stage(
  self : DeviceStepOwner,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> StagedDeviceStep raise DeviceStepError {
  require_lifecycle(self.lifecycle, Ready)
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
  )
  upload_owner_transaction(self, summary) catch {
    error => {
      self.lifecycle = Poisoned
      raise error
    }
  }
  self.stage_epoch = self.stage_epoch + 1UL
  self.staged_sequence_value = summary.sequence_value
  self.lifecycle = Staged
  StagedDeviceStep(self, self.stage_epoch)
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
  self.accepted_predecessor_value = self.staged_sequence_value
  self.lifecycle = Ready
}
