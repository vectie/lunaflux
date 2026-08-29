///|
fn descriptor_stage_allocation(
  owner : DeviceStepOwner,
  stage : DeviceStepStage,
) -> @device.Allocation {
  match stage {
    Counts => owner.resources.counts
    TokenIds => owner.resources.token_ids
    TokenPositions => owner.resources.token_positions
    QueryRowOffsets => owner.resources.query_row_offsets
    SequenceLengths => owner.resources.sequence_lengths
    PageOffsets => owner.resources.page_offsets
    PhysicalPageIndices => owner.resources.physical_page_indices
  }
}

///|
/// Builds one startup-only kernel argument for an exact reusable descriptor
/// allocation. The descriptor owner keeps all allocation and close authority;
/// callers receive only the opaque argument used to prebuild a launch record.
pub fn DeviceStepOwner::descriptor_argument(
  self : DeviceStepOwner,
  stage : DeviceStepStage,
  byte_count~ : Int64,
  alignment~ : Int64,
) -> @device.KernelArgument raise DeviceStepError {
  if self.lifecycle != Ready || byte_count <= 0L || alignment <= 0L {
    raise DescriptorArgumentMismatch(stage)
  }
  let allocation = descriptor_stage_allocation(self, stage)
  if allocation.byte_count() != byte_count {
    raise DescriptorArgumentMismatch(stage)
  }
  self.context.validate_allocation_region(
    allocation,
    offset=0L,
    byte_count~,
    alignment~,
  ) catch {
    error => raise Device(stage, error)
  }
  @device.KernelArgument::device_region(
    allocation,
    offset=0L,
    byte_count~,
    alignment~,
  ) catch {
    error => raise Device(stage, error)
  }
}

///|
/// Recreates the exact currently staged value capability from a scalar epoch.
/// This lets aggregate executors retain only scalar hot-state identity rather
/// than storing another optional capability alias.
pub fn DeviceStepOwner::rederive_staged(
  self : DeviceStepOwner,
  epoch : UInt64,
) -> StagedDeviceStep raise DeviceStepError {
  let staged = StagedDeviceStep(self, epoch)
  staged.authenticate(self)
  staged
}

///|
pub fn StagedDeviceStep::epoch_value(self : StagedDeviceStep) -> UInt64 {
  self.1
}

///|
pub fn StagedDeviceStep::sequence_value(
  self : StagedDeviceStep,
) -> UInt64 raise DeviceStepError {
  self.authenticate(self.0)
  self.0.staged_summary.sequence_value
}

///|
pub fn StagedDeviceStep::row_count(
  self : StagedDeviceStep,
) -> Int raise DeviceStepError {
  self.authenticate(self.0)
  self.0.staged_summary.row_count
}

///|
pub fn StagedDeviceStep::token_count(
  self : StagedDeviceStep,
) -> Int raise DeviceStepError {
  self.authenticate(self.0)
  self.0.staged_summary.token_count
}

///|
pub fn StagedDeviceStep::last_token_index(
  self : StagedDeviceStep,
  row : Int,
) -> Int raise DeviceStepError {
  self.authenticate(self.0)
  let row_count = self.0.staged_summary.row_count
  let token_count = self.0.staged_summary.token_count
  if row < 0 || row >= row_count {
    raise InvalidPlan(Counts)
  }
  let token_end = read_descriptor_i32(self.0.host.query_row_offsets, row + 1)
  if token_end <= 0 || token_end > token_count {
    raise InvalidPlan(Counts)
  }
  token_end - 1
}

///|
/// Authenticates that `frame` is the exact owner epoch originally staged into
/// this descriptor capability.
pub fn StagedDeviceStep::validate_frame(
  self : StagedDeviceStep,
  frame : @worker_wire.ValidatedPlanFrame,
) -> Unit raise DeviceStepError {
  self.authenticate(self.0)
  guard self.0.staged_wire_owner is Some(owner) else {
    raise StageOwnerMismatch
  }
  let matches = frame.matches_owner_epoch(owner, self.0.staged_wire_epoch) catch {
    error => raise Wire(error)
  }
  let sequence = frame.sequence_value() catch { error => raise Wire(error) }
  if !matches || sequence != self.0.staged_summary.sequence_value {
    raise StageOwnerMismatch
  }
}

///|
/// Authenticates one exact staged wire frame, then removes the descriptor's
/// optional wire-owner slot. Aggregate executors must carry the returned
/// frame identity and epoch in their external non-optional value capability;
/// the warmed descriptor retains only its ordinary scalar stage state.
pub fn StagedDeviceStep::detach_frame_authentication(
  self : StagedDeviceStep,
  frame : @worker_wire.ValidatedPlanFrame,
) -> Unit raise DeviceStepError {
  self.authenticate(self.0)
  guard self.0.staged_wire_owner is Some(owner) else {
    raise StageOwnerMismatch
  }
  let wire_epoch = self.0.staged_wire_epoch
  self.0.staged_wire_owner = None
  self.0.staged_wire_epoch = 0UL
  let matches = frame.matches_owner_epoch(owner, wire_epoch) catch {
    error => raise Wire(error)
  }
  let sequence = frame.sequence_value() catch { error => raise Wire(error) }
  if !matches || sequence != self.0.staged_summary.sequence_value {
    raise StageOwnerMismatch
  }
}

///|
fn read_descriptor_i32(buffer : FixedArray[Byte], index : Int) -> Int {
  let start = index * INT32_BYTES
  buffer[start].to_int() |
  (buffer[start + 1].to_int() << 8) |
  (buffer[start + 2].to_int() << 16) |
  (buffer[start + 3].to_int() << 24)
}
