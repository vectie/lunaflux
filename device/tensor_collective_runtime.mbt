///|
const MINIMUM_NCCL_BF16_VERSION_CODE : Int = 21403

///|
fn collective_availability_from_internal(
  availability : @nccl.NcclAvailability,
) -> TensorCollectiveRuntimeAvailability {
  match availability {
    Available => CollectiveRuntimeAvailable
    UnsupportedPlatform => CollectiveRuntimeUnsupportedPlatform
    LibraryMissing => CollectiveRuntimeLibraryMissing
    AbiIncomplete => CollectiveRuntimeAbiIncomplete
    VersionUnavailable => CollectiveRuntimeVersionUnavailable
  }
}

///|
/// Probes only the process-lifetime dynamically loaded collective dispatch.
/// It creates no communicator, context, stream, rank, or device allocation.
pub fn probe_tensor_collective_runtime() -> TensorCollectiveRuntimeInventory {
  let inventory = @nccl.probe()
  {
    availability: collective_availability_from_internal(
      inventory.availability(),
    ),
    version_code: inventory.version_code(),
  }
}

///|
pub fn TensorCollectiveRuntimeInventory::availability(
  self : TensorCollectiveRuntimeInventory,
) -> TensorCollectiveRuntimeAvailability {
  self.availability
}

///|
pub fn TensorCollectiveRuntimeInventory::version_code(
  self : TensorCollectiveRuntimeInventory,
) -> Int {
  self.version_code
}

///|
fn error_from_nccl_for(
  error : @nccl.NcclError,
  operation : DeviceOperation,
) -> DeviceError {
  match error {
    Unavailable(_) => Unavailable(operation)
    VersionMismatch(_) => Unsupported(operation)
    InvalidArgument(_) => InvalidArgument(operation)
    Closed(_) => Closed(operation)
    Busy(_) => Busy(operation)
    RuntimeFailure(_) => DriverFailure(operation)
    GenerationMismatch(_) => InvalidArgument(operation)
    SequenceMismatch(_) => InvalidArgument(operation)
    Failed(_) => DriverFailure(operation)
  }
}

///|
/// Admits the immutable process-lifetime NCCL dispatch against explicit bounds.
pub fn admit_tensor_collective_runtime(
  minimum_version_code~ : Int,
  maximum_version_code~ : Int,
) -> TensorCollectiveRuntime raise DeviceError {
  // NCCL 2.14.3 is the first admitted nonblocking communicator ABI.
  if minimum_version_code < MINIMUM_NCCL_BF16_VERSION_CODE {
    raise InvalidArgument(TensorCollectiveRuntimeAdmission)
  }
  let raw = @nccl.admit_runtime(minimum_version_code~, maximum_version_code~) catch {
    error => raise error_from_nccl_for(error, TensorCollectiveRuntimeAdmission)
  }
  { raw, }
}

///|
pub fn TensorCollectiveRuntime::collective_runtime_version_code(
  self : TensorCollectiveRuntime,
) -> Int {
  self.raw.version_code()
}

///|
pub fn TensorCollectiveRuntime::create_group_id(
  self : TensorCollectiveRuntime,
) -> TensorCollectiveGroupId raise DeviceError {
  let raw = self.raw.create_communicator_id() catch {
    error => raise error_from_nccl_for(error, TensorCollectiveGroupIdCreate)
  }
  { raw, }
}

///|
pub fn tensor_collective_group_id_from_bytes(
  source : BytesView,
) -> TensorCollectiveGroupId raise DeviceError {
  let raw = @nccl.communicator_id_from_bytes(source) catch {
    error => raise error_from_nccl_for(error, TensorCollectiveGroupIdCreate)
  }
  { raw, }
}

///|
pub fn TensorCollectiveGroupId::to_bytes(
  self : TensorCollectiveGroupId,
) -> Bytes {
  self.raw.to_bytes()
}

///|
/// Creates the thread-confined communicator owner for this exact context,
/// rank, immutable contract, and nonzero group generation.
pub fn Context::create_tensor_collective_owner(
  self : Context,
  runtime~ : TensorCollectiveRuntime,
  group_id~ : TensorCollectiveGroupId,
  contract~ : TensorCollectiveContract,
  generation~ : UInt64,
  previous_plan_sequence_value~ : UInt64,
) -> TensorCollectiveOwner raise DeviceError {
  @cuda.link_collective_interop()
  if generation == 0UL ||
    previous_plan_sequence_value == 18446744073709551615UL ||
    !self.capability().supports_bf16() {
    raise InvalidArgument(TensorCollectiveOwnerCreate)
  }
  let communicator = runtime.raw.create_communicator_on_context(
    group_id.raw,
    world_size=contract.world_size,
    rank=contract.rank,
    generation~,
    previous_plan_sequence=previous_plan_sequence_value,
    context=self.raw.collective_token(),
  ) catch {
    error => raise error_from_nccl_for(error, TensorCollectiveOwnerCreate)
  }
  {
    context: self,
    contract,
    communicator,
    generation,
    state: CollectiveOwnerStarting,
    plan_sequence: previous_plan_sequence_value,
    next_global_sequence: 1UL,
    next_site_index: contract.site_count(),
    pending_plan_sequence: 0UL,
    pending_global_sequence: 0UL,
    pending_site_index: 0,
  }
}

///|
/// Polls nonblocking communicator rendezvous. False means still pending.
pub fn TensorCollectiveOwner::poll_ready(
  self : TensorCollectiveOwner,
) -> Bool raise DeviceError {
  @cuda.link_collective_interop()
  match self.state {
    CollectiveOwnerStarting => ()
    CollectiveOwnerLive => return true
    CollectiveOwnerInFlight => raise Busy(TensorCollectiveOwnerCreate)
    CollectiveOwnerFailed => raise DriverFailure(TensorCollectiveOwnerCreate)
    CollectiveOwnerClosed => raise Closed(TensorCollectiveOwnerCreate)
  }
  let ready = self.communicator.poll_ready() catch {
    error => {
      self.state = CollectiveOwnerFailed
      raise error_from_nccl_for(error, TensorCollectiveOwnerCreate)
    }
  }
  if ready {
    self.state = CollectiveOwnerLive
  }
  ready
}
