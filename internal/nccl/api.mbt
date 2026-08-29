///|
const NCCL_COMMUNICATOR_ID_BYTES : Int = 128

///|
fn availability_from_code(code : Int) -> NcclAvailability {
  match code {
    0 => Available
    1 => UnsupportedPlatform
    2 => LibraryMissing
    3 => AbiIncomplete
    _ => VersionUnavailable
  }
}

///|
fn check_status(
  status : Int,
  operation : NcclOperation,
) -> Unit raise NcclError {
  match status {
    0 => ()
    -1 => raise Unavailable(operation)
    -2 => raise InvalidArgument(operation)
    -3 => raise Closed(operation)
    -4 => raise Busy(operation)
    -5 => raise RuntimeFailure(operation)
    -6 => raise VersionMismatch(operation)
    -7 => raise GenerationMismatch(operation)
    -8 => raise SequenceMismatch(operation)
    _ => raise Failed(operation)
  }
}

///|
pub fn probe() -> NcclRuntimeInventory {
  let version = Ref(0)
  let availability = availability_from_code(raw_probe(version))
  { availability, version_code: version.val }
}

///|
/// Admits the loaded process-lifetime runtime against explicit caller bounds.
pub fn admit_runtime(
  minimum_version_code~ : Int,
  maximum_version_code~ : Int,
) -> NcclRuntimeAdmission raise NcclError {
  let version = Ref(0)
  check_status(
    raw_runtime_admit(minimum_version_code, maximum_version_code, version),
    RuntimeAdmission,
  )
  { version_code: version.val }
}

///|
/// Creates one rendezvous identity under the exact admitted runtime version.
pub fn NcclRuntimeAdmission::create_communicator_id(
  self : NcclRuntimeAdmission,
) -> NcclCommunicatorId raise NcclError {
  let bytes = FixedArray::make(NCCL_COMMUNICATOR_ID_BYTES, b'\x00')
  check_status(
    raw_communicator_id_create(self.version_code, bytes),
    CommunicatorIdCreate,
  )
  { bytes, }
}

///|
/// Imports one exactly sized communicator identity copied from trusted framing.
pub fn communicator_id_from_bytes(
  source : BytesView,
) -> NcclCommunicatorId raise NcclError {
  if source.length() != NCCL_COMMUNICATOR_ID_BYTES {
    raise InvalidArgument(CommunicatorIdCreate)
  }
  let bytes = FixedArray::make(NCCL_COMMUNICATOR_ID_BYTES, b'\x00')
  for index, unit in source {
    bytes[index] = unit
  }
  { bytes, }
}

///|
/// Copies the opaque rendezvous bytes for authenticated inter-rank transport.
pub fn NcclCommunicatorId::to_bytes(self : NcclCommunicatorId) -> Bytes {
  Bytes::from_array(self.bytes)
}

///|
/// Creates the sole communicator owner for one exact rank and generation.
pub fn NcclRuntimeAdmission::create_communicator(
  self : NcclRuntimeAdmission,
  id : NcclCommunicatorId,
  world_size~ : Int,
  rank~ : Int,
  generation~ : UInt64,
  previous_plan_sequence~ : UInt64,
) -> NcclCommunicator raise NcclError {
  let status = Ref(0)
  let communicator = raw_communicator_create(
    self.version_code,
    id.bytes,
    world_size,
    rank,
    generation,
    previous_plan_sequence,
    status,
  )
  check_status(status.val, CommunicatorCreate)
  communicator
}

///|
/// Creates one communicator while retaining the exact opaque device context
/// admitted by its higher-level owner.
pub fn NcclRuntimeAdmission::create_communicator_on_context(
  self : NcclRuntimeAdmission,
  id : NcclCommunicatorId,
  world_size~ : Int,
  rank~ : Int,
  generation~ : UInt64,
  previous_plan_sequence~ : UInt64,
  context~ : @device_collective_interop.ContextToken,
) -> NcclCommunicator raise NcclError {
  let status = Ref(0)
  let communicator = raw_communicator_create_on_context(
    self.version_code,
    id.bytes,
    world_size,
    rank,
    generation,
    previous_plan_sequence,
    context,
    status,
  )
  check_status(status.val, CommunicatorCreate)
  communicator
}

///|
fn collective_operation_code(operation : NcclCollectiveOperation) -> Int {
  match operation {
    AllReduce => 1
    AllGather => 2
    ReduceScatter => 3
  }
}

///|
/// Polls nonblocking communicator initialization. False means still pending.
pub fn NcclCommunicator::poll_ready(
  self : NcclCommunicator,
) -> Bool raise NcclError {
  let ready = Ref(0)
  check_status(raw_communicator_poll_ready(self, ready), CommunicatorCreate)
  ready.val == 1
}

///|
/// Submits one scalar-authenticated BF16 collective without committing its
/// sequence. The retained fixed in-flight record is consumed by poll or abort.
pub fn NcclCommunicator::submit_bf16(
  self : NcclCommunicator,
  generation~ : UInt64,
  plan_sequence~ : UInt64,
  collective_sequence~ : UInt64,
  operation_id~ : Int,
  collective_kind~ : NcclCollectiveOperation,
  context~ : @device_collective_interop.ContextToken,
  send~ : @device_collective_interop.RegionToken,
  send_offset~ : Int64,
  send_elements~ : Int64,
  receive~ : @device_collective_interop.RegionToken,
  receive_offset~ : Int64,
  receive_elements~ : Int64,
  stream~ : @device_collective_interop.QueueToken,
) -> Unit raise NcclError {
  check_status(
    raw_communicator_submit_bf16(
      self,
      generation,
      plan_sequence,
      collective_sequence,
      operation_id,
      collective_operation_code(collective_kind),
      context,
      send,
      send_offset,
      send_elements,
      receive,
      receive_offset,
      receive_elements,
      stream,
    ),
    CollectiveExecution,
  )
}

///|
/// Polls NCCL async state and opaque queue completion without blocking. True
/// is the sole successful sequence-commit event; false means still pending.
pub fn NcclCommunicator::poll_collective(
  self : NcclCommunicator,
) -> Bool raise NcclError {
  match raw_communicator_poll_collective_state(self) {
    0 => false
    1 => true
    status => {
      check_status(status, CollectiveExecution)
      false
    }
  }
}

///|
/// Latches rank loss, timeout, or runtime failure for this exact generation.
pub fn NcclCommunicator::invalidate(
  self : NcclCommunicator,
  generation~ : UInt64,
) -> Unit raise NcclError {
  check_status(
    raw_communicator_invalidate(self, generation),
    CommunicatorInvalidate,
  )
}

///|
/// Collectively destroys a healthy communicator. The native handle is consumed
/// exactly once even when NCCL reports a destroy failure.
pub fn NcclCommunicator::close(self : NcclCommunicator) -> Unit raise NcclError {
  check_status(raw_communicator_close(self), CommunicatorClose)
}

///|
/// Aborts a failed/draining communicator and consumes its handle exactly once.
pub fn NcclCommunicator::abort(self : NcclCommunicator) -> Unit raise NcclError {
  check_status(raw_communicator_abort(self), CommunicatorAbort)
}

///|
pub fn NcclCommunicator::generation(self : NcclCommunicator) -> UInt64 {
  raw_communicator_generation(self)
}

///|
pub fn NcclCommunicator::next_collective_sequence(
  self : NcclCommunicator,
) -> UInt64 {
  raw_communicator_next_sequence(self)
}

///|
pub fn NcclCommunicator::world_size(self : NcclCommunicator) -> Int {
  raw_communicator_world_size(self)
}

///|
pub fn NcclCommunicator::rank(self : NcclCommunicator) -> Int {
  raw_communicator_rank(self)
}
