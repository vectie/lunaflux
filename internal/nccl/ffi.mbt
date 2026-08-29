///|
#borrow(version)
extern "c" fn raw_probe(version : Ref[Int]) -> Int = "lunaflux_nccl_probe"

///|
#borrow(version)
extern "c" fn raw_runtime_admit(
  minimum_version : Int,
  maximum_version : Int,
  version : Ref[Int],
) -> Int = "lunaflux_nccl_runtime_admit"

///|
#borrow(output)
extern "c" fn raw_communicator_id_create(
  admitted_version : Int,
  output : FixedArray[Byte],
) -> Int = "lunaflux_nccl_unique_id_create"

///|
#borrow(unique_id, status)
extern "c" fn raw_communicator_create(
  admitted_version : Int,
  unique_id : FixedArray[Byte],
  world_size : Int,
  rank : Int,
  generation : UInt64,
  previous_plan_sequence : UInt64,
  status : Ref[Int],
) -> NcclCommunicator = "lunaflux_nccl_communicator_create"

///|
#borrow(unique_id, context, status)
extern "c" fn raw_communicator_create_on_context(
  admitted_version : Int,
  unique_id : FixedArray[Byte],
  world_size : Int,
  rank : Int,
  generation : UInt64,
  previous_plan_sequence : UInt64,
  context : @device_collective_interop.ContextToken,
  status : Ref[Int],
) -> NcclCommunicator = "lunaflux_nccl_communicator_create_on_context"

///|
#borrow(communicator)
extern "c" fn raw_communicator_close(communicator : NcclCommunicator) -> Int = "lunaflux_nccl_communicator_close"

///|
#borrow(communicator)
extern "c" fn raw_communicator_abort(communicator : NcclCommunicator) -> Int = "lunaflux_nccl_communicator_abort"

///|
#borrow(communicator, ready)
extern "c" fn raw_communicator_poll_ready(
  communicator : NcclCommunicator,
  ready : Ref[Int],
) -> Int = "lunaflux_nccl_communicator_poll_ready"

///|
#borrow(communicator, context, send, receive, stream)
extern "c" fn raw_communicator_submit_bf16(
  communicator : NcclCommunicator,
  generation : UInt64,
  plan_sequence : UInt64,
  collective_sequence : UInt64,
  operation_id : Int,
  collective_kind : Int,
  context : @device_collective_interop.ContextToken,
  send : @device_collective_interop.RegionToken,
  send_offset : Int64,
  send_elements : Int64,
  receive : @device_collective_interop.RegionToken,
  receive_offset : Int64,
  receive_elements : Int64,
  stream : @device_collective_interop.QueueToken,
) -> Int = "lunaflux_nccl_communicator_submit_bf16"

///|
#borrow(communicator)
extern "c" fn raw_communicator_poll_collective_state(
  communicator : NcclCommunicator,
) -> Int = "lunaflux_nccl_communicator_poll_collective_state"

///|
#borrow(communicator)
extern "c" fn raw_communicator_invalidate(
  communicator : NcclCommunicator,
  generation : UInt64,
) -> Int = "lunaflux_nccl_communicator_invalidate"

///|
#borrow(communicator)
extern "c" fn raw_communicator_generation(
  communicator : NcclCommunicator,
) -> UInt64 = "lunaflux_nccl_communicator_generation"

///|
#borrow(communicator)
extern "c" fn raw_communicator_next_sequence(
  communicator : NcclCommunicator,
) -> UInt64 = "lunaflux_nccl_communicator_next_sequence"

///|
#borrow(communicator)
extern "c" fn raw_communicator_world_size(
  communicator : NcclCommunicator,
) -> Int = "lunaflux_nccl_communicator_world_size"

///|
#borrow(communicator)
extern "c" fn raw_communicator_rank(communicator : NcclCommunicator) -> Int = "lunaflux_nccl_communicator_rank"
