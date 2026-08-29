///|
type Context

///|
type Stream

///|
type Event

///|
type Allocation

///|
type AllocationLease

///|
type Module

///|
type Function

///|
type OrderedKernelExecutor

///|
type CublasLt

///|
type Bf16GemmPlan

///|
extern "c" fn raw_collective_interop_link_anchor() -> Unit = "lunaflux_cuda_collective_interop_link_anchor"

///|
#borrow(context)
extern "c" fn raw_collective_context_token(
  context : Context,
) -> @device_collective_interop.ContextToken = "lunaflux_device_interop_context_token"

///|
#borrow(region)
extern "c" fn raw_collective_region_token(
  region : Allocation,
) -> @device_collective_interop.RegionToken = "lunaflux_device_interop_region_token"

///|
#borrow(queue)
extern "c" fn raw_collective_queue_token(
  queue : Stream,
) -> @device_collective_interop.QueueToken = "lunaflux_device_interop_queue_token"

///|
#borrow(output)
extern "c" fn raw_probe(output : FixedArray[Int64]) -> Int = "lunaflux_cuda_probe"

///|
#borrow(numeric, name, uuid, pci)
extern "c" fn raw_device_info(
  ordinal : Int,
  numeric : FixedArray[Int64],
  name : FixedArray[Byte],
  uuid : FixedArray[Byte],
  pci : FixedArray[Byte],
) -> Int = "lunaflux_cuda_device_info"

///|
#borrow(output)
extern "c" fn raw_device_can_access_peer(
  source_ordinal : Int,
  destination_ordinal : Int,
  output : Ref[Int],
) -> Int = "lunaflux_cuda_device_can_access_peer"

///|
#borrow(status)
extern "c" fn raw_context_create(ordinal : Int, status : Ref[Int]) -> Context = "lunaflux_cuda_context_create"

///|
#borrow(context)
extern "c" fn raw_context_close(context : Context) -> Int = "lunaflux_cuda_context_close"

///|
#borrow(context, status)
extern "c" fn raw_stream_create(context : Context, status : Ref[Int]) -> Stream = "lunaflux_cuda_stream_create"

///|
#borrow(stream)
extern "c" fn raw_stream_close(stream : Stream) -> Int = "lunaflux_cuda_stream_close"

///|
#borrow(stream)
extern "c" fn raw_stream_synchronize(stream : Stream) -> Int = "lunaflux_cuda_stream_synchronize"

///|
#borrow(context, status)
extern "c" fn raw_event_create(context : Context, status : Ref[Int]) -> Event = "lunaflux_cuda_event_create"

///|
#borrow(event)
extern "c" fn raw_event_close(event : Event) -> Int = "lunaflux_cuda_event_close"

///|
#borrow(event, stream)
extern "c" fn raw_event_record(event : Event, stream : Stream) -> Int = "lunaflux_cuda_event_record"

///|
#borrow(event)
extern "c" fn raw_event_synchronize(event : Event) -> Int = "lunaflux_cuda_event_synchronize"

///|
#borrow(start, finish, status)
extern "c" fn raw_event_elapsed(
  start : Event,
  finish : Event,
  status : Ref[Int],
) -> Double = "lunaflux_cuda_event_elapsed"

///|
#borrow(context, status)
extern "c" fn raw_allocation_create(
  context : Context,
  byte_count : Int64,
  status : Ref[Int],
) -> Allocation = "lunaflux_cuda_allocation_create"

///|
#borrow(allocation)
extern "c" fn raw_allocation_close(allocation : Allocation) -> Int = "lunaflux_cuda_allocation_close"

///|
#borrow(context, allocation, status)
extern "c" fn raw_allocation_lease_create(
  context : Context,
  allocation : Allocation,
  status : Ref[Int],
) -> AllocationLease = "lunaflux_cuda_allocation_lease_create"

///|
#borrow(lease)
extern "c" fn raw_allocation_lease_close(lease : AllocationLease) -> Int = "lunaflux_cuda_allocation_lease_close"

///|
#borrow(context, allocation)
extern "c" fn raw_context_validate_allocation_region(
  context : Context,
  allocation : Allocation,
  offset : Int64,
  byte_count : Int64,
  alignment : Int64,
) -> Int = "lunaflux_cuda_context_validate_allocation_region"

///|
#borrow(allocation, source)
extern "c" fn raw_copy_to_device(
  allocation : Allocation,
  source : Bytes,
  source_offset : Int64,
  destination_offset : Int64,
  byte_count : Int64,
) -> Int = "lunaflux_cuda_copy_to_device"

///|
#borrow(context, allocation, source)
extern "c" fn raw_context_copy_fixed_to_device(
  context : Context,
  allocation : Allocation,
  source : FixedArray[Byte],
  source_offset : Int64,
  destination_offset : Int64,
  byte_count : Int64,
) -> Int = "lunaflux_cuda_context_copy_fixed_to_device"

///|
#borrow(context, allocation, destination)
extern "c" fn raw_context_copy_device_to_fixed(
  context : Context,
  allocation : Allocation,
  destination : FixedArray[Byte],
  source_offset : Int64,
  destination_offset : Int64,
  byte_count : Int64,
) -> Int = "lunaflux_cuda_context_copy_device_to_fixed"

///|
#borrow(allocation, status)
extern "c" fn raw_copy_to_host(
  allocation : Allocation,
  source_offset : Int64,
  byte_count : Int64,
  status : Ref[Int],
) -> Bytes = "lunaflux_cuda_copy_to_host"

///|
#borrow(context, image, status)
extern "c" fn raw_module_load(
  context : Context,
  image : Bytes,
  status : Ref[Int],
) -> Module = "lunaflux_cuda_module_load"

///|
#borrow(module_)
extern "c" fn raw_module_close(module_ : Module) -> Int = "lunaflux_cuda_module_close"

///|
#borrow(module_, name, status)
extern "c" fn raw_function_load(
  module_ : Module,
  name : Bytes,
  status : Ref[Int],
) -> Function = "lunaflux_cuda_function_load"

///|
#borrow(function_)
extern "c" fn raw_function_close(function_ : Function) -> Int = "lunaflux_cuda_function_close"

///|
#borrow(function_, stream, allocations, offsets, byte_counts, alignments)
extern "c" fn raw_function_launch(
  function_ : Function,
  stream : Stream,
  grid_x : Int,
  grid_y : Int,
  grid_z : Int,
  block_x : Int,
  block_y : Int,
  block_z : Int,
  shared_memory_bytes : Int,
  allocations : FixedArray[Allocation],
  offsets : FixedArray[Int64],
  byte_counts : FixedArray[Int64],
  alignments : FixedArray[Int64],
) -> Int = "lunaflux_cuda_function_launch"

///|
#borrow(context, stream, functions, dimensions, argument_starts, allocations, offsets, byte_counts, alignments, status)
extern "c" fn raw_ordered_executor_create(
  context : Context,
  stream : Stream,
  functions : FixedArray[Function],
  dimensions : FixedArray[Int],
  argument_starts : FixedArray[Int],
  allocations : FixedArray[Allocation],
  offsets : FixedArray[Int64],
  byte_counts : FixedArray[Int64],
  alignments : FixedArray[Int64],
  policy : Int,
  status : Ref[Int],
) -> OrderedKernelExecutor = "lunaflux_cuda_ordered_executor_create"

///|
#borrow(executor)
extern "c" fn raw_ordered_executor_enqueue(
  executor : OrderedKernelExecutor,
  index : Int,
) -> Int = "lunaflux_cuda_ordered_executor_enqueue"

///|
#borrow(executor)
extern "c" fn raw_ordered_executor_launch_captured(
  executor : OrderedKernelExecutor,
) -> Int = "lunaflux_cuda_ordered_executor_launch_captured"

///|
#borrow(executor)
extern "c" fn raw_ordered_executor_mode(
  executor : OrderedKernelExecutor,
) -> Int = "lunaflux_cuda_ordered_executor_mode"

///|
#borrow(executor)
extern "c" fn raw_ordered_executor_record(
  executor : OrderedKernelExecutor,
) -> Int = "lunaflux_cuda_ordered_executor_record"

///|
#borrow(executor)
extern "c" fn raw_ordered_executor_poll(
  executor : OrderedKernelExecutor,
) -> Int = "lunaflux_cuda_ordered_executor_poll"

///|
#borrow(executor)
extern "c" fn raw_ordered_executor_wait(
  executor : OrderedKernelExecutor,
) -> Int = "lunaflux_cuda_ordered_executor_wait"

///|
#borrow(executor)
extern "c" fn raw_ordered_executor_abort(
  executor : OrderedKernelExecutor,
) -> Int = "lunaflux_cuda_ordered_executor_abort"

///|
#borrow(executor)
extern "c" fn raw_ordered_executor_reset(
  executor : OrderedKernelExecutor,
) -> Int = "lunaflux_cuda_ordered_executor_reset"

///|
#borrow(executor)
extern "c" fn raw_ordered_executor_close(
  executor : OrderedKernelExecutor,
) -> Int = "lunaflux_cuda_ordered_executor_close"

///|
#borrow(context, status)
extern "c" fn raw_cublas_create(
  context : Context,
  status : Ref[Int],
) -> CublasLt = "lunaflux_cuda_cublas_create"

///|
#borrow(handle)
extern "c" fn raw_cublas_close(handle : CublasLt) -> Int = "lunaflux_cuda_cublas_close"

///|
#borrow(handle, status)
extern "c" fn raw_bf16_gemm_plan_create(
  handle : CublasLt,
  rows : Int64,
  inner : Int64,
  columns : Int64,
  workspace_byte_count : Int64,
  status : Ref[Int],
) -> Bf16GemmPlan = "lunaflux_cuda_bf16_gemm_plan_create"

///|
#borrow(plan)
extern "c" fn raw_bf16_gemm_plan_close(plan : Bf16GemmPlan) -> Int = "lunaflux_cuda_bf16_gemm_plan_close"

///|
#borrow(plan, left, right, output, workspace, stream)
extern "c" fn raw_bf16_gemm_plan_run(
  plan : Bf16GemmPlan,
  left : Allocation,
  left_offset : Int64,
  right : Allocation,
  right_offset : Int64,
  output : Allocation,
  output_offset : Int64,
  workspace : Allocation,
  workspace_offset : Int64,
  stream : Stream,
) -> Int = "lunaflux_cuda_bf16_gemm_plan_run"
