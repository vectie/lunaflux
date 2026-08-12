///|
fn availability_from_code(code : Int) -> DriverAvailability {
  match code {
    0 => Available
    1 => UnsupportedPlatform
    2 => DriverLibraryMissing
    3 => DriverAbiIncomplete
    4 => DriverInitializationFailed
    5 => NoDevices
    _ => InventoryFailed
  }
}

///|
fn check_status(
  status : Int,
  operation : CudaOperation,
) -> Unit raise CudaError {
  match status {
    0 => ()
    -1 => raise Unavailable(operation)
    -2 => raise InvalidArgument(operation)
    -3 => raise Closed(operation)
    -4 => raise Busy(operation)
    -5 => raise DriverFailure(operation)
    -6 => raise HostAllocationFailed(operation)
    -7 => raise SizeOverflow(operation)
    _ => raise Unsupported(operation)
  }
}

///|
pub fn probe() -> DriverInventory {
  let output = FixedArray::make(3, 0L)
  let availability = availability_from_code(raw_probe(output))
  {
    availability,
    driver_version: output[0].to_int(),
    device_count: output[1].to_int(),
    cublas_lt_available: output[2] != 0L,
  }
}

///|
pub fn device_info(ordinal : Int) -> RawDeviceInfo raise CudaError {
  let numeric = FixedArray::make(5, 0L)
  let name = FixedArray::make(96, b'\x00')
  check_status(raw_device_info(ordinal, numeric, name), DeviceInventory)
  let mut name_length = 0
  while name_length < name.length() && name[name_length] != b'\x00' {
    name_length += 1
  }
  {
    total_memory_bytes: numeric[0],
    compute_major: numeric[1].to_int(),
    compute_minor: numeric[2].to_int(),
    supports_bf16: numeric[3] != 0L,
    cublas_lt_available: numeric[4] != 0L,
    name: @utf8.decode(Bytes::from_array(name.view(end=name_length))) catch {
      _ => raise InvalidOutput(DeviceInventory)
    },
  }
}

///|
pub fn open_context(ordinal : Int) -> Context raise CudaError {
  let status = Ref(0)
  let context = raw_context_create(ordinal, status)
  check_status(status.val, ContextCreate)
  context
}

///|
pub fn Context::close(self : Context) -> Unit raise CudaError {
  check_status(raw_context_close(self), ContextClose)
}

///|
pub fn Context::create_stream(self : Context) -> Stream raise CudaError {
  let status = Ref(0)
  let stream = raw_stream_create(self, status)
  check_status(status.val, StreamCreate)
  stream
}

///|
pub fn Stream::close(self : Stream) -> Unit raise CudaError {
  check_status(raw_stream_close(self), StreamClose)
}

///|
pub fn Stream::synchronize(self : Stream) -> Unit raise CudaError {
  check_status(raw_stream_synchronize(self), StreamSynchronize)
}

///|
pub fn Context::create_event(self : Context) -> Event raise CudaError {
  let status = Ref(0)
  let event = raw_event_create(self, status)
  check_status(status.val, EventCreate)
  event
}

///|
pub fn Event::close(self : Event) -> Unit raise CudaError {
  check_status(raw_event_close(self), EventClose)
}

///|
pub fn Event::record(self : Event, stream : Stream) -> Unit raise CudaError {
  check_status(raw_event_record(self, stream), EventRecord)
}

///|
pub fn Event::synchronize(self : Event) -> Unit raise CudaError {
  check_status(raw_event_synchronize(self), EventSynchronize)
}

///|
pub fn Event::elapsed_milliseconds(
  self : Event,
  finish : Event,
) -> Double raise CudaError {
  let status = Ref(0)
  let elapsed = raw_event_elapsed(self, finish, status)
  check_status(status.val, EventElapsed)
  elapsed
}

///|
pub fn Context::allocate(
  self : Context,
  byte_count : Int64,
) -> Allocation raise CudaError {
  let status = Ref(0)
  let allocation = raw_allocation_create(self, byte_count, status)
  check_status(status.val, AllocationCreate)
  allocation
}

///|
pub fn Allocation::close(self : Allocation) -> Unit raise CudaError {
  check_status(raw_allocation_close(self), AllocationClose)
}

///|
pub fn Allocation::copy_from_host(
  self : Allocation,
  source : Bytes,
  byte_count : Int64,
) -> Unit raise CudaError {
  self.copy_from_host_at(source, 0L, 0L, byte_count)
}

///|
/// Copies a bounded host byte range into a bounded allocation range. The
/// borrowed host buffer is retained only for this synchronous call.
pub fn Allocation::copy_from_host_at(
  self : Allocation,
  source : Bytes,
  source_offset : Int64,
  destination_offset : Int64,
  byte_count : Int64,
) -> Unit raise CudaError {
  check_status(
    raw_copy_to_device(
      self, source, source_offset, destination_offset, byte_count,
    ),
    CopyToDevice,
  )
}

///|
pub fn Allocation::copy_to_host(
  self : Allocation,
  byte_count : Int64,
) -> Bytes raise CudaError {
  self.copy_to_host_at(0L, byte_count)
}

///|
/// Copies a bounded allocation range into a fresh host buffer.
pub fn Allocation::copy_to_host_at(
  self : Allocation,
  source_offset : Int64,
  byte_count : Int64,
) -> Bytes raise CudaError {
  let status = Ref(0)
  let output = raw_copy_to_host(self, source_offset, byte_count, status)
  check_status(status.val, CopyToHost)
  output
}

///|
pub fn Context::load_module(
  self : Context,
  image : Bytes,
) -> Module raise CudaError {
  let status = Ref(0)
  let module_ = raw_module_load(self, image, status)
  check_status(status.val, ModuleLoad)
  module_
}

///|
pub fn Module::close(self : Module) -> Unit raise CudaError {
  check_status(raw_module_close(self), ModuleClose)
}

///|
pub fn Module::load_function(
  self : Module,
  name : Bytes,
) -> Function raise CudaError {
  let status = Ref(0)
  let function_ = raw_function_load(self, name, status)
  check_status(status.val, FunctionLoad)
  function_
}

///|
pub fn Function::close(self : Function) -> Unit raise CudaError {
  check_status(raw_function_close(self), FunctionClose)
}

///|
pub fn Context::create_cublas_lt(self : Context) -> CublasLt raise CudaError {
  let status = Ref(0)
  let handle = raw_cublas_create(self, status)
  check_status(status.val, CublasLtCreate)
  handle
}

///|
pub fn CublasLt::close(self : CublasLt) -> Unit raise CudaError {
  check_status(raw_cublas_close(self), CublasLtClose)
}

///|
/// Creates a reusable row-major BF16 GEMM descriptor for
/// `output[rows, columns] = left[rows, inner] * right[inner, columns]`.
/// Accumulation and scalar computation use FP32. Workspace is caller-owned.
pub fn CublasLt::create_bf16_gemm_plan(
  self : CublasLt,
  rows : Int64,
  inner : Int64,
  columns : Int64,
  workspace_byte_count : Int64,
) -> Bf16GemmPlan raise CudaError {
  let status = Ref(0)
  let plan = raw_bf16_gemm_plan_create(
    self, rows, inner, columns, workspace_byte_count, status,
  )
  check_status(status.val, Bf16GemmPlanCreate)
  plan
}

///|
pub fn Bf16GemmPlan::close(self : Bf16GemmPlan) -> Unit raise CudaError {
  check_status(raw_bf16_gemm_plan_close(self), Bf16GemmPlanClose)
}

///|
/// Runs this GEMM on `stream` and synchronizes it before returning. Operand
/// offsets must be 16-byte aligned; workspace offsets must be 256-byte aligned.
pub fn Bf16GemmPlan::run(
  self : Bf16GemmPlan,
  left : Allocation,
  left_offset : Int64,
  right : Allocation,
  right_offset : Int64,
  output : Allocation,
  output_offset : Int64,
  workspace : Allocation,
  workspace_offset : Int64,
  stream : Stream,
) -> Unit raise CudaError {
  check_status(
    raw_bf16_gemm_plan_run(
      self, left, left_offset, right, right_offset, output, output_offset, workspace,
      workspace_offset, stream,
    ),
    Bf16GemmRun,
  )
}
