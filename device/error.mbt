///|
fn operation_from_cuda(operation : @cuda.CudaOperation) -> DeviceOperation {
  match operation {
    DeviceInventory => DeviceInventory
    ContextCreate => ContextCreate
    ContextClose => ContextClose
    StreamCreate => StreamCreate
    StreamClose => StreamClose
    StreamSynchronize => StreamSynchronize
    EventCreate => EventCreate
    EventClose => EventClose
    EventRecord => EventRecord
    EventSynchronize => EventSynchronize
    EventElapsed => EventElapsed
    AllocationCreate => AllocationCreate
    AllocationClose => AllocationClose
    AllocationLeaseCreate => AllocationLeaseCreate
    AllocationLeaseClose => AllocationLeaseClose
    AllocationRegionValidate => AllocationRegionValidate
    CopyToDevice => CopyToDevice
    CopyToHost => CopyToHost
    ModuleLoad => ModuleLoad
    ModuleClose => ModuleClose
    FunctionLoad => FunctionLoad
    FunctionClose => FunctionClose
    AotKernelLaunch => AotKernelLaunch
    CublasLtCreate => CublasLtCreate
    CublasLtClose => CublasLtClose
    Bf16GemmPlanCreate => Bf16GemmPlanCreate
    Bf16GemmPlanClose => Bf16GemmPlanClose
    Bf16GemmRun => Bf16GemmRun
  }
}

///|
fn error_from_cuda(error : @cuda.CudaError) -> DeviceError {
  match error {
    Unavailable(operation) => Unavailable(operation_from_cuda(operation))
    InvalidArgument(operation) =>
      InvalidArgument(operation_from_cuda(operation))
    Closed(operation) => Closed(operation_from_cuda(operation))
    Busy(operation) => Busy(operation_from_cuda(operation))
    DriverFailure(operation) => DriverFailure(operation_from_cuda(operation))
    HostAllocationFailed(operation) =>
      HostAllocationFailed(operation_from_cuda(operation))
    SizeOverflow(operation) => SizeOverflow(operation_from_cuda(operation))
    Unsupported(operation) => Unsupported(operation_from_cuda(operation))
    InvalidOutput(operation) => InvalidOutput(operation_from_cuda(operation))
  }
}

///|
fn error_from_cuda_for(
  error : @cuda.CudaError,
  operation : DeviceOperation,
) -> DeviceError {
  match error {
    Unavailable(_) => Unavailable(operation)
    InvalidArgument(_) => InvalidArgument(operation)
    Closed(_) => Closed(operation)
    Busy(_) => Busy(operation)
    DriverFailure(_) => DriverFailure(operation)
    HostAllocationFailed(_) => HostAllocationFailed(operation)
    SizeOverflow(_) => SizeOverflow(operation)
    Unsupported(_) => Unsupported(operation)
    InvalidOutput(_) => InvalidOutput(operation)
  }
}
