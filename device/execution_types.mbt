///|
/// Stable device operations used in payload-safe execution failures. These
/// values deliberately contain no CUDA result codes or vendor strings.
pub(all) enum DeviceOperation {
  DeviceInventory
  ContextCreate
  ContextClose
  StreamCreate
  StreamClose
  StreamSynchronize
  EventCreate
  EventClose
  EventRecord
  EventSynchronize
  EventElapsed
  AllocationCreate
  AllocationClose
  CopyToDevice
  CopyToHost
  ModuleLoad
  ModuleClose
  FunctionLoad
  FunctionClose
  AotKernelLaunch
  CublasLtCreate
  CublasLtClose
  Bf16GemmPlanCreate
  Bf16GemmPlanClose
  Bf16GemmRun
} derive(Debug, Eq)

///|
/// Public, bounded execution failure. Native pointers, library paths, numeric
/// CUDA results, vendor diagnostics, and user payloads never enter this type.
pub(all) suberror DeviceError {
  Unavailable(DeviceOperation)
  InvalidArgument(DeviceOperation)
  Closed(DeviceOperation)
  Busy(DeviceOperation)
  DriverFailure(DeviceOperation)
  HostAllocationFailed(DeviceOperation)
  SizeOverflow(DeviceOperation)
  Unsupported(DeviceOperation)
  InvalidOutput(DeviceOperation)
} derive(Debug, Eq)

///|
/// Owns one CUDA context. Child resources retain its native lifetime, while
/// explicit close remains the primary release mechanism.
pub struct Context {
  priv raw : @cuda.Context
  priv ordinal : Int
  priv capability : DeviceCapability
}

///|
pub fn Context::ordinal(self : Context) -> Int {
  self.ordinal
}

///|
pub fn Context::capability(self : Context) -> DeviceCapability {
  self.capability
}

///|
pub struct Stream {
  priv raw : @cuda.Stream
}

///|
pub struct Event {
  priv raw : @cuda.Event
}

///|
pub struct Allocation {
  priv raw : @cuda.Allocation
  priv byte_count : Int64
}

///|
pub fn Allocation::byte_count(self : Allocation) -> Int64 {
  self.byte_count
}

///|
pub struct Module {
  priv raw : @cuda.Module
}

///|
pub struct Function {
  priv raw : @cuda.Function
}

///|
pub struct CublasLt {
  priv raw : @cuda.CublasLt
}

///|
/// Reusable descriptors for one row-major BF16 GEMM shape. The plan does not
/// own operand or workspace storage.
pub struct Bf16GemmPlan {
  priv raw : @cuda.Bf16GemmPlan
  priv rows : Int64
  priv inner : Int64
  priv columns : Int64
  priv workspace_byte_count : Int64
}

///|
pub fn Bf16GemmPlan::rows(self : Bf16GemmPlan) -> Int64 {
  self.rows
}

///|
pub fn Bf16GemmPlan::inner(self : Bf16GemmPlan) -> Int64 {
  self.inner
}

///|
pub fn Bf16GemmPlan::columns(self : Bf16GemmPlan) -> Int64 {
  self.columns
}

///|
pub fn Bf16GemmPlan::workspace_byte_count(self : Bf16GemmPlan) -> Int64 {
  self.workspace_byte_count
}
