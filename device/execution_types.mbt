///|
/// Stable device operations used in payload-safe execution failures. These
/// values deliberately contain no CUDA result codes or vendor strings.
pub(all) enum DeviceOperation {
  DeviceInventory
  DevicePeerAccess
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
  AllocationLeaseCreate
  AllocationLeaseClose
  AllocationRegionValidate
  CopyToDevice
  CopyToHost
  ModuleLoad
  ModuleClose
  FunctionLoad
  FunctionClose
  AotKernelLaunch
  OrderedExecutorCreate
  OrderedExecutorEnqueue
  OrderedExecutorGraphLaunch
  OrderedExecutorRecord
  OrderedExecutorPoll
  OrderedExecutorWait
  OrderedExecutorAbort
  OrderedExecutorReset
  OrderedExecutorClose
  CublasLtCreate
  CublasLtClose
  Bf16GemmPlanCreate
  Bf16GemmPlanClose
  Bf16GemmRun
  TensorCollectiveContractCreate
  TensorCollectiveRuntimeAdmission
  TensorCollectiveGroupIdCreate
  TensorCollectiveOwnerCreate
  TensorCollectiveExecute
  TensorCollectiveInvalidate
  TensorCollectiveClose
  TensorCollectiveAbort
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
/// Immutable, root-free identity captured from the exact CUDA ordinal opened
/// by a live context. This value owns no native resource and grants no launch,
/// memory, or context authority.
pub struct ContextDeviceIdentity {
  priv ordinal : Int
  priv name : String
  priv uuid : String
  priv pci : String
  priv capability : DeviceCapability
} derive(Debug, Eq)

///|
pub fn ContextDeviceIdentity::ordinal(self : ContextDeviceIdentity) -> Int {
  self.ordinal
}

///|
pub fn ContextDeviceIdentity::name(self : ContextDeviceIdentity) -> String {
  self.name
}

///|
pub fn ContextDeviceIdentity::uuid(self : ContextDeviceIdentity) -> String {
  self.uuid
}

///|
pub fn ContextDeviceIdentity::pci(self : ContextDeviceIdentity) -> String {
  self.pci
}

///|
pub fn ContextDeviceIdentity::capability(
  self : ContextDeviceIdentity,
) -> DeviceCapability {
  self.capability
}

///|
pub fn ContextDeviceIdentity::total_memory_bytes(
  self : ContextDeviceIdentity,
) -> Int64 {
  self.capability.total_memory_bytes()
}

///|
/// Owns one CUDA context. Child resources retain its native lifetime, while
/// explicit close remains the primary release mechanism.
pub struct Context {
  priv raw : @cuda.Context
  priv identity : ContextDeviceIdentity
}

///|
pub fn Context::ordinal(self : Context) -> Int {
  self.identity.ordinal()
}

///|
pub fn Context::capability(self : Context) -> DeviceCapability {
  self.identity.capability()
}

///|
/// Returns only the immutable identity snapshot retained at context open.
/// The returned value cannot access or extend the native context lifetime.
pub fn Context::device_identity(self : Context) -> ContextDeviceIdentity {
  self.identity
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
/// Opaque deterministic lifetime proof for one allocation. A live lease keeps
/// explicit allocation close retryably busy but does not expose memory or
/// launch authority.
pub struct AllocationLease {
  priv raw : @cuda.AllocationLease
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
/// One immutable launch record consumed by an ordered executor at startup.
/// Its function and argument resources are leased natively for the executor's
/// full lifetime, so aliased deterministic close attempts remain busy.
pub struct OrderedKernelLaunch {
  priv function : Function
  priv dimensions : KernelLaunchDimensions
  priv arguments : KernelArguments
}

///|
/// Owns prevalidated ordered AOT launch records, long-lived native resource
/// leases, and one reusable completion event for an exact stream.
pub struct OrderedKernelExecutor {
  priv raw : @cuda.OrderedKernelExecutor
  priv kernel_count : Int
  priv mode : OrderedKernelExecutionMode
}

///|
/// Startup admission for one immutable, already authenticated launch list.
/// Eager fallback is possible only in the explicitly named policy.
pub(all) enum OrderedKernelExecutionPolicy {
  OrderedKernelEagerOnly
  OrderedKernelCaptureRequired
  OrderedKernelCaptureWithEagerFallback
} derive(Debug, Eq)

///|
/// Actual reusable execution mechanism fixed for the executor lifetime.
pub(all) enum OrderedKernelExecutionMode {
  OrderedKernelEager
  OrderedKernelCaptured
} derive(Debug, Eq)

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
