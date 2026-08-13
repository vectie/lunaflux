///|
priv struct RegionBinding {
  allocation_space : @device_profile.ExecutionBufferSpace
  offset_bytes : Int64
  byte_count : Int64
  alignment : Int64
}

///|
priv struct AotStepBlueprint {
  function_index : Int
  dimensions : @device.KernelLaunchDimensions
  operands : ReadOnlyArray[RegionBinding]
}

///|
priv struct VendorStepBlueprint {
  operation_id : @plan.OperationId
  rows : Int64
  inner : Int64
  columns : Int64
  workspace_bytes : Int64
  left_offset : Int64
  weight_offset : Int64
  output_offset : Int64
  workspace_offset : Int64
}

///|
priv enum StepBlueprint {
  Aot(AotStepBlueprint)
  Vendor(VendorStepBlueprint)
}

///|
priv struct FunctionBlueprint {
  identity : @catalog.AotKernelEntryPoint
  module_index : Int
  symbol : String
}

///|
priv struct PreparationBlueprint {
  token_staging_bytes : Int64
  vocabulary_size : Int
  steps : ReadOnlyArray[StepBlueprint]
  functions : ReadOnlyArray[FunctionBlueprint]
  has_vendor_steps : Bool
}

///|
priv struct OwnedStream {
  resource : @device.Stream
  mut open : Bool
}

///|
priv struct OwnedAllocation {
  resource : @device.Allocation
  mut open : Bool
}

///|
priv struct OwnedAllocationLease {
  resource : @device.AllocationLease
  mut open : Bool
}

///|
priv struct OwnedArena {
  resource : @device_memory.DeviceMemoryArena
  mut open : Bool
}

///|
priv struct OwnedModule {
  resource : @device.Module
  mut open : Bool
}

///|
priv struct OwnedFunction {
  resource : @device.Function
  module_index : Int
  mut open : Bool
}

///|
priv struct OwnedVendorGemmHandle {
  resource : @device.CublasLt
  mut open : Bool
}

///|
priv struct OwnedVendorPlan {
  resource : @device.Bf16GemmPlan
  mut open : Bool
}

///|
priv struct ExecutorResources {
  mut stream : OwnedStream?
  mut weight_lease : OwnedAllocationLease?
  mut token_staging : OwnedAllocation?
  mut arena : OwnedArena?
  modules : Array[OwnedModule]
  functions : Array[OwnedFunction]
  mut vendor_gemm : OwnedVendorGemmHandle?
  vendor_plans : Array[OwnedVendorPlan]
}

///|
priv struct PreparedAotStep {
  function_index : Int
  dimensions : @device.KernelLaunchDimensions
  arguments : @device.KernelArguments
}

///|
priv struct PreparedVendorStep {
  plan_index : Int
  left_offset : Int64
  weight_offset : Int64
  output_offset : Int64
  workspace_offset : Int64
}

///|
priv enum PreparedStep {
  Aot(PreparedAotStep)
  Vendor(PreparedVendorStep)
}

///|
priv enum ExecutorLifecycle {
  Ready
  ClosingFailed
  Closed
} derive(Eq)

///|
fn lifecycle_is_ready(lifecycle : ExecutorLifecycle) -> Bool {
  lifecycle == Ready
}

///|
fn lifecycle_begin_close() -> ExecutorLifecycle {
  ClosingFailed
}

///|
fn lifecycle_finish_close() -> ExecutorLifecycle {
  Closed
}

///|
fn require_ready(
  lifecycle : ExecutorLifecycle,
) -> Unit raise DeviceExecutorError {
  if !lifecycle_is_ready(lifecycle) {
    raise Closed
  }
}

///|
/// One exact-shape Phase-1 executor prepared entirely before dispatch.
///
/// The caller retains lifecycle authority for `Context` and `DeviceWeights`.
/// A private allocation lease keeps weight close retryably busy until this
/// executor begins closing. The executor owns every resource it opens: lease,
/// stream, token staging, activation/workspace arena, modules, functions,
/// vendor GEMM handle, and GEMM plans.
pub struct PreparedDeviceExecutor {
  priv resources : ExecutorResources
  priv weight_allocation : @device.Allocation
  priv steps : ReadOnlyArray[PreparedStep]
  priv terminal_output : @device_profile.FinalOutputView
  priv token_bytes : Int64
  priv vocabulary_size : Int
  priv mut output_valid : Bool
  priv mut lifecycle : ExecutorLifecycle
}

///|
/// Explicit result of preparation after native resource construction begins.
/// A cleanup guard is returned instead of losing ownership when construction
/// and its first cleanup attempt both fail.
pub(all) enum PreparationOutcome {
  Ready(PreparedDeviceExecutor)
  CleanupRequired(FailedDevicePreparation)
}

///|
/// Opaque owner of resources left open by a failed preparation cleanup.
pub struct FailedDevicePreparation {
  priv resources : ExecutorResources
  priv cause : DeviceExecutorError
  priv mut cleanup_failure : DeviceExecutorError
  priv mut lifecycle : ExecutorLifecycle
}

///|
pub fn FailedDevicePreparation::cause(
  self : FailedDevicePreparation,
) -> DeviceExecutorError {
  self.cause
}

///|
pub fn FailedDevicePreparation::cleanup_failure(
  self : FailedDevicePreparation,
) -> DeviceExecutorError {
  self.cleanup_failure
}

///|
pub fn FailedDevicePreparation::is_closed(
  self : FailedDevicePreparation,
) -> Bool {
  self.lifecycle == Closed
}

///|
pub fn PreparedDeviceExecutor::final_output_view(
  self : PreparedDeviceExecutor,
) -> @device_profile.FinalOutputView {
  self.terminal_output
}

///|
pub fn PreparedDeviceExecutor::token_byte_count(
  self : PreparedDeviceExecutor,
) -> Int64 {
  self.token_bytes
}
