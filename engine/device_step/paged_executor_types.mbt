///|
priv struct PagedOwnedStream {
  resource : @device.Stream
  mut open : Bool
}

///|
priv struct PagedOwnedAllocation {
  resource : @device.Allocation
  mut open : Bool
}

///|
priv struct PagedOwnedLease {
  resource : @device.AllocationLease
  mut open : Bool
}

///|
priv struct PagedOwnedModule {
  resource : @device.Module
  mut open : Bool
}

///|
priv struct PagedOwnedFunction {
  resource : @device.Function
  module_index : Int
  mut open : Bool
}

///|
priv enum PagedDescriptorOwner {
  Complete(DeviceStepOwner)
  Failed(FailedDeviceStepPreparation)
}

///|
priv struct PagedGraphResources {
  mut descriptor : PagedDescriptorOwner?
  mut stream : PagedOwnedStream?
  mut weight_lease : PagedOwnedLease?
  mut activation_workspace : PagedOwnedAllocation?
  mut persistent_kv : PagedOwnedAllocation?
  modules : Array[PagedOwnedModule]
  functions : Array[PagedOwnedFunction]
}

///|
priv struct PagedFunctionBlueprint {
  identity : @catalog.AotKernelEntryPoint
  module_index : Int
  symbol : String
}

///|
priv struct PreparedPagedGraphStep {
  operation_id : @plan.OperationId
  function_index : Int
  dimensions : @device.KernelLaunchDimensions
  arguments : @device.KernelArguments
}

///|
/// Single-owner synchronous executor for one admitted full paged graph.
///
/// The owner never exposes allocations, modules, functions, streams, kernel
/// arguments, or close authority for its private native resources.
pub struct PagedGraphExecutor {
  priv resources : PagedGraphResources
  priv steps : ReadOnlyArray[PreparedPagedGraphStep]
  priv mut lifecycle : PagedGraphLifecycle
}

///|
/// Exact staged-step capability. Its constructor and fields remain opaque.
#valtype
pub struct StagedPagedGraphExecution {
  priv executor : PagedGraphExecutor
  priv epoch : UInt64
}

///|
/// Exact proof that every synchronous graph launch completed successfully.
#valtype
pub struct ExecutedPagedGraphExecution {
  priv executor : PagedGraphExecutor
  priv epoch : UInt64
}

///|
pub(all) enum PagedGraphPreparation {
  PagedReady(PagedGraphExecutor)
  PagedCleanupRequired(FailedPagedGraphPreparation)
}

///|
/// Cleanup authority retained when preparation and its first cleanup both
/// fail. No executable owner is returned in this state.
pub struct FailedPagedGraphPreparation {
  priv resources : PagedGraphResources
  priv cause : DeviceStepError
  priv mut cleanup_failure : DeviceStepError
  priv mut lifecycle : PagedGraphLifecycle
}

///|
pub fn PagedGraphExecutor::lifecycle(
  self : PagedGraphExecutor,
) -> PagedGraphLifecycle {
  self.lifecycle
}

///|
pub fn FailedPagedGraphPreparation::cause(
  self : FailedPagedGraphPreparation,
) -> DeviceStepError {
  self.cause
}

///|
pub fn FailedPagedGraphPreparation::cleanup_failure(
  self : FailedPagedGraphPreparation,
) -> DeviceStepError {
  self.cleanup_failure
}

///|
pub fn FailedPagedGraphPreparation::is_closed(
  self : FailedPagedGraphPreparation,
) -> Bool {
  self.lifecycle == GraphClosed
}
