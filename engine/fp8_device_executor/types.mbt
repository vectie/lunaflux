///|
pub(all) enum Fp8DeviceExecutorRule {
  Limit
  RecipeVersion
  Target
  NumericCapability
  WeightAuthority
  ReleaseCount
  ReleaseOrder
  ReleaseAuthority
  ArtifactModule
  ArtifactSymbol
  LaunchDimensions
  RawOperands
  Workspace
  EphemeralRegion
  Operation
  Output
  ScaleEvidence
  Owner
  Epoch
  ResourceState
} derive(Debug, Eq)

///|
pub(all) enum Fp8DeviceExecutorResource {
  WeightLease
  Stream
  Workspace
  Ephemeral
  Module
  Function
  Arguments
  InputUpload
  Launch
  ScaleReadback
  OutputReadback
} derive(Debug, Eq)

///|
pub(all) suberror Fp8DeviceExecutorError {
  InvalidFp8DeviceExecutor(Fp8DeviceExecutorRule)
  Fp8OperationRejected(
    operation~ : @plan.OperationId,
    rule~ : Fp8DeviceExecutorRule
  )
  Fp8DeviceFailure(Fp8DeviceExecutorResource, @device.DeviceError)
  Fp8WeightFailure(@numeric_weight_file.NumericWeightFileError)
} derive(Debug, Eq)

///|
pub struct Fp8DeviceExecutorLimits {
  priv max_operations : Int
  priv max_ephemeral_allocations : Int
  priv max_ephemeral_bytes : Int64
}

///|
pub fn Fp8DeviceExecutorLimits::new(
  max_operations~ : Int,
  max_ephemeral_allocations~ : Int,
  max_ephemeral_bytes~ : Int64,
) -> Fp8DeviceExecutorLimits raise Fp8DeviceExecutorError {
  if max_operations <= 0 ||
    max_operations > 65536 ||
    max_ephemeral_allocations <= 0 ||
    max_ephemeral_allocations > 65536 ||
    max_ephemeral_bytes <= 0L ||
    max_ephemeral_bytes > 1099511627776L {
    raise InvalidFp8DeviceExecutor(Limit)
  }
  { max_operations, max_ephemeral_allocations, max_ephemeral_bytes }
}

///|
pub(all) enum Fp8DeviceExecutorLifecycle {
  ExecutorReady
  ExecutorExecuting
  ExecutorExecuted
  ExecutorPoisoned
  ExecutorClosingFailed
  ExecutorClosed
} derive(Debug, Eq)

///|
priv enum Fp8EphemeralStorageKey {
  Runtime(@plan.OperationRuntimeInput)
  External(@plan.ModelInputRef)
  Activation(@plan.ActivationRef)
} derive(Eq)

///|
priv struct Fp8OwnedStream {
  resource : @device.Stream
  mut open : Bool
}

///|
priv struct Fp8OwnedLease {
  resource : @device.AllocationLease
  mut open : Bool
}

///|
priv struct Fp8OwnedAllocation {
  key : Fp8EphemeralStorageKey?
  resource : @device.Allocation
  byte_count : Int64
  mut open : Bool
}

///|
priv struct Fp8OwnedModule {
  digest : @catalog.AotArtifactDigest
  resource : @device.Module
  mut open : Bool
}

///|
priv struct Fp8OwnedFunction {
  module_index : Int
  resource : @device.Function
  mut open : Bool
}

///|
priv struct Fp8PreparedOperation {
  operation : @plan.OperationId
  function_index : Int
  dimensions : @device.KernelLaunchDimensions
  arguments : @device.KernelArguments
  workspace_offset : Int64
  workspace_bytes : Int64
  outputs : ReadOnlyArray[@plan.ActivationRef]
}

///|
priv struct Fp8DeviceResources {
  context : @device.Context
  mut weights : @numeric_weight_file.AuthenticatedNumericWeightAuthority?
  mut weight_lease : Fp8OwnedLease?
  mut stream : Fp8OwnedStream?
  mut workspace : Fp8OwnedAllocation?
  ephemeral : Array[Fp8OwnedAllocation]
  modules : Array[Fp8OwnedModule]
  functions : Array[Fp8OwnedFunction]
}

///|
pub struct Fp8DeviceExecutor {
  priv resources : Fp8DeviceResources
  priv recipe_digest : @fp8_runtime_recipe.Fp8RuntimeRecipeV2Digest
  priv steps : ReadOnlyArray[Fp8PreparedOperation]
  priv scale_scratch : FixedArray[Byte]
  priv mut epoch : UInt64
  priv mut lifecycle : Fp8DeviceExecutorLifecycle
}

///|
pub fn Fp8DeviceExecutor::recipe_digest(
  self : Fp8DeviceExecutor,
) -> @fp8_runtime_recipe.Fp8RuntimeRecipeV2Digest {
  self.recipe_digest
}

///|
#valtype
pub struct ExecutedFp8Operation {
  priv executor : Fp8DeviceExecutor
  priv operation : @plan.OperationId
  priv epoch : UInt64
}

///|
pub(all) enum Fp8DeviceExecutorPreparation {
  Fp8ExecutorReady(Fp8DeviceExecutor)
  Fp8ExecutorCleanupRequired(FailedFp8DeviceExecutorPreparation)
}

///|
pub struct FailedFp8DeviceExecutorPreparation {
  priv resources : Fp8DeviceResources
  priv cause : Fp8DeviceExecutorError
  priv mut cleanup_failure : Fp8DeviceExecutorError
  priv mut closed : Bool
}

///|
pub fn Fp8DeviceExecutor::lifecycle(
  self : Fp8DeviceExecutor,
) -> Fp8DeviceExecutorLifecycle {
  self.lifecycle
}

///|
pub fn Fp8DeviceExecutor::operation_count(self : Fp8DeviceExecutor) -> Int {
  self.steps.length()
}

///|
pub fn FailedFp8DeviceExecutorPreparation::cause(
  self : FailedFp8DeviceExecutorPreparation,
) -> Fp8DeviceExecutorError {
  self.cause
}

///|
pub fn FailedFp8DeviceExecutorPreparation::cleanup_failure(
  self : FailedFp8DeviceExecutorPreparation,
) -> Fp8DeviceExecutorError {
  self.cleanup_failure
}

///|
pub fn FailedFp8DeviceExecutorPreparation::is_closed(
  self : FailedFp8DeviceExecutorPreparation,
) -> Bool {
  self.closed
}
