///|
pub struct TensorParallelExecutionPlanDigest {
  priv sha256 : String
} derive(Debug, Eq)

///|
pub(all) enum TensorParallelMemorySpace {
  DescriptorSpace
  RankConstantsSpace
  WeightSpace
  PersistentKvSpace
  ActivationWorkspaceSpace
} derive(Debug, Eq)

///|
pub struct TensorParallelScalarRegion {
  priv space : TensorParallelMemorySpace
  priv offset_bytes : Int64
  priv byte_count : Int64
  priv reserved_bytes : Int64
  priv alignment : Int64
} derive(Debug, Eq)

///|
pub(all) enum TensorParallelDescriptorRole {
  RuntimeDescriptor(@plan.OperationRuntimeInput)
  ExternalDescriptor(@plan.ModelInputRef)
} derive(Debug, Eq)

///|
pub struct TensorParallelDescriptorBinding {
  priv role : TensorParallelDescriptorRole
  priv region : TensorParallelScalarRegion
} derive(Debug, Eq)

///|
pub struct TensorParallelActivationBinding {
  priv reference : @plan.ActivationRef
  priv slot : Int
  priv produced_by : @plan.OperationId
  priv live_through_step : Int
  priv region : TensorParallelScalarRegion
} derive(Debug, Eq)

///|
pub struct TensorParallelPhysicalOperand {
  priv role : @tensor_parallel_launch_contract.TensorParallelOperandRole
  priv region : TensorParallelScalarRegion
} derive(Debug, Eq)

///|
pub struct TensorParallelOperationBinding {
  priv operation_id : @plan.OperationId
  priv operation_kind : @plan.OperationKind
  priv capability : @plan.KernelCapabilityId
  priv entry_point : @catalog.AotKernelEntryPoint
  priv module_digest : @catalog.AotArtifactDigest
  priv function_symbol : String
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[TensorParallelPhysicalOperand]
} derive(Debug, Eq)

///|
pub(all) enum TensorParallelPhysicalCollectiveKind {
  InPlaceSumAllReduce
  InPlaceLastAxisAllGather
} derive(Debug, Eq)

///|
pub struct TensorParallelCollectiveBinding {
  priv sequence : Int
  priv operation_id : @plan.OperationId
  priv kind : TensorParallelPhysicalCollectiveKind
  priv region : TensorParallelScalarRegion
  priv local_send_offset_bytes : Int64
  priv local_send_bytes : Int64
  priv receive_bytes : Int64
} derive(Debug, Eq)

///|
pub struct TensorParallelExecutionPlan {
  priv identity : @spec.ModelIdentity
  priv rank : Int
  priv world_size : Int
  priv device_ordinal : Int
  priv target : @catalog.DeviceTarget
  priv catalog_version : @catalog.CatalogVersion
  priv profile : @launch_contract.PagedKernelProfile
  priv digest : TensorParallelExecutionPlanDigest
  priv descriptor_bytes : Int64
  priv rank_constants : TensorParallelScalarRegion
  priv weight_arena_bytes : Int64
  priv kv_arena_bytes : Int64
  priv activation_workspace_bytes : Int64
  priv workspace : TensorParallelScalarRegion
  priv descriptors : ReadOnlyArray[TensorParallelDescriptorBinding]
  priv activations : ReadOnlyArray[TensorParallelActivationBinding]
  priv operations : ReadOnlyArray[TensorParallelOperationBinding]
  priv collectives : ReadOnlyArray[TensorParallelCollectiveBinding]
} derive(Debug, Eq)

///|
priv struct MutableActivationFact {
  reference : @plan.ActivationRef
  byte_count : Int64
  alignment : Int64
  producer : @plan.OperationId
  live_through : Int
  slot : Int
}

///|
priv struct MutableActivationSlot {
  byte_count : Int64
  alignment : Int64
  available_after : Int
  offset_bytes : Int64
}

///|
priv struct PlanningMemory {
  descriptors : ReadOnlyArray[TensorParallelDescriptorBinding]
  descriptor_bytes : Int64
  rank_constants : TensorParallelScalarRegion
  activations : ReadOnlyArray[TensorParallelActivationBinding]
  activation_workspace_bytes : Int64
  workspace : TensorParallelScalarRegion
}
