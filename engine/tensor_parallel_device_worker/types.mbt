///|
pub(all) enum TensorParallelDeviceWorkerRule {
  StartupContract
  ExecutionPlan
  ModelGeneration
  GroupGeneration
  Rank
  WorldSize
  DeviceOrdinal
  DeviceTarget
  ProfileEnvelope
  KvBinding
  WeightBinding
  ArtifactBinding
  DescriptorBinding
  RankConstants
  ActivationWorkspace
  FinalOutput
  CollectiveOrder
  PlanFrame
  CompletionRole
  ResourceState
  SequenceOverflow
} derive(Debug, Eq)

///|
pub(all) enum TensorParallelDeviceWorkerResource {
  Collective
  Function
  Module
  KernelExecutor
  Stream
  PersistentKv
  ActivationWorkspace
  RankConstants
  Descriptor
  ShardedWeights
  Context
  CompletionWriter
} derive(Debug, Eq)

///|
pub(all) enum TensorParallelDeviceWorkerFailureClass {
  CollectiveFailure
  RankExecutionFailure
} derive(Debug, Eq)

///|
pub(all) suberror TensorParallelDeviceWorkerError {
  Invalid(TensorParallelDeviceWorkerRule)
  Bootstrap(
    @tensor_parallel_worker_bootstrap.TensorParallelWorkerBootstrapError
  )
  ExecutionPlan(
    @tensor_parallel_execution_plan.TensorParallelExecutionPlanError
  )
  Protocol(@worker_protocol.WorkerProtocolError)
  Wire(@worker_wire.WorkerWireError)
  Device(TensorParallelDeviceWorkerResource, @device.DeviceError)
  Weights(@device_materialize.DeviceWeightError)
  Descriptor(@device_step.DeviceStepError)
  Kv(@device_arena.DeviceKvArenaError)
  Sampling(@sampling.SamplingError)
} derive(Debug, Eq)

///|
pub(all) enum TensorParallelDeviceWorkerLifecycle {
  WorkerStarting
  WorkerReady
  WorkerStaged
  WorkerExecuting
  WorkerCollectivePending
  WorkerDevicePending
  WorkerExecuted
  WorkerDraining
  WorkerFaulted
  WorkerClosingFailed
  WorkerClosed
} derive(Debug, Eq)

///|
priv enum TensorParallelDescriptorResource {
  DescriptorComplete(@device_step.DeviceStepOwner)
  DescriptorFailed(@device_step.FailedDeviceStepPreparation)
}

///|
priv enum TensorParallelWeightResource {
  WeightComplete(@device_materialize.DeviceShardedWeights)
  WeightFailed(@device_materialize.FailedDeviceShardedWeightPreparation)
}

///|
priv enum TensorParallelKvResource {
  KvComplete(@device_arena.DeviceKvArena)
  KvFailed(@device_arena.FailedDeviceKvArenaPreparation)
}

///|
priv struct TensorParallelOwnedAllocation {
  resource : @device.Allocation
  mut open : Bool
}

///|
priv struct TensorParallelOwnedStream {
  resource : @device.Stream
  mut open : Bool
}

///|
priv struct TensorParallelOwnedModule {
  resource : @device.Module
  mut open : Bool
}

///|
priv struct TensorParallelOwnedFunction {
  resource : @device.Function
  module_index : Int
  mut open : Bool
}

///|
priv struct TensorParallelOwnedExecutor {
  resource : @device.OrderedKernelExecutor
  mut open : Bool
  mut abort_required : Bool
}

///|
priv struct TensorParallelOwnedCollective {
  resource : @device.TensorCollectiveOwner
  mut open : Bool
  mut abort_required : Bool
}

///|
priv struct TensorParallelPreparedStep {
  operation_id : @plan.OperationId
}

///|
priv struct TensorParallelPreparedLaunches {
  steps : ReadOnlyArray[TensorParallelPreparedStep]
  launches : ReadOnlyArray[@device.OrderedKernelLaunch]
}

///|
priv struct TensorParallelFunctionBlueprint {
  module_index : Int
  symbol : String
}

///|
priv struct TensorParallelLeaderScratch {
  completion : @worker_wire.CompletionFrameBuffer
  sampling_limits : @sampling.SamplingLimits
  sampler : @sampling.SamplerScratch
  encoded_logits : FixedArray[Byte]
  logits : FixedArray[Double]
  selected_tokens : FixedArray[Int]
  final_output_offset : Int64
  row_bytes : Int64
  vocabulary_size : Int
}

///|
priv enum TensorParallelCompletionAuthority {
  LeaderAuthority(TensorParallelLeaderScratch)
  FollowerAuthority
}

///|
priv struct TensorParallelPartialResources {
  mut collective : TensorParallelOwnedCollective?
  mut executor : TensorParallelOwnedExecutor?
  mut functions : Array[TensorParallelOwnedFunction]
  mut modules : Array[TensorParallelOwnedModule]
  mut stream : TensorParallelOwnedStream?
  mut kv : TensorParallelKvResource?
  mut activation_workspace : TensorParallelOwnedAllocation?
  mut rank_constants : TensorParallelOwnedAllocation?
  mut descriptor : TensorParallelDescriptorResource?
  mut weights : TensorParallelWeightResource?
  mut context : @device.Context?
}

///|
/// Exact warmed ownership set. Every mandatory owner is present without an
/// optional capability slot; booleans exist only for retryable close state.
priv struct TensorParallelLiveResources {
  collective : TensorParallelOwnedCollective
  executor : TensorParallelOwnedExecutor
  functions : Array[TensorParallelOwnedFunction]
  modules : Array[TensorParallelOwnedModule]
  stream : TensorParallelOwnedStream
  kv : @device_arena.DeviceKvArena
  activation_workspace : TensorParallelOwnedAllocation
  rank_constants : TensorParallelOwnedAllocation
  descriptor : @device_step.DeviceStepOwner
  weights : @device_materialize.DeviceShardedWeights
  context : @device.Context
}

///|
pub struct TensorParallelDeviceWorkerPlan {
  priv startup : @worker_wire.WorkerStartupContract
  priv model : @plan.ModelPlan
  priv model_generation : @worker_protocol.ModelPlanGeneration
  priv group_generation : @rank_group_protocol.RankGroupGeneration
  priv inspection : @device_materialize.DeviceShardedWeightFileInspection
  priv execution : @tensor_parallel_execution_plan.TensorParallelExecutionPlan
  priv artifacts : @artifact.KernelArtifactBundle
  priv rank_constants : @tensor_parallel_launch_contract.TensorParallelRankConstants
  priv kv_plan : @tensor_parallel_kv_plan.TensorParallelKvPlan
  priv collective_contract : @device.TensorCollectiveContract
  priv rendezvous_id : @device.TensorCollectiveGroupId
  priv collective_runtime_version_code : Int
  priv descriptor_limits : @device_step.DeviceStepLimits
}

///|
pub struct TensorParallelDeviceWorkerOwner {
  priv plan : TensorParallelDeviceWorkerPlan
  priv resources : TensorParallelLiveResources
  priv steps : ReadOnlyArray[TensorParallelPreparedStep]
  priv completion : TensorParallelCompletionAuthority
  priv mut lifecycle : TensorParallelDeviceWorkerLifecycle
  priv mut active_epoch : UInt64
  priv mut active_sequence : UInt64
  priv mut active_rows : Int
  priv mut active_tokens : Int
  priv mut operation_index : Int
  priv mut collective_index : Int
  priv mut next_global_collective_sequence : UInt64
  priv mut graph_hits : UInt64
  priv mut graph_misses : UInt64
  priv mut last_graph_sequence : UInt64
  priv mut primary_failure : TensorParallelDeviceWorkerError?
  priv mut failure_class : TensorParallelDeviceWorkerFailureClass?
  priv mut cleanup_failure : TensorParallelDeviceWorkerError?
}

///|
pub(all) enum TensorParallelDeviceWorkerPreparation {
  TensorParallelWorkerPrepared(TensorParallelDeviceWorkerOwner)
  TensorParallelWorkerCleanupRequired(
    FailedTensorParallelDeviceWorkerPreparation
  )
}

///|
pub struct FailedTensorParallelDeviceWorkerPreparation {
  priv resources : TensorParallelPartialResources
  priv cause : TensorParallelDeviceWorkerError
  priv cleanup_failure : TensorParallelDeviceWorkerError
  priv mut lifecycle : TensorParallelDeviceWorkerLifecycle
}

///|
#valtype
pub struct StagedTensorParallelDeviceExecution(
  TensorParallelDeviceWorkerOwner,
  UInt64,
  @worker_wire.PlanFrameOwnerIdentity,
  UInt64
)

///|
#valtype
pub struct ExecutedTensorParallelDeviceExecution(
  TensorParallelDeviceWorkerOwner,
  UInt64,
  @worker_wire.PlanFrameOwnerIdentity,
  UInt64
)

///|
#valtype
pub struct TensorParallelDeviceExecutionProgress(
  TensorParallelDeviceWorkerOwner,
  UInt64,
  @worker_wire.PlanFrameOwnerIdentity,
  UInt64,
  Bool
)

///|
#valtype
pub struct TensorParallelFollowerExecutionAck {
  priv group_generation_value : UInt64
  priv model_generation_value : UInt64
  priv plan_sequence_value : UInt64
  priv rank : Int
  priv world_size : Int
}
