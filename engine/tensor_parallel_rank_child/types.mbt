///|
pub(all) enum TensorParallelRankChildRule {
  ControlBoundary
  ConfigureFrame
  BindingMismatch
  WorkerMismatch
  InvalidLifecycle
  SourcePolicy
  ModelEnvelope
  Limit
  PlanFrame
  Completion
} derive(Debug, Eq)

///|
pub(all) suberror TensorParallelRankChildError {
  Invalid(TensorParallelRankChildRule)
  Control(@rank_child_control.RankChildControlError)
  Configure(@tensor_parallel_rank_configure.TensorParallelRankConfigureError)
  Worker(@tensor_parallel_device_worker.TensorParallelDeviceWorkerError)
  Root(@approved_fs.ApprovedFsError)
  Topology(@device_topology.LocalTopologyError)
  Config(@config_file.ModelConfigFileError)
  Plan(@plan.PlanValidationError)
  Weights(@device_materialize.DeviceWeightError)
  Runtime(@tensor_parallel_worker_bootstrap.TensorParallelWorkerBootstrapError)
  Kv(@tensor_parallel_kv_plan.TensorParallelKvPlanError)
  DevicePlan(@tensor_parallel_device_plan.TensorParallelDevicePlanError)
  Manifest(@execution_manifest_file.ExecutionManifestFileError)
  Device(@device.DeviceError)
  Wire(@worker_wire.WorkerWireError)
} derive(Debug, Eq)

///|
pub(all) enum TensorParallelRankChildConfigureProgress {
  ConfigurePending
  ConfigureReady(TensorParallelRankChildConfigured)
}

///|
pub(all) enum TensorParallelRankChildReadyProgress {
  CollectivePending
  ReadySending
  ReadyPublished
} derive(Debug, Eq)

///|
pub(all) enum TensorParallelRankChildPreparation {
  RankChildStarting(TensorParallelRankChildStarting)
  RankChildCleanupRequired(FailedTensorParallelRankChildPreparation)
}

///|
pub(all) enum TensorParallelRankChildControlProgress {
  ControlPending
  ControlAdvanced
  ControlClosed
} derive(Debug, Eq)

///|
pub struct TensorParallelRankChildConfigureReceiver {
  priv control : @rank_child_control.RankChildControl
  priv storage : FixedArray[Byte]
  priv mut receive_started : Bool
  priv mut configured : Bool
}

///|
pub struct TensorParallelRankChildConfigured {
  priv receiver : TensorParallelRankChildConfigureReceiver
  priv inbound : @rank_child_control.RankChildInbound
  priv contract : @tensor_parallel_rank_configure.TensorParallelRankConfigureContract
}

///|
pub struct TensorParallelRankChildStarting {
  priv configured : TensorParallelRankChildConfigured
  priv worker : @tensor_parallel_device_worker.TensorParallelDeviceWorkerOwner
  priv plan_buffer : @worker_wire.PlanFrameBuffer
  priv plan_storage : FixedArray[Byte]
  priv completion_storage : FixedArray[Byte]
  priv mut ready_started : Bool
}

///|
pub struct TensorParallelRankChildReady {
  priv control_owner : @rank_child_control.RankChildControl
  priv startup : @worker_wire.WorkerStartupContract
  priv binding : @rank_group_wire.RankGroupWireBinding
  priv worker : @tensor_parallel_device_worker.TensorParallelDeviceWorkerOwner
  priv plan_buffer : @worker_wire.PlanFrameBuffer
  priv plan_storage : FixedArray[Byte]
  priv completion_storage : FixedArray[Byte]
  priv graph_telemetry : @worker_wire.WorkerGraphTelemetrySlot
  priv mut execution : TensorParallelRankChildExecution
  priv mut receive_active : Bool
  priv mut send_active : Bool
  priv mut close_response : Bool
}

///|
priv enum TensorParallelRankChildExecution {
  ChildIdle
  ChildExecuting(
    @worker_wire.ValidatedPlanFrame,
    @tensor_parallel_device_worker.StagedTensorParallelDeviceExecution
  )
  ChildExecuted(
    @worker_wire.ValidatedPlanFrame,
    @tensor_parallel_device_worker.ExecutedTensorParallelDeviceExecution
  )
  ChildExecutionFailed(
    @tensor_parallel_device_worker.TensorParallelDeviceWorkerFailureClass
  )
}

///|
priv enum TensorParallelRankChildCleanupAuthority {
  CleanupLiveWorker(
    @tensor_parallel_device_worker.TensorParallelDeviceWorkerOwner
  )
  CleanupFailedWorker(
    @tensor_parallel_device_worker.FailedTensorParallelDeviceWorkerPreparation
  )
}

///|
pub struct FailedTensorParallelRankChildPreparation {
  priv worker : TensorParallelRankChildCleanupAuthority
  priv cause : TensorParallelRankChildError
  priv mut cleanup_failure : TensorParallelRankChildError?
  priv mut closed : Bool
}

///|
pub fn TensorParallelRankChildConfigured::contract(
  self : TensorParallelRankChildConfigured,
) -> @tensor_parallel_rank_configure.TensorParallelRankConfigureContract {
  self.contract
}

///|
pub fn TensorParallelRankChildReady::worker(
  self : TensorParallelRankChildReady,
) -> @tensor_parallel_device_worker.TensorParallelDeviceWorkerOwner {
  self.worker
}

///|
pub fn TensorParallelRankChildReady::control(
  self : TensorParallelRankChildReady,
) -> @rank_child_control.RankChildControl {
  self.control_owner
}
