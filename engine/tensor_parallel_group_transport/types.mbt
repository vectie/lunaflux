///|
pub(all) enum TensorParallelGroupTransportRule {
  Limit
  Lifecycle
  Source
  Root
  ModelIdentity
  ModelGeneration
  Topology
  RankInspection
  RankCapacity
  DevicePlan
  KvPlan
  Manifest
  ExecutionPlan
  CollectiveRuntime
  GroupGeneration
  Rendezvous
  WorkerContract
  Bootstrap
  Configure
  Process
  SchedulerPredecessor
  Completion
  Cleanup
} derive(Debug, Eq)

///|
pub(all) suberror TensorParallelGroupTransportError {
  Rejected(rule~ : TensorParallelGroupTransportRule, rank~ : Int)
} derive(Debug, Eq)

///|
pub(all) enum TensorParallelGroupTransportLifecycle {
  Prepared
  Ready
  Faulted
  Recovering
  Closing
  Closed
} derive(Debug, Eq)

///|
#valtype
pub(all) enum TensorParallelGroupStart {
  GroupReady
  GroupCleanupRequired
} derive(Debug, Eq)

///|
#valtype
pub(all) enum TensorParallelGroupCleanupProgress {
  GroupCleanupAdvanced
  GroupCleanupPending
  GroupCleanupStuck(rank~ : Int)
  GroupReplacementReady
  GroupCloseComplete
} derive(Debug, Eq)

///|
pub struct TensorParallelGroupPlanningInputs {
  priv source : @worker_wire.EncodedBootstrapSource
  priv model_plan : @plan.ModelPlan
  priv model_generation : @worker_protocol.ModelPlanGeneration
  priv topology_declaration : @device_topology.LocalTopologyDeclaration
  priv topology : @device_topology.AdmittedLocalTopology
  priv inspection_blueprints : ReadOnlyArray[
    @device_materialize.DeviceShardedWeightFileInspection,
  ]
  priv device_step_limits : @device_step.DeviceStepLimits
  priv manifest_limits : @execution_manifest_file.ExecutionManifestFileLimits
  priv launch_limits : @tensor_parallel_launch_contract.TensorParallelLaunchContractLimits
  priv execution_limits : @tensor_parallel_execution_plan.TensorParallelExecutionPlanLimits
  priv worker_limits : @worker_protocol.WorkerProtocolLimits
  priv inference_limits : @inference.InferenceLimits
  priv process_limits : @rank_group_process.RankGroupProcessLimits
  // Exact `WorkerExecutableAdmission.activation_path()` bytes admitted by the
  // deployment boundary. Engine code neither reopens nor re-verifies a path.
  priv admitted_rank_child_activation_path : Bytes
  // Exact digest joined into the same template. The approved transport proves
  // both retained values match its executable capability before opening roots.
  priv rank_child_executable_digest_sha256 : String
  priv template_digest : TensorParallelGroupTemplateDigest
}

///|
/// Canonical pre-resource identity of the required production group shape.
pub struct TensorParallelGroupTemplateDigest {
  priv hex : String
} derive(Debug, Eq)

///|
pub fn TensorParallelGroupTemplateDigest::as_sha256(self : Self) -> String {
  self.hex
}

///|
priv enum TensorParallelGroupOwnerState {
  NoGroup
  Published(@rank_group_process.RankGroupProcessSupervisor)
  Failed(@rank_group_process.RankGroupProcessSupervisor)
  Cleaning(
    @rank_group_process.RankGroupProcessSupervisor,
    replace_after_cleanup~ : Bool
  )
  ClosingRoots
  OwnerClosed
}

///|
pub struct TensorParallelGroupTransport {
  priv inputs : TensorParallelGroupPlanningInputs
  priv approved_executable : @worker_executable_file.WorkerExecutableAdmission
  priv roots : @approved_fs.WorkerApprovedRoots
  priv plan_buffer_identities : ReadOnlyArray[
    @worker_protocol.PlanBufferIdentity,
  ]
  priv mut next_group_generation_value : UInt64
  priv mut current_previous_plan_sequence_value : UInt64
  priv mut active_sequence_value : UInt64
  priv mut completion_accepted : Bool
  priv mut replacement_required : Bool
  priv mut retained_failure_report : @rank_group_protocol.RankGroupFailureReport?
  priv mut failure_report_taken : Bool
  priv mut invalid_failure_report : Bool
  priv mut state : TensorParallelGroupOwnerState
}

///|
pub struct TensorParallelGroupReceivedCompletion(
  @rank_group_process.ReceivedRankGroupCompletion
)

///|
pub struct TensorParallelGroupAcceptedCompletion(
  @rank_group_process.AcceptedRankGroupCompletion
)

///|
priv struct TensorParallelGroupAttempt {
  startups : ReadOnlyArray[@rank_group_process.RankGroupProcessRankStartup]
}
