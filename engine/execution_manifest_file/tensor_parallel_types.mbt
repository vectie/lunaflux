///|
/// Immutable result of admitting one exact rank-local tensor-parallel
/// execution manifest.
///
/// It retains no filesystem root, locator, device allocation, backend handle,
/// loaded native module, mutable file buffer, or full-model weight layout.
pub struct TensorParallelExecutionAdmission {
  priv model_plan : @plan.ModelPlan
  priv model_generation : @worker_protocol.ModelPlanGeneration
  priv rank_plan : @tensor_parallel_plan.TensorParallelRankPlan
  priv device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan
  priv local_kv_layout : @device_layout.DeviceKvLayout
  priv local_kv_binding : @tensor_parallel_kv_plan.TensorParallelKvRankBinding
  priv profile : @launch_contract.PagedKernelProfile
  priv contracts : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet
  priv artifacts : @artifact.KernelArtifactBundle
  priv execution_plan : @tensor_parallel_execution_plan.TensorParallelExecutionPlan
}

///|
pub fn TensorParallelExecutionAdmission::model_plan(
  self : TensorParallelExecutionAdmission,
) -> @plan.ModelPlan {
  self.model_plan
}

///|
pub fn TensorParallelExecutionAdmission::model_generation(
  self : TensorParallelExecutionAdmission,
) -> @worker_protocol.ModelPlanGeneration {
  self.model_generation
}

///|
pub fn TensorParallelExecutionAdmission::rank_plan(
  self : TensorParallelExecutionAdmission,
) -> @tensor_parallel_plan.TensorParallelRankPlan {
  self.rank_plan
}

///|
pub fn TensorParallelExecutionAdmission::device_plan(
  self : TensorParallelExecutionAdmission,
) -> @tensor_parallel_device_plan.TensorParallelDevicePlan {
  self.device_plan
}

///|
pub fn TensorParallelExecutionAdmission::local_kv_layout(
  self : TensorParallelExecutionAdmission,
) -> @device_layout.DeviceKvLayout {
  self.local_kv_layout
}

///|
pub fn TensorParallelExecutionAdmission::local_kv_binding(
  self : TensorParallelExecutionAdmission,
) -> @tensor_parallel_kv_plan.TensorParallelKvRankBinding {
  self.local_kv_binding
}

///|
pub fn TensorParallelExecutionAdmission::profile(
  self : TensorParallelExecutionAdmission,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn TensorParallelExecutionAdmission::contracts(
  self : TensorParallelExecutionAdmission,
) -> @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet {
  self.contracts
}

///|
pub fn TensorParallelExecutionAdmission::artifacts(
  self : TensorParallelExecutionAdmission,
) -> @artifact.KernelArtifactBundle {
  self.artifacts
}

///|
pub fn TensorParallelExecutionAdmission::execution_plan(
  self : TensorParallelExecutionAdmission,
) -> @tensor_parallel_execution_plan.TensorParallelExecutionPlan {
  self.execution_plan
}
