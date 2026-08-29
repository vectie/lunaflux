///|
pub struct TensorParallelExecutionPlanLimits {
  priv max_operations : Int
  priv max_operands : Int
  priv max_descriptors : Int
  priv max_collectives : Int
  priv max_descriptor_bytes : Int64
  priv max_activation_workspace_bytes : Int64
  priv artifact_limits : @artifact.KernelArtifactLimits
}

///|
pub fn TensorParallelExecutionPlanLimits::new(
  max_operations~ : Int,
  max_operands~ : Int,
  max_descriptors~ : Int,
  max_collectives~ : Int,
  max_descriptor_bytes~ : Int64,
  max_activation_workspace_bytes~ : Int64,
  artifact_limits~ : @artifact.KernelArtifactLimits,
) -> TensorParallelExecutionPlanLimits raise TensorParallelExecutionPlanError {
  if max_operations <= 0 ||
    max_operands <= 0 ||
    max_descriptors <= 0 ||
    max_collectives <= 0 ||
    max_descriptor_bytes <= 0L ||
    max_activation_workspace_bytes <= 0L {
    raise rejected(InvalidLimit)
  }
  {
    max_operations,
    max_operands,
    max_descriptors,
    max_collectives,
    max_descriptor_bytes,
    max_activation_workspace_bytes,
    artifact_limits,
  }
}

///|
pub fn TensorParallelExecutionPlanLimits::max_operations(
  self : TensorParallelExecutionPlanLimits,
) -> Int {
  self.max_operations
}

///|
pub fn TensorParallelExecutionPlanLimits::max_operands(
  self : TensorParallelExecutionPlanLimits,
) -> Int {
  self.max_operands
}

///|
pub fn TensorParallelExecutionPlanLimits::max_descriptors(
  self : TensorParallelExecutionPlanLimits,
) -> Int {
  self.max_descriptors
}

///|
pub fn TensorParallelExecutionPlanLimits::max_collectives(
  self : TensorParallelExecutionPlanLimits,
) -> Int {
  self.max_collectives
}

///|
pub fn TensorParallelExecutionPlanLimits::max_descriptor_bytes(
  self : TensorParallelExecutionPlanLimits,
) -> Int64 {
  self.max_descriptor_bytes
}

///|
pub fn TensorParallelExecutionPlanLimits::max_activation_workspace_bytes(
  self : TensorParallelExecutionPlanLimits,
) -> Int64 {
  self.max_activation_workspace_bytes
}

///|
pub fn TensorParallelExecutionPlanLimits::artifact_limits(
  self : TensorParallelExecutionPlanLimits,
) -> @artifact.KernelArtifactLimits {
  self.artifact_limits
}
