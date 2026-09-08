///|
fn validate_projection_companion(
  operation : @plan.OperationId,
  kind : @plan.OperationKind,
  workspace : Bool,
  dimensions : @launch_contract.AotLaunchDimensions?,
) -> Unit raise DeviceStepError {
  if dimensions is Some(d) {
    if kind != GatedMlp ||
      !workspace ||
      d.grid_y() != 1 ||
      d.grid_z() != 1 ||
      d.block_y() != 1 ||
      d.block_z() != 1 ||
      d.block_x() < 32 ||
      d.block_x() > 512 ||
      (d.block_x() & (d.block_x() - 1)) != 0 ||
      d.shared_memory_bytes() != d.block_x() * 32 {
      raise blueprint_operation_error(operation, ArtifactEntryPoint)
    }
  }
}

///|
fn paged_companion_dimensions(
  blueprint : PagedKvExecutionBlueprint,
  operation : @plan.OperationId,
  legacy : @device.KernelLaunchDimensions,
) -> @device.KernelLaunchDimensions raise DeviceStepError {
  let step = paged_blueprint_step(blueprint, operation)
  match step.companion_dimensions {
    Some(dimensions) => paged_device_launch_dimensions(dimensions)
    None => legacy
  }
}
