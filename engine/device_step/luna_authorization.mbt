///|
fn validate_luna_execution_authorization(
  blueprint : PagedKvExecutionBlueprint,
  authorization : @luna_artifact_admission.LunaAotKernelAdmission?,
) -> Unit raise DeviceStepError {
  guard authorization is Some(authorization) else { return }
  if authorization.identity() != blueprint.identity() ||
    authorization.target() != blueprint.target() ||
    authorization.catalog_version() != blueprint.catalog_version() ||
    authorization.operation_kind() != ResidualAdd ||
    authorization.capability() != @plan.KernelCapabilityId::residual_add() ||
    authorization.execution_disposition() != LunaValidatedEagerFallback {
    raise InvalidExecutor(ArtifactModule)
  }
  let mut matches = 0
  for step in blueprint.steps() {
    if step.operation_id() == authorization.operation_id() {
      matches += 1
      if step.operation_kind() != authorization.operation_kind() ||
        step.capability() != authorization.capability() ||
        step.entry_point() != authorization.entry_point() ||
        step.artifact_module_index() != authorization.artifact_module_index() {
        raise InvalidExecutor(ArtifactModule)
      }
    }
  }
  if matches != 1 {
    raise InvalidExecutor(ArtifactModule)
  }
}
