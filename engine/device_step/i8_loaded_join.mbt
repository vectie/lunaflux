///|
/// Joins the exact pre-hardware I8 blueprint/bootstrap pair with the loader's
/// opaque four-way authenticated numeric-weight authority.
///
/// The resulting value remains inert: it exposes no allocation, module,
/// function, launch, worker-startup, executor, or readiness constructor.
pub fn join_i8_loaded_execution_blueprint_v1(
  blueprint : I8PagedExecutionBlueprint,
  bootstrap : I8DeviceWorkerBootstrapManifestV3,
  weights : @numeric_weight_file.AuthenticatedNumericWeightAuthority,
  expected_weight_artifact_digest~ : @numeric_weight_file.NumericWeightArtifactDigest,
) -> I8LoadedExecutionBlueprint raise DeviceStepError {
  // Liveness is part of the authority. Do not retain this transient alias;
  // executor preparation re-borrows only after all exact evidence is joined.
  ignore(weights.allocation() catch { _ => raise NumericWeightClose })
  let memory_plan = blueprint.memory_plan
  let artifacts = blueprint.artifacts
  if bootstrap.identity() != blueprint.identity() ||
    bootstrap.numeric_schema_digest() != blueprint.numeric_schema_digest() ||
    bootstrap.i8_admission_digest() != blueprint.i8_admission_digest() ||
    bootstrap.device_numeric_digest() != blueprint.device_numeric_digest() ||
    bootstrap.target() != blueprint.target() ||
    bootstrap.catalog_version() != blueprint.catalog_version() ||
    bootstrap.device_step_limits() != blueprint.limits() ||
    bootstrap.step_count() != blueprint.steps().length() {
    raise blueprint_error(Identity)
  }
  if memory_plan.identity() != blueprint.identity() ||
    memory_plan.target() != blueprint.target() ||
    memory_plan.arena_bytes() != blueprint.activation_workspace_arena_bytes() ||
    artifacts.model_identity() != blueprint.identity() ||
    artifacts.device_target() != blueprint.target() ||
    artifacts.catalog_version() != blueprint.catalog_version() ||
    artifacts.module_count() != bootstrap.module_count() ||
    artifacts.entry_point_count() != bootstrap.entry_point_count() {
    raise blueprint_error(Identity)
  }
  for step in blueprint.steps() {
    guard artifacts.entry_point(step.entry_point()) is Some(entry) &&
      artifacts.modules().get(entry.module_index()) is Some(module_image) &&
      entry.module_index() == step.artifact_module_index() &&
      entry.function_symbol() == step.function_symbol() &&
      module_image.digest() == step.entry_point().family().module_digest() else {
      raise blueprint_operation_error(step.operation_id(), ArtifactModule)
    }
    for operand in step.operands() {
      match operand.role() {
        ActivationInput(reference) => {
          let expected = bind_activation_operand(
            step.operation_id(),
            reference,
            false,
            @launch_contract.AotOperand::new(
              role=operand.role(),
              byte_count=operand.byte_count(),
              alignment=operand.alignment(),
            ) catch {
              _ =>
                raise blueprint_operation_error(
                  step.operation_id(),
                  ActivationRegion,
                )
            },
            blueprint.profile(),
            memory_plan,
          )
          if expected != operand {
            raise blueprint_operation_error(
              step.operation_id(),
              ActivationRegion,
            )
          }
        }
        ActivationOutput(reference) => {
          let expected = bind_activation_operand(
            step.operation_id(),
            reference,
            true,
            @launch_contract.AotOperand::new(
              role=operand.role(),
              byte_count=operand.byte_count(),
              alignment=operand.alignment(),
            ) catch {
              _ =>
                raise blueprint_operation_error(
                  step.operation_id(),
                  ActivationRegion,
                )
            },
            blueprint.profile(),
            memory_plan,
          )
          if expected != operand {
            raise blueprint_operation_error(
              step.operation_id(),
              ActivationRegion,
            )
          }
        }
        Workspace => {
          let expected = bind_workspace_operand(
            step.operation_id(),
            @launch_contract.AotOperand::new(
              role=operand.role(),
              byte_count=operand.byte_count(),
              alignment=operand.alignment(),
            ) catch {
              _ =>
                raise blueprint_operation_error(
                  step.operation_id(),
                  WorkspaceRegion,
                )
            },
            memory_plan,
          )
          if expected != operand {
            raise blueprint_operation_error(
              step.operation_id(),
              WorkspaceRegion,
            )
          }
        }
        _ => ()
      }
    }
  }
  if weights.identity() != blueprint.identity() {
    raise blueprint_error(Identity)
  }
  if weights.numeric_schema_digest() != blueprint.numeric_schema_digest() {
    raise blueprint_error(NumericSchema)
  }
  if weights.layout() != blueprint.weight_layout() {
    raise blueprint_error(ScaleRegion)
  }
  if weights.artifact_digest() != expected_weight_artifact_digest {
    raise blueprint_error(ArtifactModule)
  }
  {
    blueprint,
    bootstrap,
    weight_artifact_digest: expected_weight_artifact_digest,
    weights,
  }
}
