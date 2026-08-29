///|
priv struct Fp8EphemeralSpec {
  key : Fp8EphemeralStorageKey
  byte_count : Int64
  alignment : Int64
}

///|
fn operation_error(
  operation : @plan.OperationId,
  rule : Fp8DeviceExecutorRule,
) -> Fp8DeviceExecutorError {
  Fp8OperationRejected(operation~, rule~)
}

///|
fn ephemeral_key(
  role : @launch_contract.AotOperandRole,
) -> Fp8EphemeralStorageKey? {
  match role {
    RuntimeInput(input) => Some(Runtime(input))
    ExternalInput(input) => Some(External(input))
    ActivationInput(reference) | ActivationOutput(reference) =>
      Some(Activation(reference))
    _ => None
  }
}

///|
fn ephemeral_spec_index(
  specs : Array[Fp8EphemeralSpec],
  key : Fp8EphemeralStorageKey,
) -> Int? {
  for index, spec in specs {
    if spec.key == key {
      return Some(index)
    }
  }
  None
}

///|
fn collect_ephemeral_specs(
  recipe : @fp8_runtime_recipe.Fp8RuntimeRecipeV2,
  limits : Fp8DeviceExecutorLimits,
) -> ReadOnlyArray[Fp8EphemeralSpec] raise Fp8DeviceExecutorError {
  let specs : Array[Fp8EphemeralSpec] = []
  let mut total = 0L
  for launch in recipe.launches() {
    for operand in launch.raw_operands() {
      match ephemeral_key(operand.role()) {
        Some(key) =>
          match ephemeral_spec_index(specs, key) {
            Some(index) => {
              let retained = specs[index]
              if retained.byte_count != operand.byte_count() ||
                retained.alignment != operand.alignment() {
                raise operation_error(launch.operation(), EphemeralRegion)
              }
            }
            None => {
              if specs.length() >= limits.max_ephemeral_allocations ||
                operand.byte_count() <= 0L ||
                operand.byte_count() > limits.max_ephemeral_bytes - total {
                raise InvalidFp8DeviceExecutor(Limit)
              }
              total += operand.byte_count()
              specs.push({
                key,
                byte_count: operand.byte_count(),
                alignment: operand.alignment(),
              })
            }
          }
        None =>
          match operand.role() {
            WeightInput(_) | WeightScaleInput(_, _) | Workspace => ()
            PersistentKvComponent(_, _) =>
              raise operation_error(launch.operation(), RawOperands)
            RuntimeInput(_)
            | ExternalInput(_)
            | ActivationInput(_)
            | ActivationOutput(_) =>
              raise operation_error(launch.operation(), EphemeralRegion)
          }
      }
    }
  }
  ReadOnlyArray::from_array(specs)
}

///|
fn fp8_executor_target_supported(target : @catalog.DeviceTarget) -> Bool {
  (target.compute_major() == 8 && target.compute_minor() == 9) ||
  (target.compute_major() == 9 && target.compute_minor() == 0) ||
  (target.compute_major() == 12 && target.compute_minor() == 0)
}

///|
fn validate_fp8_target(
  context : @device.Context,
  recipe : @fp8_runtime_recipe.Fp8RuntimeRecipeV2,
) -> Unit raise Fp8DeviceExecutorError {
  let startup = recipe.startup()
  let numeric = startup.device_numeric_capability()
  let target = startup.kernel_artifacts().manifest().target()
  if !fp8_executor_target_supported(target) ||
    !target.matches_capability(context.capability()) {
    raise InvalidFp8DeviceExecutor(Target)
  }
  let observed = @device.admit_finite_fp8_e4m3_w8a8(context.capability()) catch {
    _ => raise InvalidFp8DeviceExecutor(NumericCapability)
  }
  if observed.digest() != numeric.digest() ||
    observed.compute_major() != target.compute_major() ||
    observed.compute_minor() != target.compute_minor() {
    raise InvalidFp8DeviceExecutor(NumericCapability)
  }
}

///|
fn validate_fp8_weights(
  recipe : @fp8_runtime_recipe.Fp8RuntimeRecipeV2,
  weights : @numeric_weight_file.AuthenticatedNumericWeightAuthority,
) -> @device.Allocation raise Fp8DeviceExecutorError {
  let inspection = recipe.weight_inspection()
  if weights.identity() != inspection.identity() ||
    weights.numeric_schema_digest() != inspection.numeric_schema_digest() ||
    weights.layout() != inspection.layout() ||
    weights.artifact_digest() != inspection.artifact_digest() {
    raise InvalidFp8DeviceExecutor(WeightAuthority)
  }
  weights.allocation() catch {
    error => raise Fp8WeightFailure(error)
  }
}

///|
fn validate_fp8_release_authority(
  recipe : @fp8_runtime_recipe.Fp8RuntimeRecipeV2,
  compiled : ArrayView[@fp8_release_authority.Fp8ReleaseOperationAuthorityV2],
  limits : Fp8DeviceExecutorLimits,
) -> Unit raise Fp8DeviceExecutorError {
  let launches = recipe.launches()
  if launches.is_empty() ||
    launches.length() != compiled.length() ||
    launches.length() > limits.max_operations {
    raise InvalidFp8DeviceExecutor(ReleaseCount)
  }
  let artifacts = recipe.startup().kernel_artifacts().artifacts()
  for index, launch in launches {
    let authority = compiled[index]
    if authority.operation() != launch.operation() {
      raise operation_error(launch.operation(), ReleaseOrder)
    }
    if authority.module_digest() != launch.module_digest() ||
      authority.entry_point() != launch.entry_point() {
      raise operation_error(launch.operation(), ReleaseAuthority)
    }
    if authority.dimensions() != launch.launch_dimensions() {
      raise operation_error(launch.operation(), LaunchDimensions)
    }
    if authority.operands() != launch.raw_operands() {
      raise operation_error(launch.operation(), RawOperands)
    }
    if authority.workspace_bytes() != launch.workspace().byte_length() ||
      authority.workspace_alignment() != launch.workspace().alignment() {
      raise operation_error(launch.operation(), Workspace)
    }
    guard artifacts.find_module(authority.module_digest()) is Some(module_) &&
      artifacts.entry_point(authority.entry_point()) is Some(entry) else {
      raise operation_error(launch.operation(), ArtifactModule)
    }
    if module_.module_bytes() != authority.module_bytes() {
      raise operation_error(launch.operation(), ArtifactModule)
    }
    guard artifacts.modules().get(entry.module_index()) is Some(indexed) else {
      raise operation_error(launch.operation(), ArtifactModule)
    }
    if indexed.digest() != authority.module_digest() ||
      entry.function_symbol() != authority.function_symbol() {
      raise operation_error(launch.operation(), ArtifactSymbol)
    }
  }
}
