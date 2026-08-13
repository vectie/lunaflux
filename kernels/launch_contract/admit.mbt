///|
fn profile_is_known(
  profiles : ArrayView[KernelProfile],
  profile_id : KernelProfileId,
) -> Bool {
  profiles.any(profile => profile.id == profile_id)
}

///|
fn validate_profiles(
  profiles : ArrayView[KernelProfile],
  constraints : @plan.ShapeConstraints,
  limits : LaunchContractLimits,
) -> Unit raise LaunchContractError {
  if profiles.is_empty() || profiles.length() > limits.max_profiles {
    raise ProfileCountExceeded(maximum=limits.max_profiles)
  }
  for index, profile in profiles {
    if profile.batch_rows > constraints.max_batch_rows() ||
      profile.sequence_tokens > constraints.max_sequence_tokens() {
      raise InvalidProfile(OutsideModelEnvelope)
    }
    for earlier in 0..<index {
      if profiles[earlier].id == profile.id {
        raise DuplicateProfile(profile.id)
      }
    }
  }
}

///|
fn operation_index(
  operations : ReadOnlyArray[@plan.PlanOperation],
  operation_id : @plan.OperationId,
) -> Int? {
  for index, operation in operations {
    if operation.id() == operation_id {
      return Some(index)
    }
  }
  None
}

///|
fn role_expected_by_operation(
  role : AotOperandRole,
  operation : @plan.PlanOperation,
) -> Bool {
  match role {
    ExternalInput(reference) =>
      operation.inputs().any(input => input == ModelInput(reference))
    WeightInput(reference) =>
      operation.inputs().any(input => input == Tensor(reference))
    ActivationInput(reference) =>
      operation.inputs().any(input => input == Activation(reference))
    RuntimeInput(_) | PersistentKvComponent(_, _) => false
    ActivationOutput(reference) => operation.outputs().contains(reference)
    Workspace => false
  }
}

///|
fn operand_has_role(
  operands : ReadOnlyArray[AotOperand],
  role : AotOperandRole,
) -> Bool {
  operands.any(operand => operand.role == role)
}

///|
fn validate_operand_set(
  operation : @plan.PlanOperation,
  binding : @catalog.KernelBinding,
  operands : ReadOnlyArray[AotOperand],
  maximum : Int,
) -> Unit raise LaunchContractError {
  if operands.is_empty() || operands.length() > maximum {
    raise OperandCountExceeded(operation=operation.id(), maximum~)
  }
  for index, operand in operands {
    for earlier in 0..<index {
      if operands[earlier].role == operand.role {
        raise InvalidOperand(operation.id(), DuplicateRole)
      }
    }
    match operand.role {
      Workspace =>
        if binding.workspace_bytes() == 0L ||
          operand.byte_count != binding.workspace_bytes() ||
          operand.alignment != binding.workspace_alignment() {
          raise InvalidOperand(operation.id(), WorkspaceMismatch)
        }
      role =>
        if !role_expected_by_operation(role, operation) {
          raise InvalidOperand(operation.id(), UnexpectedSemanticRole)
        }
    }
  }
  for input in operation.inputs() {
    let role = match input {
      ModelInput(reference) => ExternalInput(reference)
      Tensor(reference) => WeightInput(reference)
      Activation(reference) => ActivationInput(reference)
    }
    if !operand_has_role(operands, role) {
      raise InvalidOperand(operation.id(), MissingSemanticRole)
    }
  }
  for output in operation.outputs() {
    if !operand_has_role(operands, ActivationOutput(output)) {
      raise InvalidOperand(operation.id(), MissingSemanticRole)
    }
  }
  let has_workspace = operand_has_role(operands, Workspace)
  if has_workspace != (binding.workspace_bytes() > 0L) {
    raise InvalidOperand(operation.id(), WorkspaceMismatch)
  }
}

///|
fn find_input(
  inputs : ArrayView[AotLaunchContractInput],
  profile_id : KernelProfileId,
  operation_id : @plan.OperationId,
) -> AotLaunchContractInput? {
  for input in inputs {
    if input.profile_id == profile_id && input.operation_id == operation_id {
      return Some(input)
    }
  }
  None
}

///|
/// Admits exact per-profile AOT launch metadata against an already-resolved
/// catalog. Vendor GEMM bindings deliberately have no AOT contract.
///
/// The returned arrays are ordered by profile and semantic operation. Runtime
/// lookup therefore requires no string construction, hash-map growth, symbol
/// parsing, compilation, or JIT.
pub fn admit(
  model_plan : @plan.ModelPlan,
  resolved : @catalog.ResolvedKernelCatalog,
  profiles : ArrayView[KernelProfile],
  inputs : ArrayView[AotLaunchContractInput],
  limits : LaunchContractLimits,
) -> LaunchContractSet raise LaunchContractError {
  if resolved.model_identity() != model_plan.identity() {
    raise IdentityMismatch
  }
  validate_profiles(profiles, model_plan.shape_constraints(), limits)
  if inputs.length() > limits.max_contracts {
    raise ContractCountExceeded(maximum=limits.max_contracts)
  }
  let operations = model_plan.operations()
  let bindings = resolved.bindings()
  if operations.length() != bindings.length() {
    raise BindingCountMismatch
  }
  for index, operation in operations {
    if bindings[index].operation_id() != operation.id() ||
      bindings[index].capability() != operation.required_capability() {
      raise BindingOrderMismatch(operation.id())
    }
  }
  for index, input in inputs {
    if !profile_is_known(profiles, input.profile_id) {
      raise UnknownProfile(input.profile_id)
    }
    guard operation_index(operations, input.operation_id) is Some(op_index) else {
      raise BindingOrderMismatch(input.operation_id)
    }
    let binding = bindings[op_index]
    match binding.implementation() {
      VendorBf16GemmImplicitHeuristicV1 =>
        raise ContractForVendorKernel(input.profile_id, input.operation_id)
      ContentAddressedAot(family) =>
        if input.entry_point.family() != family {
          raise KernelFamilyMismatch(input.profile_id, input.operation_id)
        }
    }
    for earlier in 0..<index {
      if inputs[earlier].profile_id == input.profile_id &&
        inputs[earlier].operation_id == input.operation_id {
        raise DuplicateContract(input.profile_id, input.operation_id)
      }
    }
    validate_operand_set(
      operations[op_index],
      binding,
      input.operands,
      limits.max_operands_per_contract,
    )
  }
  let admitted : Array[AotLaunchContract] = []
  for profile in profiles {
    for index, operation in operations {
      let binding = bindings[index]
      match binding.implementation() {
        VendorBf16GemmImplicitHeuristicV1 => ()
        ContentAddressedAot(_) => {
          guard find_input(inputs, profile.id, operation.id()) is Some(input) else {
            raise MissingContract(profile.id, operation.id())
          }
          if admitted.length() >= limits.max_contracts {
            raise ContractCountExceeded(maximum=limits.max_contracts)
          }
          admitted.push({
            profile,
            operation_id: operation.id(),
            operation_kind: operation.kind(),
            operation_shape: operation.shape(),
            capability: operation.required_capability(),
            entry_point: input.entry_point,
            dimensions: input.dimensions,
            operands: input.operands,
            workspace_bytes: binding.workspace_bytes(),
            workspace_alignment: binding.workspace_alignment(),
          })
        }
      }
    }
  }
  {
    model_identity: model_plan.identity(),
    device_target: resolved.device_target(),
    catalog_version: resolved.catalog_version(),
    profiles: ReadOnlyArray::from_array(profiles),
    contracts: ReadOnlyArray::from_array(admitted),
  }
}
