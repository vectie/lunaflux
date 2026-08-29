///|
fn residual_contract_operands_match(
  operation : @plan.PlanOperation,
  contract : @launch_contract.PagedKvAotLaunchContract,
  operand_bytes : Int64,
) -> Bool {
  let inputs = operation.inputs()
  let outputs = operation.outputs()
  let operands = contract.operands()
  if inputs.length() != 2 || outputs.length() != 1 || operands.length() != 3 {
    return false
  }
  guard inputs[0] is Activation(lhs) && inputs[1] is Activation(rhs) else {
    return false
  }
  operands[0].role() == ActivationInput(lhs) &&
  operands[1].role() == ActivationInput(rhs) &&
  operands[2].role() == ActivationOutput(outputs[0]) &&
  operands[0].byte_count() == operand_bytes &&
  operands[1].byte_count() == operand_bytes &&
  operands[2].byte_count() == operand_bytes
}

///|
/// Integrates the independently admitted residual-only compiler output. This
/// path accepts only the exact fixed-shape three-pointer ABI; paged residual
/// operations carrying live StepCounts use the pointwise family instead.
pub fn admit_residual_compiled_operation(
  lowering : @luna_cuda_aot.LunaCudaAotLowering,
  compiled : @luna_cuda_aot.LunaCudaAotArtifact,
  specialized : @luna_cuda_aot.LunaCudaAotSpecializedArtifact,
  operation : @plan.PlanOperation,
  contract : @launch_contract.PagedKvAotLaunchContract,
  limits : LunaKernelBundleLimits,
) -> LunaKernelCompiledOperation raise LunaKernelBundleError {
  if operation.kind() != ResidualAdd ||
    operation.id() != contract.operation_id() ||
    operation.id() != specialized.operation_id() ||
    !operation.runtime_inputs().is_empty() {
    raise Unsupported(OperationKind)
  }
  guard operation.shape() is ElementwiseShape(width~) else {
    raise Invalid(OperationKind)
  }
  if width.to_int64() != lowering.width() ||
    contract.dimensions() != lowering.launch_dimensions() ||
    specialized.launch_dimensions() != lowering.launch_dimensions() ||
    !residual_contract_operands_match(
      operation,
      contract,
      lowering.operand_byte_count(),
    ) {
    raise Invalid(OperandContract)
  }
  let entry_input = specialized.entry_point_input(
    contract.entry_point().family().family_id(),
  )
  if entry_input.identity() != contract.entry_point() ||
    specialized.module_digest() != compiled.module_digest() ||
    compiled.module_digest() != contract.entry_point().family().module_digest() ||
    compiled.target() != lowering.target() ||
    compiled.source_digest() != lowering.source_digest() ||
    compiled.recipe_digest() != lowering.recipe_digest() {
    raise Invalid(ModuleIdentity)
  }
  let module_bytes = compiled.module_bytes()
  let module_digest = kernel_bundle_sha256(module_bytes)
  let receipt = LunaDeterministicCompileReceipt::from_offline_builder(
    source_digest_sha256=lowering.source_digest().as_hex(),
    recipe_digest_sha256=lowering.recipe_digest().as_hex(),
    toolchain_digest_sha256=lowering
      .compiler_policy()
      .identity()
      .toolchain_digest_sha256(),
    driver_identity_sha256=compiled.driver_identity_sha256(),
    first_artifact_digest_sha256=module_digest,
    second_artifact_digest_sha256=module_digest,
  )
  checked_compiled_operation(
    contract,
    lowering.target(),
    entry_input.function_symbol(),
    lowering.launch_dimensions(),
    contract.operands(),
    contract.workspace_bytes(),
    contract.workspace_alignment(),
    lowering.source_digest().as_hex(),
    lowering.recipe_digest().as_hex(),
    module_bytes,
    module_bytes,
    receipt,
    limits,
  )
}
