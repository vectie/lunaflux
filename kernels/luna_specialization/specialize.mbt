///|
fn validate_luna_numerical_policy(
  compiler : LunaCompilerPolicy,
  numerical : LunaNumericalPolicy,
) -> Unit raise LunaSpecializationError {
  match numerical {
    LunaBitExactOperationOrderV1 =>
      if compiler.reassociate {
        raise Mismatch(NumericalPolicy)
      }
    LunaToleranceV1(
      maximum_absolute_error_ppb~,
      maximum_relative_error_ppb~,
      ..
    ) =>
      if maximum_absolute_error_ppb < 0L ||
        maximum_relative_error_ppb < 0L ||
        (maximum_absolute_error_ppb == 0L && maximum_relative_error_ppb == 0L) {
        raise Invalid(NumericalPolicy)
      }
  }
}

///|
fn validate_luna_residual_profile_value(
  value : @device_profile.ActivationValueView,
  reference : @plan.ActivationRef,
  width : Int64,
  token_rows : Int64,
) -> Unit raise LunaSpecializationError {
  if value.reference() != reference ||
    value.width() != width ||
    value.token_rows() != token_rows {
    raise Mismatch(OperationBinding)
  }
  if width <= 0L || token_rows <= 0L || width > 9223372036854775807L / 2L {
    raise Invalid(OperationShape)
  }
  let row_bytes = width * 2L
  if token_rows > 9223372036854775807L / row_bytes ||
    value.region().logical_bytes() != token_rows * row_bytes {
    raise Mismatch(TensorLayout)
  }
}

///|
fn validate_luna_residual_binding(
  operation : @plan.PlanOperation,
  profile : @device_profile.OperationExecutionProfile,
  semantic : @luna_tile_ir.LunaTileResidualAddSemantics,
) -> Unit raise LunaSpecializationError {
  guard operation.kind() == ResidualAdd &&
    operation.shape() is ElementwiseShape(width=shape_width) &&
    operation.inputs()
    is [Activation(left_reference), Activation(right_reference)] &&
    operation.outputs() is [output_reference] &&
    profile.inputs() is [Activation(left), Activation(right)] &&
    profile.outputs() is [output] else {
    raise Mismatch(OperationBinding)
  }
  if shape_width <= 0 || semantic.width() != Int64::from_int(shape_width) {
    raise Mismatch(OperationShape)
  }
  let token_rows = profile.kernel_shape().token_rows()
  if left_reference == right_reference ||
    output_reference == left_reference ||
    output_reference == right_reference {
    raise Mismatch(OperationBinding)
  }
  validate_luna_residual_profile_value(
    left,
    left_reference,
    semantic.width(),
    token_rows,
  )
  validate_luna_residual_profile_value(
    right,
    right_reference,
    semantic.width(),
    token_rows,
  )
  validate_luna_residual_profile_value(
    output,
    output_reference,
    semantic.width(),
    token_rows,
  )
}

///|
pub fn specialize_luna_tile(
  model_plan : @plan.ModelPlan,
  static_plan : @device_plan.StaticDevicePlan,
  profile : @device_profile.DeviceExecutionProfile,
  catalog : @catalog.ResolvedKernelCatalog,
  operation_id~ : @plan.OperationId,
  entry_point_id~ : LunaSpecializedEntryPointId,
  program : @luna_tile_ir.LunaTileProgram,
  cuda_aot : @luna_tile_ir.LunaTileCudaAotInput,
  compiler : LunaCompilerPolicy,
  numerical : LunaNumericalPolicy,
  evidence : LunaSpecializationEvidence,
  limits : LunaSpecializationLimits,
) -> LunaSpecializationRecord raise LunaSpecializationError {
  let identity = model_plan.identity()
  if static_plan.identity() != identity ||
    profile.identity() != identity ||
    catalog.model_identity() != identity {
    raise Mismatch(Identity)
  }
  let target = static_plan.target()
  if profile.target() != target || catalog.device_target() != target {
    raise Mismatch(Target)
  }
  if static_plan.catalog_version() != catalog.catalog_version() {
    raise Mismatch(CatalogVersion)
  }
  if cuda_aot.program_digest() != program.digest() {
    raise Mismatch(OperationBinding)
  }
  if model_plan.kv_cache_geometry().dtype() != BFloat16 ||
    !target.supports_bf16() {
    raise Unsupported(Quantization)
  }
  if !program.uses_dtype(BFloat16) ||
    program.uses_dtype(Float16) ||
    program.uses_dtype(Int8) ||
    program.uses_dtype(Int32) {
    raise Unsupported(Quantization)
  }
  validate_luna_numerical_policy(compiler, numerical)
  guard luna_specialization_plan_operation(model_plan, operation_id)
    is Some(operation) else {
    raise Invalid(Operation)
  }
  guard luna_specialization_static_operation(static_plan, operation_id)
    is Some(static_operation) else {
    raise Mismatch(Operation)
  }
  guard profile.operation(operation_id) is Some(operation_profile) else {
    raise Mismatch(ExecutionProfile)
  }
  guard luna_specialization_binding(catalog, operation_id) is Some(binding) else {
    raise Mismatch(OperationBinding)
  }
  guard program.residual_add_semantics() is Some(residual_semantic) else {
    raise Mismatch(OperationBinding)
  }
  validate_luna_residual_binding(
    operation, operation_profile, residual_semantic,
  )
  if static_operation.kind() != operation.kind() ||
    static_operation.shape() != operation.shape() ||
    static_operation.binding() != binding ||
    operation_profile.operation_id() != operation_id {
    raise Mismatch(OperationBinding)
  }
  let module_digest = match binding.implementation() {
    ContentAddressedAot(family) => family.module_digest()
    _ => raise Unsupported(OperationBinding)
  }
  if binding.workspace_bytes() < operation.workspace_bytes() ||
    operation_profile.workspace().logical_bytes() < binding.workspace_bytes() ||
    operation_profile.workspace().alignment_bytes() <
    binding.workspace_alignment() {
    raise Mismatch(Workspace)
  }
  let regions = collect_luna_specialization_regions(
    operation_profile,
    limits.max_bound_regions,
  )
  let (rope_theta_bits, rms_epsilon_bits) = collect_luna_semantic_bits(
    model_plan,
  )
  let canonical = encode_luna_specialization(
    identity,
    target,
    catalog.catalog_version(),
    operation,
    binding,
    module_digest,
    entry_point_id,
    profile,
    compiler,
    numerical,
    regions,
    rope_theta_bits,
    rms_epsilon_bits,
    program,
    cuda_aot,
    evidence,
    limits.max_canonical_bytes,
  )
  let digest = LunaSpecializationDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  {
    identity,
    target,
    catalog_version: catalog.catalog_version(),
    operation_id,
    operation_kind: operation.kind(),
    operation_shape: operation.shape(),
    capability: binding.capability(),
    semantic_version: binding.semantic_version(),
    module_digest,
    entry_point_id,
    compiler,
    numerical,
    quantization: LunaUnquantizedBf16V1,
    profile_shape: profile.shape(),
    regions,
    workspace_bytes: binding.workspace_bytes(),
    workspace_alignment: binding.workspace_alignment(),
    rope_theta_bits,
    rms_epsilon_bits,
    program_digest: program.digest(),
    cuda_aot_digest: cuda_aot.digest(),
    evidence,
    canonical,
    digest,
  }
}
