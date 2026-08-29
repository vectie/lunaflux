///|
fn fp8_validate_target_v2(
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
) -> Unit raise Fp8ProjectionAotError {
  let admitted = (target.compute_major() == 8 && target.compute_minor() == 9) ||
    (target.compute_major() == 9 && target.compute_minor() == 0) ||
    (target.compute_major() == 12 && target.compute_minor() == 0)
  if !admitted || !target.supports_bf16() {
    raise Unsupported(DeviceTarget)
  }
  if compiler.reassociate() {
    raise Unsupported(CompilerPolicy)
  }
}

///|
pub fn lower_fp8_projection_cuda_aot_candidate_v2(
  model_plan : @plan.ModelPlan,
  operation~ : @plan.OperationId,
  profile : @launch_contract.PagedKernelProfile,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
  limits : Fp8ProjectionAotLimits,
) -> Fp8ProjectionCudaLoweringV2 raise Fp8ProjectionAotError {
  guard model_plan.operations().get(operation.as_int()) is Some(plan_operation) &&
    plan_operation.id() == operation else {
    raise Mismatch(Operation)
  }
  fp8_validate_target_v2(target, compiler)
  let parameters = fp8_projection_parameters_v2(plan_operation)
  let (execution, policy, workspace) = fp8_validate_schema_v2(
    model_plan, plan_operation,
  )
  let expected = derive_fp8_projection_operands_v2(
    model_plan,
    operation~,
    profile,
  )
  fp8_validate_operands(expected, operands)
  let live = profile.max_query_tokens().to_int64() *
    parameters.output_width.to_int64()
  if live <= 0L || live > 2147483647L * 256L {
    raise Invalid(Profile)
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=((live + 255L) / 256L).to_int(),
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    _ => raise Invalid(Profile)
  }
  let entry_point_id = @catalog.AotKernelEntryPointId::from_stable_int(2) catch {
    _ => raise Invalid(Operation)
  }
  let symbol = fp8_projection_symbol_v2(parameters.family)
  let source = fp8_projection_source_v2(
    symbol,
    parameters,
    profile.max_query_tokens(),
  )
  if source.length() > limits.max_source_bytes {
    raise LimitExceeded(rule=Source, maximum=limits.max_source_bytes)
  }
  let source_digest = Fp8ProjectionCudaSourceDigest(fp8_sha256(source))
  let recipe = fp8_projection_recipe_v2(
    parameters, operation, execution, policy, entry_point_id, target, compiler, profile,
    operands, dimensions, symbol, source_digest, workspace,
  )
  if recipe.length() > limits.max_recipe_bytes {
    raise LimitExceeded(rule=Recipe, maximum=limits.max_recipe_bytes)
  }
  {
    model_identity: model_plan.identity(),
    family: parameters.family,
    operation_id: operation,
    execution_digest: execution,
    activation_scale_policy: policy,
    entry_point_id,
    target,
    profile,
    operands: ReadOnlyArray::from_array(operands),
    dimensions,
    numerical_contract: if parameters.family == GatedMlp {
      GatedExternalAndPostSiluBoundExpfFiniteE4M3F32AccumulateBf16V2
    } else {
      ExternalDynamicPerTensorFiniteE4M3F32AccumulateBf16V2
    },
    compiler,
    function_symbol: symbol,
    source,
    source_digest,
    recipe,
    recipe_digest: Fp8ProjectionCudaRecipeDigest(fp8_sha256(recipe)),
    workspace_bytes: workspace,
  }
}
