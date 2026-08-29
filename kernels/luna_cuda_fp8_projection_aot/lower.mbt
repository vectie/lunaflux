///|
fn fp8_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn fp8_validate_target(
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
) -> Unit raise Fp8ProjectionAotError {
  let admitted = (target.compute_major() == 8 && target.compute_minor() == 9) ||
    (target.compute_major() == 9 && target.compute_minor() == 0)
  if !admitted || !target.supports_bf16() {
    raise Unsupported(DeviceTarget)
  }
  if compiler.reassociate() {
    raise Unsupported(CompilerPolicy)
  }
}

///|
fn fp8_validate_operands(
  expected : ReadOnlyArray[@launch_contract.AotOperand],
  actual : ArrayView[@launch_contract.AotOperand],
) -> Unit raise Fp8ProjectionAotError {
  if expected.length() != actual.length() {
    raise Mismatch(OperandOrder)
  }
  for index, operand in actual {
    if operand.role() != expected[index].role() {
      raise Mismatch(OperandOrder)
    }
    if operand.byte_count() != expected[index].byte_count() {
      raise Mismatch(OperandSize)
    }
    if operand.alignment() != expected[index].alignment() {
      raise Mismatch(OperandAlignment)
    }
  }
}

///|
pub fn lower_fp8_projection_cuda_aot_candidate(
  model_plan : @plan.ModelPlan,
  operation~ : @plan.OperationId,
  profile : @launch_contract.PagedKernelProfile,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
  limits : Fp8ProjectionAotLimits,
) -> Fp8ProjectionCudaLowering raise Fp8ProjectionAotError {
  guard model_plan.operations().get(operation.as_int()) is Some(plan_operation) &&
    plan_operation.id() == operation else {
    raise Mismatch(Operation)
  }
  fp8_validate_target(target, compiler)
  let parameters = fp8_projection_parameters(plan_operation)
  let execution = fp8_validate_schema(model_plan, plan_operation)
  let expected = derive_fp8_projection_operands(model_plan, operation~, profile)
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
  let entry_point_id = @catalog.AotKernelEntryPointId::from_stable_int(1) catch {
    _ => raise Invalid(Operation)
  }
  let symbol = fp8_projection_symbol(parameters.family)
  let source = fp8_projection_source(
    symbol,
    parameters,
    profile.max_query_tokens(),
  )
  if source.length() > limits.max_source_bytes {
    raise LimitExceeded(rule=Source, maximum=limits.max_source_bytes)
  }
  let source_digest = Fp8ProjectionCudaSourceDigest(fp8_sha256(source))
  let recipe = fp8_projection_recipe(
    parameters, operation, execution, entry_point_id, target, compiler, profile,
    operands, dimensions, symbol, source_digest,
  )
  if recipe.length() > limits.max_recipe_bytes {
    raise LimitExceeded(rule=Recipe, maximum=limits.max_recipe_bytes)
  }
  {
    model_identity: model_plan.identity(),
    family: parameters.family,
    operation_id: operation,
    execution_digest: execution,
    entry_point_id,
    target,
    profile,
    input_width: parameters.input_width,
    output_width: parameters.output_width,
    intermediate_width: parameters.intermediate_width,
    operands: ReadOnlyArray::from_array(operands),
    dimensions,
    numerical_contract: if parameters.family == GatedMlp {
      DynamicPerTensorFiniteE4M3F32SiluGatedBf16V1
    } else {
      DynamicPerTensorFiniteE4M3F32AccumulateBf16V1
    },
    compiler,
    function_symbol: symbol,
    source,
    source_digest,
    recipe,
    recipe_digest: Fp8ProjectionCudaRecipeDigest(fp8_sha256(recipe)),
  }
}
