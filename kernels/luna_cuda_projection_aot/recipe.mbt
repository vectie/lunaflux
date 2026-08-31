///|
fn projection_family_tag(family : ProjectionAotFamily) -> String {
  match family {
    QkvProjection => "qkv_projection"
    DenseProjection => "dense_projection"
    GatedMlp => "gated_mlp"
    LanguageModelHead => "language_model_head"
  }
}

///|
fn projection_numerical_tag(contract : ProjectionNumericalContract) -> String {
  match contract {
    OrderedF32AccumulationBf16RoundV1 => "ordered_f32_mul_add_bf16_round_v1"
    OrderedF32SiluGatedBf16RoundV1 => "ordered_f32_silu_gated_bf16_round_v1"
    TensorCoreF32AccumulationBf16RoundV1 =>
      "tensor_core_f32_accumulation_bf16_round_v1"
    TensorCoreF32SiluGatedBf16RoundV1 =>
      "tensor_core_f32_silu_gated_bf16_round_v1"
  }
}

///|
fn projection_strategy_tag(
  strategy : @luna_projection_strategy.ProjectionStrategy,
) -> String {
  match strategy.kernel_class() {
    ScalarTile => "scalar-tile"
    SubgroupGemv => "subgroup-gemv"
    MatrixTile => "matrix-tile"
  }
}

///|
fn projection_bool_tag(value : Bool) -> String {
  if value {
    "true"
  } else {
    "false"
  }
}

///|
fn projection_role_tag(role : @launch_contract.AotOperandRole) -> String {
  match role {
    RuntimeInput(StepCounts) => "runtime:step_counts"
    RuntimeInput(_) => "runtime:unsupported"
    ExternalInput(TokenIds) => "external:token_ids"
    WeightInput(reference) => "weight:" + reference.as_int().to_string()
    ActivationInput(reference) =>
      "activation_in:" + reference.as_int().to_string()
    ActivationOutput(reference) =>
      "activation_out:" + reference.as_int().to_string()
    PersistentKvComponent(layer, component) =>
      "kv:" +
      layer.as_int().to_string() +
      ":" +
      (match component {
        Key => "key"
        Value => "value"
      })
    Workspace => "workspace"
    WeightScaleInput(reference, scale) =>
      "weight_scale:" +
      reference.as_int().to_string() +
      ":" +
      scale.tensor().as_int().to_string()
  }
}

///|
fn projection_operand_recipe(
  operands : ArrayView[@launch_contract.AotOperand],
) -> String {
  let result = StringBuilder()
  for index, operand in operands {
    if index > 0 {
      result.write_char(',')
    }
    result.write_string(projection_role_tag(operand.role()))
    result.write_char('@')
    result.write_string(operand.byte_count().to_string())
    result.write_char('/')
    result.write_string(operand.alignment().to_string())
  }
  result.to_string()
}

///|
fn projection_numerical_contract(
  parameters : ProjectionParameters,
  dispatch : @luna_projection_strategy.ProjectionDispatchPlan,
) -> ProjectionNumericalContract {
  if projection_tensor_core_mlp_eligible(parameters, dispatch) {
    return TensorCoreF32SiluGatedBf16RoundV1
  }
  if projection_tensor_core_eligible(parameters, dispatch) {
    return TensorCoreF32AccumulationBf16RoundV1
  }
  match parameters.family {
    GatedMlp => OrderedF32SiluGatedBf16RoundV1
    QkvProjection | DenseProjection | LanguageModelHead =>
      OrderedF32AccumulationBf16RoundV1
  }
}

///|
fn projection_symbol(
  family : ProjectionAotFamily,
  operation : @plan.OperationId,
  entry_point_id : @catalog.AotKernelEntryPointId,
) -> String {
  "lunaflux_luna_" +
  projection_family_tag(family) +
  "_bf16_reference_v1_op_" +
  operation.as_int().to_string() +
  "_ep_" +
  entry_point_id.as_int().to_string()
}

///|
fn projection_recipe(
  parameters : ProjectionParameters,
  operation : @plan.OperationId,
  entry_point_id : @catalog.AotKernelEntryPointId,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  profile : @launch_contract.PagedKernelProfile,
  operands : ArrayView[@launch_contract.AotOperand],
  dimensions : @launch_contract.AotLaunchDimensions,
  numerical_contract : ProjectionNumericalContract,
  dispatch_plan : @luna_projection_strategy.ProjectionDispatchPlan,
  function_symbol : String,
  source_digest : ProjectionCudaSourceDigest,
  embed_greedy_sampling : Bool,
) -> Bytes {
  let identity = compiler.identity()
  @utf8.encode(
    "schema=lunaflux-luna-cuda-projection-aot-recipe-v1\n" +
    "family=" +
    projection_family_tag(parameters.family) +
    "\n" +
    "operation_id=" +
    operation.as_int().to_string() +
    "\n" +
    "profile_id=" +
    profile.id().as_int().to_string() +
    "\n" +
    "entry_point_id=" +
    entry_point_id.as_int().to_string() +
    "\n" +
    "function_symbol=" +
    function_symbol +
    "\n" +
    "source_sha256=" +
    source_digest.as_hex() +
    "\n" +
    "toolchain_sha256=" +
    identity.toolchain_digest_sha256() +
    "\n" +
    "compiler_version=" +
    identity.major().to_string() +
    "." +
    identity.minor().to_string() +
    "." +
    identity.patch().to_string() +
    "\n" +
    "target=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
    "\n" +
    "output=cubin\noptimization=" +
    compiler.optimization_level().to_string() +
    "\n" +
    "fmad=" +
    projection_bool_tag(compiler.fused_multiply_add()) +
    "\n" +
    "reassociate=false\nmax_registers=" +
    compiler.max_registers().to_string() +
    "\n" +
    "input_width=" +
    parameters.input_width.to_string() +
    "\n" +
    "output_width=" +
    parameters.output_width.to_string() +
    "\n" +
    "intermediate_width=" +
    parameters.intermediate_width.to_string() +
    "\n" +
    "max_query_tokens=" +
    profile.max_query_tokens().to_string() +
    "\n" +
    "grid=" +
    dimensions.grid_x().to_string() +
    ",1,1\n" +
    "block=" +
    dimensions.block_x().to_string() +
    ",1,1\nshared_memory_bytes=" +
    dimensions.shared_memory_bytes().to_string() +
    "\nmanifest_bindable=false\n" +
    (if embed_greedy_sampling {
      "embedded_sampling=greedy_bf16_rows_v1\n" +
      "sampling_function_symbol=" +
      @luna_cuda_sampling_aot.greedy_sampling_cuda_function_symbol() +
      "\nsampling_maximum_rows=" +
      profile.max_query_rows().to_string() +
      "\nsampling_vocabulary_size=" +
      parameters.output_width.to_string() +
      "\nsampling_result_row_bytes=8\n"
    } else {
      ""
    }) +
    "vendor_fast_path=row-major-weight-transpose-compatible\n" +
    "decode_strategy=" +
    projection_strategy_tag(dispatch_plan.decode()) +
    "\ndecode_row_bucket=" +
    dispatch_plan.decode().row_bucket().to_string() +
    "\nprefill_strategy=" +
    projection_strategy_tag(dispatch_plan.prefill()) +
    "\nprefill_row_bucket=" +
    dispatch_plan.prefill().row_bucket().to_string() +
    "\n" +
    "numerical_contract=" +
    projection_numerical_tag(numerical_contract) +
    "\n" +
    "operands=" +
    projection_operand_recipe(operands) +
    "\n",
  )
}
