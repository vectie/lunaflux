///|
fn fp8_family_tag(family : Fp8ProjectionFamily) -> String {
  match family {
    QkvProjection => "qkv_projection"
    DenseProjection => "dense_projection"
    GatedMlp => "gated_mlp"
    LanguageModelHead => "language_model_head"
  }
}

///|
fn fp8_role_tag(role : @launch_contract.AotOperandRole) -> String {
  match role {
    RuntimeInput(StepCounts) => "runtime:step_counts"
    RuntimeInput(_) => "runtime:unsupported"
    ActivationInput(reference) =>
      "activation_in:" + reference.as_int().to_string()
    ActivationOutput(reference) =>
      "activation_out:" + reference.as_int().to_string()
    WeightInput(reference) => "weight:" + reference.as_int().to_string()
    WeightScaleInput(weight, scale) =>
      "weight_scale:" +
      weight.as_int().to_string() +
      ":" +
      scale.tensor().as_int().to_string()
    ExternalInput(TokenIds) => "external:token_ids"
    PersistentKvComponent(layer, component) =>
      "kv:" +
      layer.as_int().to_string() +
      ":" +
      (match component {
        Key => "key"
        Value => "value"
      })
    Workspace => "workspace"
  }
}

///|
fn fp8_operand_tag(operands : ArrayView[@launch_contract.AotOperand]) -> String {
  let output = StringBuilder()
  for index, operand in operands {
    if index > 0 {
      output.write_char(',')
    }
    output.write_string(fp8_role_tag(operand.role()))
    output.write_char('@')
    output.write_string(operand.byte_count().to_string())
    output.write_char('/')
    output.write_string(operand.alignment().to_string())
  }
  output.to_string()
}

///|
fn fp8_bool(value : Bool) -> String {
  if value {
    "true"
  } else {
    "false"
  }
}

///|
fn fp8_projection_recipe(
  parameters : Fp8ProjectionParameters,
  operation : @plan.OperationId,
  execution : @numeric_contract.OperationExecutionDigest,
  entry_point_id : @catalog.AotKernelEntryPointId,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  profile : @launch_contract.PagedKernelProfile,
  operands : ArrayView[@launch_contract.AotOperand],
  dimensions : @launch_contract.AotLaunchDimensions,
  function_symbol : String,
  source_digest : Fp8ProjectionCudaSourceDigest,
) -> Bytes {
  let identity = compiler.identity()
  @utf8.encode(
    "schema=lunaflux-luna-cuda-fp8-projection-aot-recipe-v1\n" +
    "family=" +
    fp8_family_tag(parameters.family) +
    "\n" +
    "operation_id=" +
    operation.as_int().to_string() +
    "\n" +
    "operation_execution_sha256=" +
    execution.as_hex() +
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
    "output=cubin\nrequires_cuda_fp8_header=true\n" +
    "requires_native_e4m3_conversion=true\n" +
    "e4m3_conversion=__nv_cvt_float_to_fp8_satfinite_rne_v1\n" +
    "optimization=" +
    compiler.optimization_level().to_string() +
    "\n" +
    "fmad=" +
    fp8_bool(compiler.fused_multiply_add()) +
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
    "max_query_rows=" +
    profile.max_query_rows().to_string() +
    "\n" +
    "max_query_tokens=" +
    profile.max_query_tokens().to_string() +
    "\n" +
    "max_page_table_entries=" +
    profile.max_page_table_entries().to_string() +
    "\n" +
    "grid=" +
    dimensions.grid_x().to_string() +
    ",1,1\nblock=256,1,1\n" +
    "numeric=bf16_input_dynamic_per_tensor_f32_scale_finite_e4m3_weight_per_tensor_f32_scale_f32_accumulator_bf16_output_v1\n" +
    "raw_parameter_abi=canonical_paged_v4_operands_v1\n" +
    "activation_scale_workspace_bytes=4\nactivation_scale_workspace_alignment=4\n" +
    "workspace_role=activation_scale_workspace_f32_scalar\n" +
    "operands=" +
    fp8_operand_tag(operands) +
    "\n",
  )
}

///|
fn fp8_projection_symbol(family : Fp8ProjectionFamily) -> String {
  "lunaflux_luna_" + fp8_family_tag(family) + "_fp8_e4m3_w8a8_release_v1"
}
