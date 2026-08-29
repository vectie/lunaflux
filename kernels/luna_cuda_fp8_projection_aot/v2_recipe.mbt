///|
fn fp8_policy_tag_v2(
  policy : @numeric_contract.ActivationScalePolicy,
) -> String raise Fp8ProjectionAotError {
  if policy ==
    @numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1() {
    "external_operation_input_f32_v1"
  } else if policy ==
    @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_bound_expf_f32_v2() {
    "gated_mlp_external_and_post_silu_bound_expf_f32_v2"
  } else {
    raise Unsupported(ActivationScalePolicy)
  }
}

///|
fn fp8_projection_recipe_v2(
  parameters : Fp8ProjectionParameters,
  operation : @plan.OperationId,
  execution : @numeric_contract.OperationExecutionDigest,
  policy : @numeric_contract.ActivationScalePolicy,
  entry_point_id : @catalog.AotKernelEntryPointId,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  profile : @launch_contract.PagedKernelProfile,
  operands : ArrayView[@launch_contract.AotOperand],
  dimensions : @launch_contract.AotLaunchDimensions,
  function_symbol : String,
  source_digest : Fp8ProjectionCudaSourceDigest,
  workspace_bytes : Int64,
) -> Bytes raise Fp8ProjectionAotError {
  let identity = compiler.identity()
  let compound = parameters.family == GatedMlp
  @utf8.encode(
    "schema=lunaflux-luna-cuda-fp8-projection-aot-recipe-v2\n" +
    "canonical_domain=lunaflux.luna-cuda-fp8-projection-aot.v2\n" +
    "family=" +
    fp8_family_tag(parameters.family) +
    "\n" +
    "operation_id=" +
    operation.as_int().to_string() +
    "\n" +
    "operation_execution_sha256=" +
    execution.as_hex() +
    "\n" +
    "activation_scale_policy=" +
    fp8_policy_tag_v2(policy) +
    "\n" +
    "activation_scale_stage_0=external_operation_input_v1\n" +
    (if compound {
      "activation_scale_stage_1=post_silu_gate_up_product_v1\n"
    } else {
      ""
    }) +
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
    "canonical_positive_zero=true\n" +
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
    "numeric=bf16_input_dynamic_finite_e4m3_weight_per_tensor_f32_scale_f32_accumulator_bf16_output_v2\n" +
    "gated_exponential=" +
    (if compound { "target_compiler_bound_expf_f32_v2" } else { "absent" }) +
    "\nnonfinite_failure_sentinel_bits=0x7fc00000\n" +
    "workspace_sentinel_initialized_before_validation=true\n" +
    "workspace_success_published_after_full_output_validation=true\n" +
    "executor_must_require_all_scale_cells_finite_positive=true\n" +
    "reference_execution=single_thread_two_pass_validation_v1\n" +
    "raw_parameter_abi=canonical_paged_v4_operands_v2\n" +
    "workspace_bytes=" +
    workspace_bytes.to_string() +
    "\nworkspace_alignment=4\nworkspace_pointer_count=1\n" +
    "workspace_cells=" +
    (if compound {
      "external_f32_at_0,post_silu_f32_at_4"
    } else {
      "external_f32_at_0"
    }) +
    "\nno_grid_synchronization=true\n" +
    "post_silu_scale_reduction=" +
    (if compound {
      "strict_token_row_then_intermediate_column_recompute_v1"
    } else {
      "absent"
    }) +
    "\nscale_consumption=serial_index_zero_exact_ordered_reduction_v2\n" +
    "workspace_publication=serial_index_zero_after_full_output_validation_v2\n" +
    "\noperands=" +
    fp8_operand_tag(operands) +
    "\n",
  )
}

///|
fn fp8_projection_symbol_v2(family : Fp8ProjectionFamily) -> String {
  "lunaflux_luna_" + fp8_family_tag(family) + "_fp8_e4m3_w8a8_release_v2"
}
