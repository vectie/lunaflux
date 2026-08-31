///|
fn fused_compiler_record(
  compiler : @luna_specialization.LunaCompilerPolicy,
) -> String {
  let identity = compiler.identity()
  "toolchain_sha256=" +
  identity.toolchain_digest_sha256() +
  "\ncompiler_version=" +
  identity.major().to_string() +
  "." +
  identity.minor().to_string() +
  "." +
  identity.patch().to_string() +
  "\noptimization=" +
  compiler.optimization_level().to_string() +
  "\nfmad=false\nreassociate=false\nmax_registers=" +
  compiler.max_registers().to_string() +
  "\n"
}

///|
fn fused_common_recipe(
  family : String,
  target : @catalog.DeviceTarget,
  profile : @launch_contract.PagedKernelProfile,
  compiler : @luna_specialization.LunaCompilerPolicy,
  source_digest : FusedParallelDigest,
  symbol : String,
  canary : UInt,
  numerical : FusedParallelNumericalPolicy,
  grid_y : Int,
  shared_memory_bytes : Int,
) -> String {
  "schema=lunaflux-fused-parallel-cuda-aot-candidate.v1\nfamily=" +
  family +
  "\ntarget=sm_" +
  target.compute_major().to_string() +
  target.compute_minor().to_string() +
  "\nprofile_id=" +
  profile.id().as_int().to_string() +
  "\nmax_query_rows=" +
  profile.max_query_rows().to_string() +
  "\nmax_query_tokens=" +
  profile.max_query_tokens().to_string() +
  "\nmax_page_table_entries=" +
  profile.max_page_table_entries().to_string() +
  "\nsource_sha256=" +
  source_digest.as_hex() +
  "\nfunction_symbol=" +
  symbol +
  "\nblock=128,1,1\ngrid=" +
  profile.max_query_tokens().to_string() +
  "," +
  grid_y.to_string() +
  ",1\nshared_memory_bytes=" +
  shared_memory_bytes.to_string() +
  "\ndispatch_canary=" +
  canary.to_string() +
  "\nnumerics=" +
  numerical.name() +
  "\nmaximum_absolute_error_ppb=" +
  numerical.maximum_absolute_error_ppb().to_string() +
  "\nmaximum_relative_error_ppb=" +
  numerical.maximum_relative_error_ppb().to_string() +
  "\n" +
  fused_compiler_record(compiler)
}

///|
fn fused_qkv_recipe(
  candidate_source : FusedParallelDigest,
  parameters : FusedQkvParameters,
  qkv : @luna_cuda_projection_aot.ProjectionCudaLowering,
  rope : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  kv_write : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  target : @catalog.DeviceTarget,
  profile : @launch_contract.PagedKernelProfile,
  compiler : @luna_specialization.LunaCompilerPolicy,
  symbol : String,
  canary : UInt,
  numerical : FusedParallelNumericalPolicy,
) -> Bytes {
  let record = fused_common_recipe(
      "qkv-positioned-rope-paged-kvwrite-block128", target, profile, compiler, candidate_source,
      symbol, canary, numerical, 1, 0,
    ) +
    "model_content_sha256=" +
    parameters.model_content +
    "\nmodel_plan_sha256=" +
    parameters.model_plan +
    "\nqkv_operation_id=" +
    parameters.qkv_operation.as_int().to_string() +
    "\nrope_operation_id=" +
    parameters.rope_operation.as_int().to_string() +
    "\nattention_operation_id=" +
    parameters.attention_operation.as_int().to_string() +
    "\nkv_layer=" +
    parameters.kv_layer.as_int().to_string() +
    "\nshape=" +
    parameters.input_width.to_string() +
    "," +
    parameters.output_width.to_string() +
    "," +
    parameters.query_heads.to_string() +
    "," +
    parameters.kv_heads.to_string() +
    "," +
    parameters.head_dimension.to_string() +
    "\nrope_theta=" +
    parameters.rope_theta.to_string() +
    "\ntokens_per_page=" +
    parameters.tokens_per_page.to_string() +
    "\ntotal_page_count=" +
    parameters.total_page_count.to_string() +
    "\npage_stride_bytes=" +
    parameters.page_stride_bytes.to_string() +
    "\ncomponent_stride_bytes=" +
    parameters.component_stride_bytes.to_string() +
    "\nqkv_fallback_source_sha256=" +
    qkv.source_digest().as_hex() +
    "\nqkv_fallback_recipe_sha256=" +
    qkv.recipe_digest().as_hex() +
    "\nrope_fallback_source_sha256=" +
    rope.source_digest().as_hex() +
    "\nrope_fallback_recipe_sha256=" +
    rope.recipe_digest().as_hex() +
    "\nkv_write_fallback_source_sha256=" +
    kv_write.source_digest().as_hex() +
    "\nkv_write_fallback_recipe_sha256=" +
    kv_write.recipe_digest().as_hex() +
    "\nabi=step_counts,query_positions,query_row_offsets,sequence_lengths," +
    "page_table_offsets,page_table_page_indices,input,q_weight,k_weight," +
    "v_weight,rotated_qkv,key_cache,value_cache,dispatch_canary\n" +
    "reference_fallback=three-distinct-correctness-kernels\n" +
    "aot_only=true\nruntime_jit=false\n" +
    "release_binding=candidate-only-promotion-required\nmanifest_bindable=false\n"
  @utf8.encode(record)
}

///|
fn fused_qkv_production_recipe(
  candidate_source : FusedParallelDigest,
  qualification : FusedQkvRopeKvWriteCudaCandidate,
  symbol : String,
) -> Bytes {
  let p = qualification.parameters
  let profile = qualification.profile
  @utf8.encode(
    "schema=lunaflux-fused-parallel-cuda-aot-candidate.v2\n" +
    "family=qkv-positioned-rope-paged-kvwrite-production-block128\n" +
    "execution_policy=production-fast-path\nqualification_candidate_sha256=" +
    qualification.candidate_digest.as_hex() +
    "\nsource_sha256=" +
    candidate_source.as_hex() +
    "\nfunction_symbol=" +
    symbol +
    "\ntarget=sm_" +
    qualification.target.compute_major().to_string() +
    qualification.target.compute_minor().to_string() +
    "\nprofile_id=" +
    profile.id().as_int().to_string() +
    "\nmax_query_rows=" +
    profile.max_query_rows().to_string() +
    "\nmax_query_tokens=" +
    profile.max_query_tokens().to_string() +
    "\nmax_page_table_entries=" +
    profile.max_page_table_entries().to_string() +
    "\nblock=128,1,1\ngrid=" +
    profile.max_query_tokens().to_string() +
    ",1,1\nshared_memory_bytes=0\ndiagnostic_canary=absent\nnumerics=" +
    qualification.numerical_policy.name() +
    "\nmaximum_absolute_error_ppb=" +
    qualification.numerical_policy.maximum_absolute_error_ppb().to_string() +
    "\nmaximum_relative_error_ppb=" +
    qualification.numerical_policy.maximum_relative_error_ppb().to_string() +
    "\n" +
    fused_compiler_record(qualification.compiler) +
    "model_content_sha256=" +
    p.model_content +
    "\nmodel_plan_sha256=" +
    p.model_plan +
    "\nqkv_operation_id=" +
    p.qkv_operation.as_int().to_string() +
    "\nrope_operation_id=" +
    p.rope_operation.as_int().to_string() +
    "\nattention_operation_id=" +
    p.attention_operation.as_int().to_string() +
    "\nkv_layer=" +
    p.kv_layer.as_int().to_string() +
    "\nshape=" +
    p.input_width.to_string() +
    "," +
    p.output_width.to_string() +
    "," +
    p.query_heads.to_string() +
    "," +
    p.kv_heads.to_string() +
    "," +
    p.head_dimension.to_string() +
    "\nrope_theta=" +
    p.rope_theta.to_string() +
    "\ntokens_per_page=" +
    p.tokens_per_page.to_string() +
    "\ntotal_page_count=" +
    p.total_page_count.to_string() +
    "\npage_stride_bytes=" +
    p.page_stride_bytes.to_string() +
    "\ncomponent_stride_bytes=" +
    p.component_stride_bytes.to_string() +
    "\nqkv_fallback_source_sha256=" +
    qualification.qkv_fallback.source_digest().as_hex() +
    "\nrope_fallback_source_sha256=" +
    qualification.rope_fallback.source_digest().as_hex() +
    "\nkv_write_fallback_source_sha256=" +
    qualification.kv_write_fallback.source_digest().as_hex() +
    "\nabi=" +
    QkvPositionedRopePagedKvWriteProductionAbiV2.canonical_name() +
    "\nreference_fallback=three-distinct-correctness-kernels\n" +
    "aot_only=true\nruntime_jit=false\n" +
    "release_binding=external-approval-and-production-physical-gate-required\n" +
    "manifest_bindable=false\n",
  )
}

///|
fn fused_residual_norm_recipe(
  candidate_source : FusedParallelDigest,
  parameters : FusedResidualNormParameters,
  residual : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  norm : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  target : @catalog.DeviceTarget,
  profile : @launch_contract.PagedKernelProfile,
  compiler : @luna_specialization.LunaCompilerPolicy,
  symbol : String,
  canary : UInt,
  numerical : FusedParallelNumericalPolicy,
) -> Bytes {
  let record = fused_common_recipe(
      "residual-rmsnorm-block128",
      target,
      profile,
      compiler,
      candidate_source,
      symbol,
      canary,
      numerical,
      1,
      FUSED_PARALLEL_THREADS * 4,
    ) +
    "model_content_sha256=" +
    parameters.model_content +
    "\nmodel_plan_sha256=" +
    parameters.model_plan +
    "\nresidual_operation_id=" +
    parameters.residual_operation.as_int().to_string() +
    "\nnorm_operation_id=" +
    parameters.norm_operation.as_int().to_string() +
    "\nwidth=" +
    parameters.width.to_string() +
    "\nepsilon=" +
    parameters.epsilon.to_string() +
    "\nresidual_fallback_source_sha256=" +
    residual.source_digest().as_hex() +
    "\nresidual_fallback_recipe_sha256=" +
    residual.recipe_digest().as_hex() +
    "\nnorm_fallback_source_sha256=" +
    norm.source_digest().as_hex() +
    "\nnorm_fallback_recipe_sha256=" +
    norm.recipe_digest().as_hex() +
    "\nabi=step_counts,residual,branch,norm_weight,residual_output," +
    "norm_output,dispatch_canary\n" +
    "reference_fallback=two-distinct-correctness-kernels\n" +
    "aot_only=true\nruntime_jit=false\n" +
    "release_binding=candidate-only-promotion-required\nmanifest_bindable=false\n"
  @utf8.encode(record)
}

///|
fn fused_residual_norm_production_recipe(
  candidate_source : FusedParallelDigest,
  qualification : FusedResidualRmsNormCudaCandidate,
  symbol : String,
) -> Bytes {
  let parameters = qualification.parameters
  let profile = qualification.profile
  let record = "schema=lunaflux-fused-parallel-cuda-aot-candidate.v2\n" +
    "family=residual-rmsnorm-production-block128\n" +
    "execution_policy=production-fast-path\n" +
    "qualification_candidate_sha256=" +
    qualification.candidate_digest.as_hex() +
    "\nsource_sha256=" +
    candidate_source.as_hex() +
    "\nfunction_symbol=" +
    symbol +
    "\ntarget=sm_" +
    qualification.target.compute_major().to_string() +
    qualification.target.compute_minor().to_string() +
    "\nprofile_id=" +
    profile.id().as_int().to_string() +
    "\nmax_query_rows=" +
    profile.max_query_rows().to_string() +
    "\nmax_query_tokens=" +
    profile.max_query_tokens().to_string() +
    "\nmax_page_table_entries=" +
    profile.max_page_table_entries().to_string() +
    "\nblock=128,1,1\ngrid=" +
    profile.max_query_tokens().to_string() +
    ",1,1\nshared_memory_bytes=" +
    (FUSED_PARALLEL_THREADS * 4).to_string() +
    "\ndiagnostic_canary=absent\nnumerics=" +
    qualification.numerical_policy.name() +
    "\nmaximum_absolute_error_ppb=" +
    qualification.numerical_policy.maximum_absolute_error_ppb().to_string() +
    "\nmaximum_relative_error_ppb=" +
    qualification.numerical_policy.maximum_relative_error_ppb().to_string() +
    "\n" +
    fused_compiler_record(qualification.compiler) +
    "model_content_sha256=" +
    parameters.model_content +
    "\nmodel_plan_sha256=" +
    parameters.model_plan +
    "\nresidual_operation_id=" +
    parameters.residual_operation.as_int().to_string() +
    "\nnorm_operation_id=" +
    parameters.norm_operation.as_int().to_string() +
    "\nwidth=" +
    parameters.width.to_string() +
    "\nepsilon=" +
    parameters.epsilon.to_string() +
    "\nresidual_fallback_source_sha256=" +
    qualification.residual_fallback.source_digest().as_hex() +
    "\nresidual_fallback_recipe_sha256=" +
    qualification.residual_fallback.recipe_digest().as_hex() +
    "\nnorm_fallback_source_sha256=" +
    qualification.norm_fallback.source_digest().as_hex() +
    "\nnorm_fallback_recipe_sha256=" +
    qualification.norm_fallback.recipe_digest().as_hex() +
    "\nabi=step_counts,residual,branch,norm_weight,residual_output,norm_output\n" +
    "reference_fallback=two-distinct-correctness-kernels\n" +
    "aot_only=true\nruntime_jit=false\n" +
    "release_binding=external-approval-and-production-physical-gate-required\n" +
    "manifest_bindable=false\n"
  @utf8.encode(record)
}
