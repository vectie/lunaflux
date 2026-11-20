///|
fn qwen_fused_recipe(
  plan : @plan.ModelPlan,
  semantic : @luna_fusion_plan.QkvQkNormPositionedRopePagedKvWritePlan,
  layout : @device_layout.DeviceKvLayout,
  qkv : @luna_cuda_projection_aot.ProjectionCudaLowering,
  qk_norm : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  rope : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  kv_write : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  target : @catalog.DeviceTarget,
  profile : @launch_contract.PagedKernelProfile,
  compiler : @luna_specialization.LunaCompilerPolicy,
  symbol : String,
  dispatch_canary : UInt32,
  grid_x : Int32,
  tile_compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
  physical_program : QwenIngressPhysicalProgram,
  source_digest : FusedParallelDigest,
) -> Bytes {
  let (qkv_index, qk_norm_index, rope_index, attention_index) = semantic.operation_indices()
  let (input_width, output_width) = semantic.projection_shape()
  let (query_heads, kv_heads, head_dimension) = semantic.attention_shape()
  @utf8.encode(
    fused_common_recipe(
      "qkv-qknorm-positioned-rope-paged-kvwrite-head-tiled",
      target,
      profile,
      compiler,
      source_digest,
      symbol,
      dispatch_canary,
      QkvQkNormRopeKvWriteDeterministicDotTreeToleranceV1,
      grid_x,
      physical_program.packing.geometry().0,
      0,
    ) +
    "semantic_family=qkv-qknorm-positioned-rope-paged-kvwrite\n" +
    "tile_program_sha256=" +
    tile_compilation.program().digest().as_hex() +
    "\ntile_optimization_sha256=" +
    tile_compilation.optimization().digest().as_hex() +
    "\ntile_schedule_sha256=" +
    tile_compilation.schedule().digest().as_hex() +
    "\ntile_compilation_sha256=" +
    tile_compilation.digest().as_hex() +
    "\n" +
    "column_fold_plan_sha256=" +
    fused_parallel_digest(physical_program.columns.canonical_bytes()).as_hex() +
    "\nhead_packing_sha256=" +
    fused_parallel_digest(physical_program.packing.canonical_bytes()).as_hex() +
    "\nheads_per_cta=" +
    physical_program.packing.geometry().1.to_string() +
    "\n" +
    "model_content_sha256=" +
    plan.identity().content().as_hex() +
    "\nmodel_plan_sha256=" +
    plan.identity().plan().as_hex() +
    "\nqkv_operation_id=" +
    qkv_index.to_string() +
    "\nqk_norm_operation_id=" +
    qk_norm_index.to_string() +
    "\nrope_operation_id=" +
    rope_index.to_string() +
    "\nattention_operation_id=" +
    attention_index.to_string() +
    "\nshape=" +
    input_width.to_string() +
    "," +
    output_width.to_string() +
    "," +
    query_heads.to_string() +
    "," +
    kv_heads.to_string() +
    "," +
    head_dimension.to_string() +
    "\nnorm_epsilon=" +
    semantic.norm_epsilon().to_string() +
    "\nrope_theta=" +
    semantic.rope_theta().to_string() +
    "\ntokens_per_page=" +
    layout.tokens_per_page().to_string() +
    "\ntotal_page_count=" +
    layout.total_page_count().to_string() +
    "\nqkv_fallback_source_sha256=" +
    qkv.source_digest().as_hex() +
    "\nqk_norm_fallback_source_sha256=" +
    qk_norm.source_digest().as_hex() +
    "\nrope_fallback_source_sha256=" +
    rope.source_digest().as_hex() +
    "\nkv_write_fallback_source_sha256=" +
    kv_write.source_digest().as_hex() +
    "\nabi=step_counts,query_positions,query_row_offsets,sequence_lengths," +
    "page_table_offsets,page_table_page_indices,input,q_weight,k_weight," +
    "v_weight,q_norm_weight,k_norm_weight,rotated_qkv,key_cache,value_cache," +
    "dispatch_canary\nreference_fallback=four-distinct-correctness-kernels\n" +
    "aot_only=true\nruntime_jit=false\nmanifest_bindable=false\n",
  )
}

///|
fn qwen_fused_production_recipe(
  source_digest : FusedParallelDigest,
  qualification : FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
  symbol : String,
  step_rotary_cache~ : Bool = false,
) -> Bytes {
  let semantic = qualification.semantic
  let (qkv_index, qk_norm_index, rope_index, attention_index) = semantic.operation_indices()
  let (input_width, output_width) = semantic.projection_shape()
  let (query_heads, kv_heads, head_dimension) = semantic.attention_shape()
  let profile = qualification.profile
  @utf8.encode(
    "schema=lunaflux-fused-parallel-cuda-aot-candidate.v2\n" +
    "family=qkv-qknorm-positioned-rope-paged-kvwrite-production-head-tiled\n" +
    "execution_policy=production-fast-path\n" +
    "qualification_candidate_sha256=" +
    qualification.candidate_digest.as_hex() +
    "\nsource_sha256=" +
    source_digest.as_hex() +
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
    "\nquery_tile_rows=" +
    qualification.tile_compilation
    .schedule()
    .matrix_pipeline()
    .unwrap()
    .row_width()
    .to_string() +
    "\nheads_per_cta=" +
    qualification.physical_program.packing.geometry().1.to_string() +
    "\nmax_page_table_entries=" +
    profile.max_page_table_entries().to_string() +
    "\nblock=128,1,1\ngrid=" +
    qualification.dimensions.grid_x().to_string() +
    "," +
    qualification.dimensions.grid_y().to_string() +
    ",1\nshared_memory_bytes=" +
    qualification.dimensions.shared_memory_bytes().to_string() +
    "\ndiagnostic_canary=absent\n" +
    fused_compiler_record(qualification.compiler) +
    "tile_program_sha256=" +
    qualification.tile_compilation.program().digest().as_hex() +
    "\ntile_optimization_sha256=" +
    qualification.tile_compilation.optimization().digest().as_hex() +
    "\ntile_schedule_sha256=" +
    qualification.tile_compilation.schedule().digest().as_hex() +
    "\ntile_compilation_sha256=" +
    qualification.tile_compilation.digest().as_hex() +
    "\nmodel_content_sha256=" +
    qualification.model_identity.content().as_hex() +
    "\nmodel_plan_sha256=" +
    qualification.model_identity.plan().as_hex() +
    "\n" +
    "qkv_operation_id=" +
    qkv_index.to_string() +
    "\nqk_norm_operation_id=" +
    qk_norm_index.to_string() +
    "\nrope_operation_id=" +
    rope_index.to_string() +
    "\nattention_operation_id=" +
    attention_index.to_string() +
    "\nshape=" +
    input_width.to_string() +
    "," +
    output_width.to_string() +
    "," +
    query_heads.to_string() +
    "," +
    kv_heads.to_string() +
    "," +
    head_dimension.to_string() +
    "\nnorm_epsilon=" +
    semantic.norm_epsilon().to_string() +
    "\nrope_theta=" +
    semantic.rope_theta().to_string() +
    "\ntokens_per_page=" +
    qualification.layout.tokens_per_page().to_string() +
    "\ntotal_page_count=" +
    qualification.layout.total_page_count().to_string() +
    "\nqkv_fallback_source_sha256=" +
    qualification.qkv_fallback.source_digest().as_hex() +
    "\nqk_norm_fallback_source_sha256=" +
    qualification.qk_norm_fallback.source_digest().as_hex() +
    "\nrope_fallback_source_sha256=" +
    qualification.rope_fallback.source_digest().as_hex() +
    "\nkv_write_fallback_source_sha256=" +
    qualification.kv_write_fallback.source_digest().as_hex() +
    "\nabi=step_counts,query_positions,query_row_offsets,sequence_lengths," +
    "page_table_offsets,page_table_page_indices,input,q_weight,k_weight," +
    "v_weight,q_norm_weight,k_norm_weight,rotated_qkv,key_cache,value_cache" +
    (if step_rotary_cache { ",rotary_pairs\n" } else { "\n" }) +
    (if step_rotary_cache {
      qwen_step_rotary_recipe(symbol, profile, head_dimension)
    } else {
      ""
    }) +
    "reference_fallback=four-distinct-correctness-kernels\n" +
    "aot_only=true\nruntime_jit=false\n" +
    "release_binding=external-approval-and-production-physical-gate-required\n" +
    "manifest_bindable=false\n",
  )
}
