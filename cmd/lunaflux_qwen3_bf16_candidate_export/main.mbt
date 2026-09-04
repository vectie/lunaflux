///|
fn load_authenticated_paged_plan(
  arguments : Qwen3CandidateExportArguments,
) -> @plan.ModelPlan raise Qwen3Bf16CandidateExportCliError {
  let root = @approved_fs.ApprovedRoot::open_absolute(arguments.model_root) catch {
    _ => raise ModelAdmission
  }
  let result : Result[@plan.ModelPlan, Qwen3Bf16CandidateExportCliError] = try {
    let file = root.open_file(
      @approved_fs.ApprovedRelativeLocator::new(arguments.config_locator),
    )
    let config = file.read_immutable_snapshot(maximum_bytes=1048576L)
    file.close()
    if @crypto.bytes_to_hex_string(@crypto.sha256(config)) !=
      arguments.config_sha256 {
      raise ModelAdmission
    }
    let metadata = @config_reader.parse_qwen3_bytes(
      config,
      @spec.ContentDigest::from_sha256(arguments.model_content_sha256),
      max_bytes=1048576,
      max_depth=16,
    )
    let batch = @qwen3.Qwen3PagedBatchEnvelope::new(arguments.max_query_rows)
    let plan = @qwen3.build_paged(metadata, batch~)
    let inspection = @qwen3_weights.inspect_paged_file(
      root,
      metadata,
      @approved_fs.ApprovedRelativeLocator::new(
        arguments.numeric_weights_locator,
      ),
      expected_artifact_digest=@numeric_weight_file.NumericWeightArtifactDigest::from_sha256(
        arguments.numeric_weight_artifact_sha256,
      ),
      limits=@numeric_weight_file.NumericWeightFileLimits::new(
        max_file_bytes=3221225472L,
        max_header_bytes=16777216,
        transfer_chunk_bytes=4194304,
        max_tensor_count=310,
      ),
      alignment_bytes=256L,
      max_arena_bytes=3221225472L,
      expected_route_manifest_sha256=arguments.weight_route_manifest_sha256,
      batch~,
    )
    if plan.operations().length() != 283 ||
      plan.tensor_count() != 310 ||
      inspection.identity() != plan.identity() {
      raise ModelAdmission
    }
    Ok(plan)
  } catch {
    _ => Err(ModelAdmission)
  }
  let close = try {
    root.close()
    Ok(())
  } catch {
    _ => Err(ModelRootCleanup)
  }
  match (result, close) {
    (Ok(plan), Ok(_)) => plan
    (Err(error), _) | (_, Err(error)) => raise error
  }
}

///|
fn prepare_device_layout(
  plan : @plan.ModelPlan,
  arguments : Qwen3CandidateExportArguments,
) -> @device_layout.DeviceKvLayout raise Qwen3Bf16CandidateExportCliError {
  @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=arguments.tokens_per_page,
      total_page_count=arguments.total_page_count,
      max_arena_bytes=17179869184L,
    ),
  ) catch {
    _ => raise CandidatePreparation
  }
}

///|
fn prepare_candidate_set(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  arguments : Qwen3CandidateExportArguments,
) -> @luna_bf16_kernel_producer.LunaBf16CandidateSet raise Qwen3Bf16CandidateExportCliError {
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=arguments.max_query_rows,
    max_query_tokens=arguments.max_query_tokens,
    max_page_table_entries=arguments.max_page_table_entries,
  ) catch {
    _ => raise CandidatePreparation
  }
  if profile.max_query_rows() != plan.shape_constraints().max_batch_rows() ||
    profile.max_query_tokens() < profile.max_query_rows() ||
    profile.max_query_tokens().to_int64() >
    profile.max_query_rows().to_int64() *
    plan.shape_constraints().max_sequence_tokens().to_int64() {
    raise CandidatePreparation
  }
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  ) catch {
    _ => raise CandidatePreparation
  }
  let compiler = @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256=arguments.toolchain_sha256,
      major=arguments.compiler_major.reinterpret_as_int(),
      minor=arguments.compiler_minor.reinterpret_as_int(),
      patch=arguments.compiler_patch.reinterpret_as_int(),
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  ) catch {
    _ => raise CandidatePreparation
  }
  @luna_bf16_kernel_producer.prepare_luna_bf16_candidate_set(
    plan,
    layout,
    profile,
    target,
    compiler,
    @luna_bf16_kernel_producer.LunaBf16CandidateSetLimits::new(
      max_operations=4096,
      max_source_bytes=16777216,
      max_recipe_bytes=16777216,
      max_total_bytes=1073741824,
    ),
  ) catch {
    _ => raise CandidatePreparation
  }
}

///|
priv struct QwenFusedProductionSet {
  first_operation : Int
  postprocess : @luna_cuda_qknorm_rope_paged_kv_write_aot.QkNormRopePagedKvWriteCudaCandidate
  full_ingress : @luna_cuda_fused_parallel_aot.FusedQwenQkvQkNormRopeKvWriteProductionCudaCandidate
  attention_prefill_frontier : @luna_cuda_attention_tile_aot.CudaAttentionTileAotFrontier
  attention_prefill_partitioned : @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotCompilation
  attention_decode : @luna_cuda_attention_tile_aot.CudaAttentionTileAotCompilation
  attention_read : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyProductionCudaCandidate
  attention_decode_split : @luna_cuda_paged_attention_aot.PagedAttentionDecodeSplitCudaBundle
}

///|
fn qwen_partitioned_prefill_problem(
  maximum_problem : @luna_attention_strategy.AttentionCompileProblem,
  query_tile : Int,
) -> @luna_attention_strategy.AttentionCompileProblem raise Qwen3Bf16CandidateExportCliError {
  let (_, context_tokens) = maximum_problem.dimensions()
  if maximum_problem.phase() != Prefill || query_tile <= 0 {
    raise QwenReadOnlyPreparation
  }
  @luna_attention_strategy.AttentionCompileProblem::new(
    shape=maximum_problem.shape(),
    phase=maximum_problem.phase(),
    numeric_type=maximum_problem.numeric_type(),
    batch_rows=maximum_problem.batch_rows(),
    query_rows=query_tile,
    context_tokens~,
    causal=maximum_problem.is_causal(),
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
}

///|
fn prepare_qwen_compiler_partitioned_prefill(
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  baseline : @luna_cuda_attention_tile_aot.CudaAttentionTileAotCompilation,
) -> @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotCompilation raise Qwen3Bf16CandidateExportCliError {
  let maximum_problem = baseline.functional().plan().problem()
  let (query_tile, _, _) = baseline
    .functional()
    .plan()
    .candidate()
    .tile()
    .dimensions()
  // Partitioning is a specialization for the low-query-parallelism bucket,
  // not for the maximum ragged launch envelope. The generated kernel remains
  // grid-stride and therefore covers larger envelopes, while the functional
  // optimizer sees exactly one live query tile and can expose KV parallelism.
  let problem = qwen_partitioned_prefill_problem(maximum_problem, query_tile)
  let capabilities = @luna_attention_strategy.AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=100000L,
    max_pipeline_stages=2,
    supports_async_copy=false,
    matrix_shape=Some(
      @luna_attention_strategy.AttentionMatrixShape::new(
        rows=16,
        columns=16,
        reduction=16,
      ),
    ),
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
  @luna_cuda_attention_tile_aot.compile_cuda_partitioned_attention_tile_aot(
    @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotRequest::new(
      backend=@luna_cuda_attention_autotune.cuda_sm120_rtx5060ti_backend(),
      target_id=@luna_cuda_attention_autotune.cuda_sm120_rtx5060ti_target(),
      problem~,
      capabilities~,
      device_target=set.target(),
      maximum_shared_memory_bytes=100000L,
      abi=baseline.source().abi(),
      partial_symbol="lunaflux_attention_prefill_partitioned_partial_v1",
      merge_symbol="lunaflux_attention_prefill_partitioned_merge_v1",
      maximum_workspace_bytes=1073741824L,
      maximum_source_bytes=1048576,
    ),
    [],
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
}

///|
fn prepare_qwen_compiler_prefill(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  qkv_index : Int,
  attention_index : Int,
) -> @luna_cuda_attention_tile_aot.CudaAttentionTileAotFrontier raise Qwen3Bf16CandidateExportCliError {
  guard plan.operations()[qkv_index].shape()
    is ProjectionShape(output_width~, ..) &&
    plan.operations()[attention_index].shape()
    is AttentionShape(query_heads~, kv_heads~, head_dimension~) else {
    raise QwenReadOnlyPreparation
  }
  let maximum_rows = set.profile().max_query_rows()
  let maximum_tokens = set.profile().max_query_tokens()
  if maximum_tokens < maximum_rows || layout.page_stride_bytes() % 2L != 0L {
    raise QwenReadOnlyPreparation
  }
  let maximum_query_tiles = maximum_rows +
    (maximum_tokens - maximum_rows + 15) / 16
  let problem = @luna_attention_strategy.AttentionCompileProblem::new(
    shape=@luna_attention_strategy.AttentionShape::new(
      query_heads~,
      kv_heads~,
      head_dimension~,
      tokens_per_page=layout.tokens_per_page(),
    ),
    phase=Prefill,
    numeric_type=AttentionBFloat16,
    query_rows=maximum_query_tiles,
    context_tokens=plan.shape_constraints().max_sequence_tokens(),
    causal=true,
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
  let capabilities = @luna_attention_strategy.AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=100000L,
    max_pipeline_stages=2,
    supports_async_copy=false,
    matrix_shape=Some(
      @luna_attention_strategy.AttentionMatrixShape::new(
        rows=16,
        columns=16,
        reduction=16,
      ),
    ),
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
  let backend = @luna_cuda_attention_autotune.cuda_sm120_rtx5060ti_backend() catch {
    _ => raise QwenReadOnlyPreparation
  }
  let target = @luna_cuda_attention_autotune.cuda_sm120_rtx5060ti_target() catch {
    _ => raise QwenReadOnlyPreparation
  }
  // The earlier exporter reconstructed one historical candidate-310 result
  // and therefore prevented every newer compiler candidate from reaching a
  // release artifact. Until a new exact-shape physical autotune table is
  // supplied, use the pure static model instead of inventing an observation.
  let records : Array[@luna_attention_strategy.AttentionAutotuneRecord] = []
  @luna_cuda_attention_tile_aot.compile_cuda_attention_tile_aot_frontier(
    @luna_cuda_attention_tile_aot.CudaAttentionTileAotRequest::new(
      backend~,
      target_id=target,
      problem~,
      capabilities~,
      device_target=set.target(),
      maximum_shared_memory_bytes=100000L,
      abi=@luna_cuda_attention_tile_source.CudaPagedAttentionTileAbi::new(
        query_heads~,
        key_value_heads=kv_heads,
        head_dimension~,
        tokens_per_page=layout.tokens_per_page(),
        input_row_width=output_width,
        page_stride_values=layout.page_stride_bytes() / 2L,
        physical_page_capacity=layout.total_page_count().to_int64(),
        maximum_query_rows=maximum_rows,
        maximum_query_tokens=maximum_tokens,
        maximum_page_entries=set.profile().max_page_table_entries(),
      ),
      symbol="lunaflux_attention_prefill_tile_compiler_v1",
      maximum_source_bytes=1048576,
    ),
    records,
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
}

///|
fn prepare_qwen_compiler_decode(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  qkv_index : Int,
  attention_index : Int,
) -> @luna_cuda_attention_tile_aot.CudaAttentionTileAotCompilation raise Qwen3Bf16CandidateExportCliError {
  guard plan.operations()[qkv_index].shape()
    is ProjectionShape(output_width~, ..) &&
    plan.operations()[attention_index].shape()
    is AttentionShape(query_heads~, kv_heads~, head_dimension~) else {
    raise QwenReadOnlyPreparation
  }
  let maximum_rows = set.profile().max_query_rows()
  let maximum_tokens = set.profile().max_query_tokens()
  if maximum_tokens < maximum_rows || layout.page_stride_bytes() % 2L != 0L {
    raise QwenReadOnlyPreparation
  }
  let problem = @luna_attention_strategy.AttentionCompileProblem::new(
    shape=@luna_attention_strategy.AttentionShape::new(
      query_heads~,
      kv_heads~,
      head_dimension~,
      tokens_per_page=layout.tokens_per_page(),
    ),
    phase=Decode,
    numeric_type=AttentionBFloat16,
    batch_rows=maximum_rows,
    query_rows=1,
    context_tokens=plan.shape_constraints().max_sequence_tokens(),
    causal=true,
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
  let capabilities = @luna_attention_strategy.AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=100000L,
    max_pipeline_stages=2,
    supports_async_copy=false,
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
  let backend = @luna_cuda_attention_autotune.cuda_sm120_rtx5060ti_backend() catch {
    _ => raise QwenReadOnlyPreparation
  }
  let target = @luna_cuda_attention_autotune.cuda_sm120_rtx5060ti_target() catch {
    _ => raise QwenReadOnlyPreparation
  }
  let records : Array[@luna_attention_strategy.AttentionAutotuneRecord] = []
  @luna_cuda_attention_tile_aot.compile_cuda_attention_tile_aot(
    @luna_cuda_attention_tile_aot.CudaAttentionTileAotRequest::new(
      backend~,
      target_id=target,
      problem~,
      capabilities~,
      device_target=set.target(),
      maximum_shared_memory_bytes=100000L,
      abi=@luna_cuda_attention_tile_source.CudaPagedAttentionTileAbi::new(
        query_heads~,
        key_value_heads=kv_heads,
        head_dimension~,
        tokens_per_page=layout.tokens_per_page(),
        input_row_width=output_width,
        page_stride_values=layout.page_stride_bytes() / 2L,
        physical_page_capacity=layout.total_page_count().to_int64(),
        maximum_query_rows=maximum_rows,
        maximum_query_tokens=maximum_tokens,
        maximum_page_entries=set.profile().max_page_table_entries(),
      ),
      symbol="lunaflux_attention_decode_tile_compiler_v1",
      maximum_source_bytes=1048576,
    ),
    records,
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
}

///|
fn qwen_compiler_attention_recipe(
  operation : Int,
  family : String,
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  compiled : @luna_cuda_attention_tile_aot.CudaAttentionTileAotCompilation,
  frontier_sha256? : String,
  variant_stable_id? : Int,
) -> Bytes {
  let compiler = set.compiler_policy()
  let identity = compiler.identity()
  let source = compiled.source()
  let lowering = compiled.lowering()
  let candidate = compiled.functional().plan().candidate()
  let (query_tile_rows, key_value_tile_rows, head_columns) = candidate
    .tile()
    .dimensions()
  let (grid_x, grid_y, block_x, shared) = lowering.launch()
  let frontier_record = match (frontier_sha256, variant_stable_id) {
    (Some(frontier), Some(stable_id)) =>
      "\nfrontier_sha256=" +
      frontier +
      "\nvariant_stable_id=" +
      stable_id.to_string()
    _ => ""
  }
  @utf8.encode(
    "schema=lunaflux-attention-tile-compiler-cuda-aot-candidate.v1\n" +
    "family=" +
    family +
    "\n" +
    "model_operation_id=" +
    operation.to_string() +
    "\nfunction_symbol=" +
    source.symbol().to_owned() +
    "\ntarget=sm_120\ntoolchain_sha256=" +
    identity.toolchain_digest_sha256() +
    "\ncompiler_version=" +
    identity.major().to_string() +
    "." +
    identity.minor().to_string() +
    "." +
    identity.patch().to_string() +
    "\nsource_sha256=" +
    source.digest().as_hex() +
    "\noptimization=" +
    compiler.optimization_level().to_string() +
    "\nfmad=" +
    compiler.fused_multiply_add().to_string() +
    "\nreassociate=" +
    compiler.reassociate().to_string() +
    "\nmax_registers=" +
    compiler.max_registers().to_string() +
    qwen_compiler_single_launch_record(grid_x, grid_y, block_x, shared) +
    "\nfunctional_sha256=" +
    compiled.functional().digest().as_hex() +
    "\nlowering_sha256=" +
    lowering.digest().as_hex() +
    "\ncandidate_stable_id=" +
    candidate.stable_id().to_string() +
    "\nquery_tile_rows=" +
    query_tile_rows.to_string() +
    "\nkey_value_tile_rows=" +
    key_value_tile_rows.to_string() +
    "\nhead_columns=" +
    head_columns.to_string() +
    frontier_record +
    "\n",
  )
}

///|
/// Chooses a query tile that is strictly wider than the selected tile. When
/// the semantic problem cannot generate one, reuse the selected compilation:
/// the bundle ABI stays stable while dispatch remains on a profile-valid
/// kernel. A wider key/value tile is not a wider query tile.
fn qwen_wide_query_prefill_variant(
  frontier : @luna_cuda_attention_tile_aot.CudaAttentionTileAotFrontier,
) -> @luna_cuda_attention_tile_aot.CudaAttentionTileAotCompilation? {
  let selected = frontier.selected()
  let (selected_query_rows, _, _) = selected
    .functional()
    .plan()
    .candidate()
    .tile()
    .dimensions()
  let mut widest : @luna_cuda_attention_tile_aot.CudaAttentionTileAotCompilation? = None
  let mut widest_rows = selected_query_rows
  let mut widest_shared = 9223372036854775807L
  let mut widest_cost = 9223372036854775807L
  let mut widest_stable_id = 2147483647
  for variant in frontier.variants() {
    let (query_rows, _, _) = variant
      .functional()
      .plan()
      .candidate()
      .tile()
      .dimensions()
    let (_, _, _, shared) = variant.lowering().launch()
    let cost = variant.functional().plan().static_cost()
    let stable_id = variant.functional().plan().candidate().stable_id()
    if variant != selected &&
      query_rows > selected_query_rows &&
      (
        query_rows > widest_rows ||
        (query_rows == widest_rows && shared < widest_shared) ||
        (
          query_rows == widest_rows &&
          shared == widest_shared &&
          cost < widest_cost
        ) ||
        (
          query_rows == widest_rows &&
          shared == widest_shared &&
          cost == widest_cost &&
          stable_id < widest_stable_id
        )
      ) {
      widest = Some(variant)
      widest_rows = query_rows
      widest_shared = shared
      widest_cost = cost
      widest_stable_id = stable_id
    }
  }
  match widest {
    Some(variant) => Some(variant)
    None => Some(selected)
  }
}

///|
/// Canonical launch fragment for one compiler-generated kernel. Partitioned
/// attention deliberately uses a different recipe constructor because it owns
/// two entry points and a workspace.
fn qwen_compiler_single_launch_record(
  grid_x : Int,
  grid_y : Int,
  block_x : Int,
  shared_memory_bytes : Int64,
) -> String {
  "\ngrid=" +
  grid_x.to_string() +
  "," +
  grid_y.to_string() +
  ",1\nblock=" +
  block_x.to_string() +
  ",1,1\nshared_memory_bytes=" +
  shared_memory_bytes.to_string()
}

///|
fn qwen_compiler_partitioned_attention_recipe(
  operation : Int,
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  compiled : @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotCompilation,
) -> Bytes {
  let compiler = set.compiler_policy()
  let identity = compiler.identity()
  let source = compiled.source()
  let lowering = compiled.lowering()
  let (grid_x, grid_y, block_x, shared) = lowering.launch()
  let (partial_symbol, merge_symbol) = source.symbols()
  let workspace = source.workspace()
  let (maximum_offset, denominator_offset, numerator_offset) = workspace.region_offsets_bytes()
  @utf8.encode(
    "schema=lunaflux-attention-tile-compiler-partitioned-cuda-aot-candidate.v1\n" +
    "family=paged-attention-prefill-functional-partitioned-tile\n" +
    "model_operation_id=" +
    operation.to_string() +
    "\npartial_function_symbol=" +
    partial_symbol.to_owned() +
    "\nmerge_function_symbol=" +
    merge_symbol.to_owned() +
    "\ntarget=sm_120\ntoolchain_sha256=" +
    identity.toolchain_digest_sha256() +
    "\ncompiler_version=" +
    identity.major().to_string() +
    "." +
    identity.minor().to_string() +
    "." +
    identity.patch().to_string() +
    "\nsource_sha256=" +
    source.digest().as_hex() +
    "\noptimization=" +
    compiler.optimization_level().to_string() +
    "\nfmad=" +
    compiler.fused_multiply_add().to_string() +
    "\nreassociate=" +
    compiler.reassociate().to_string() +
    "\nmax_registers=" +
    compiler.max_registers().to_string() +
    qwen_compiler_partitioned_launch_record(
      grid_x,
      grid_y,
      lowering.key_value_partitions(),
      block_x,
      shared,
    ) +
    "\nworkspace_bytes=" +
    workspace.total_bytes().to_string() +
    "\nworkspace_offsets=" +
    maximum_offset.to_string() +
    "," +
    denominator_offset.to_string() +
    "," +
    numerator_offset.to_string() +
    "\nfunctional_sha256=" +
    compiled.functional().digest().as_hex() +
    "\nlowering_sha256=" +
    lowering.digest().as_hex() +
    "\n",
  )
}

///|
/// Canonical launch fragment for the partial and merge entry points emitted by
/// the partitioned functional attention compilation.
fn qwen_compiler_partitioned_launch_record(
  grid_x : Int,
  grid_y : Int,
  partitions : Int,
  block_x : Int,
  partial_shared_memory_bytes : Int64,
) -> String {
  "\npartial_grid=" +
  grid_x.to_string() +
  "," +
  grid_y.to_string() +
  "," +
  partitions.to_string() +
  "\nmerge_grid=" +
  grid_x.to_string() +
  "," +
  grid_y.to_string() +
  ",1\nblock=" +
  block_x.to_string() +
  ",1,1\npartial_shared_memory_bytes=" +
  partial_shared_memory_bytes.to_string() +
  "\nmerge_shared_memory_bytes=0"
}

///|
fn qwen_postprocess_activation_input(
  operand : @launch_contract.AotOperand,
) -> @launch_contract.AotOperand raise Qwen3Bf16CandidateExportCliError {
  guard operand.role() is ActivationOutput(reference) else {
    raise QwenIngressPreparation
  }
  @launch_contract.AotOperand::new(
    role=ActivationInput(reference),
    byte_count=operand.byte_count(),
    alignment=operand.alignment(),
  ) catch {
    _ => raise QwenIngressPreparation
  }
}

///|
fn qwen_postprocess_operands(
  qkv : @luna_cuda_projection_aot.ProjectionCudaLowering,
  qk_norm : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  rope : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  kv_write : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
) -> Array[@launch_contract.AotOperand] raise Qwen3Bf16CandidateExportCliError {
  let operands : Array[@launch_contract.AotOperand] = []
  for operand in kv_write.operands() {
    if operand.role() is RuntimeInput(_) {
      operands.push(operand)
    }
  }
  for operand in qkv.operands() {
    if operand.role() is ActivationOutput(_) {
      // Projection writes this allocation, while postprocess reads it. Only
      // the ABI direction changes; extent and alignment remain admitted by
      // the projection lowering.
      operands.push(qwen_postprocess_activation_input(operand))
    }
  }
  for operand in qk_norm.operands() {
    if operand.role() is WeightInput(_) {
      operands.push(operand)
    }
  }
  for operand in rope.operands() {
    if operand.role() is ActivationOutput(_) {
      operands.push(operand)
    }
  }
  for operand in kv_write.operands() {
    if operand.role() is PersistentKvComponent(_, _) {
      operands.push(operand)
    }
  }
  if operands.length() != 12 {
    raise QwenIngressPreparation
  }
  operands
}

///|
fn prepare_qwen_fused_production_set(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
) -> QwenFusedProductionSet raise Qwen3Bf16CandidateExportCliError {
  let candidates = set.candidates()
  for index in 0..<(candidates.length() - 3) {
    guard candidates[index] is Projection(qkv) &&
      candidates[index + 1] is Pointwise(qk_norm) &&
      qk_norm.family() == QkRmsNorm &&
      candidates[index + 2] is Pointwise(rope) &&
      rope.family() == PositionedRotary &&
      candidates[index + 3] is PagedAttention(attention) else {
      continue
    }
    let operation = plan.operations()[index + 3]
    if plan.operations()[index].kind() != QkvProjection ||
      plan.operations()[index + 1].kind() != QkRmsNorm ||
      plan.operations()[index + 2].kind() != RotaryEmbedding ||
      operation.kind() != CausalAttention {
      continue
    }
    let kv_operands : Array[@launch_contract.AotOperand] = []
    for operand in attention.operands() {
      match operand.role() {
        ActivationOutput(_) => ()
        _ => kv_operands.push(operand)
      }
    }
    let pointwise_limits = @luna_cuda_pointwise_aot.PointwiseAotLimits::new(
      max_source_bytes=1048576,
      max_recipe_bytes=65536,
    ) catch {
      _ => raise QwenKvWritePreparation
    }
    let kv_write = @luna_cuda_pointwise_aot.lower_auxiliary_kv_cache_write_cuda_aot(
      operation,
      set.profile(),
      layout,
      @catalog.AotKernelEntryPointId::from_stable_int(3901),
      set.target(),
      set.compiler_policy(),
      kv_operands,
      pointwise_limits,
    ) catch {
      _ => raise QwenKvWritePreparation
    }
    let semantic = @luna_fusion_plan.match_qkv_qknorm_positioned_rope_paged_kv_write(
      plan,
      qkv.operation_id(),
    ) catch {
      _ => raise QwenIngressPreparation
    }
    let postprocess_limits = @luna_cuda_qknorm_rope_paged_kv_write_aot.QkNormRopePagedKvWriteAotLimits::new(
      max_source_bytes=1048576,
      max_recipe_bytes=65536,
      max_module_bytes=8388608L,
      max_receipt_bytes=65536,
      max_binding_bytes=65536,
      maximum_grid_x=4096,
    ) catch {
      _ => raise QwenIngressPreparation
    }
    let postprocess = @luna_cuda_qknorm_rope_paged_kv_write_aot.lower_qknorm_rope_paged_kv_write_cuda_aot_candidate(
      plan,
      semantic,
      set.profile(),
      layout,
      set.target(),
      set.compiler_policy(),
      qwen_postprocess_operands(qkv, qk_norm, rope, kv_write),
      postprocess_limits,
    ) catch {
      _ => raise QwenIngressPreparation
    }
    let full_ingress_qualification = @luna_cuda_fused_parallel_aot.lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_candidate(
      plan,
      layout,
      qkv,
      qk_norm,
      rope,
      kv_write,
      set.target(),
      set.compiler_policy(),
      @luna_cuda_fused_parallel_aot.FusedParallelAotLimits::new(
        maximum_source_bytes=1048576,
        maximum_recipe_bytes=65536,
        maximum_promotion_bytes=65536,
      ),
    ) catch {
      _ => raise QwenIngressPreparation
    }
    let full_ingress = @luna_cuda_fused_parallel_aot.lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_production_candidate(
      full_ingress_qualification,
      @luna_cuda_fused_parallel_aot.FusedParallelAotLimits::new(
        maximum_source_bytes=1048576,
        maximum_recipe_bytes=65536,
        maximum_promotion_bytes=65536,
      ),
    ) catch {
      _ => raise QwenIngressPreparation
    }
    let readonly_limits = @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyAotLimits::new(
      maximum_source_bytes=1048576,
      maximum_recipe_bytes=65536,
      maximum_canonical_bytes=65536,
    ) catch {
      _ => raise QwenReadOnlyPreparation
    }
    let readonly_qualification = @luna_cuda_paged_attention_readonly_aot.lower_paged_attention_readonly_cuda_aot_candidate(
      attention,
      @catalog.AotKernelEntryPointId::from_stable_int(3902),
      readonly_limits,
    ) catch {
      _ => raise QwenReadOnlyPreparation
    }
    let attention_read = @luna_cuda_paged_attention_readonly_aot.lower_paged_attention_readonly_production_cuda_aot_candidate(
      readonly_qualification, readonly_limits,
    ) catch {
      _ => raise QwenReadOnlyPreparation
    }
    let attention_decode_split = @luna_cuda_paged_attention_aot.lower_paged_attention_decode_split_cuda_aot_candidate(
      operation,
      set.profile(),
      layout,
      @catalog.AotKernelEntryPointId::from_stable_int(3903),
      @catalog.AotKernelEntryPointId::from_stable_int(3904),
      set.target(),
      set.compiler_policy(),
      attention.operands(),
      8,
      268435456L,
      @luna_cuda_paged_attention_aot.PagedAttentionAotLimits::new(
        max_source_bytes=1048576,
        max_recipe_bytes=65536,
      ),
    ) catch {
      _ => raise QwenDecodeSplitPreparation
    }
    let attention_prefill_frontier = prepare_qwen_compiler_prefill(
      plan,
      layout,
      set,
      index,
      index + 3,
    )
    let attention_prefill_partitioned = prepare_qwen_compiler_partitioned_prefill(
      set,
      attention_prefill_frontier.selected(),
    )
    let attention_decode = prepare_qwen_compiler_decode(
      plan,
      layout,
      set,
      index,
      index + 3,
    )
    return {
      first_operation: index,
      postprocess,
      full_ingress,
      attention_prefill_frontier,
      attention_prefill_partitioned,
      attention_decode,
      attention_read,
      attention_decode_split,
    }
  }
  raise QwenFusionMissing
}

///|
async fn write_new_candidate_pair(
  root : String,
  source : Bytes,
  recipe : Bytes,
) -> Unit raise Qwen3Bf16CandidateExportCliError {
  @fs.mkdir(root, permission=0o700) catch {
    _ => raise Publication
  }
  @fs.write_file(
    root + "/kernel.cu",
    source,
    create_mode=CreateNew,
    permission=0o600,
    sync=Full,
  ) catch {
    _ => raise Publication
  }
  @fs.write_file(
    root + "/kernel.recipe",
    recipe,
    create_mode=CreateNew,
    permission=0o600,
    sync=Full,
  ) catch {
    _ => raise Publication
  }
}

///|
fn prepare_residual_rmsnorm_candidate(
  plan : @plan.ModelPlan,
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
) -> @luna_cuda_fused_parallel_aot.FusedResidualRmsNormCudaCandidate raise Qwen3Bf16CandidateExportCliError {
  let candidates = set.candidates()
  for index in 0..<(candidates.length() - 1) {
    guard candidates[index] is Pointwise(residual) &&
      residual.family() == ResidualAdd &&
      candidates[index + 1] is Pointwise(norm) &&
      norm.family() == RmsNorm else {
      continue
    }
    return @luna_cuda_fused_parallel_aot.lower_fused_residual_rmsnorm_candidate(
      plan,
      residual,
      norm,
      set.target(),
      set.compiler_policy(),
      @luna_cuda_fused_parallel_aot.FusedParallelAotLimits::new(
        maximum_source_bytes=1048576,
        maximum_recipe_bytes=65536,
        maximum_promotion_bytes=65536,
      ),
    ) catch {
      error => {
        println("residual_fusion_detail=" + fused_parallel_error_name(error))
        raise ResidualFusionPreparation
      }
    }
  }
  raise CandidatePreparation
}

///|
async fn run_qwen3_candidate_export(
  raw_arguments : ArrayView[String],
) -> Unit raise Qwen3Bf16CandidateExportCliError {
  let arguments = parse_qwen3_candidate_export_arguments(raw_arguments)
  let plan = load_authenticated_paged_plan(arguments)
  let layout = prepare_device_layout(plan, arguments)
  let set = prepare_candidate_set(plan, layout, arguments)
  let residual = prepare_residual_rmsnorm_candidate(plan, set)
  let production = @luna_cuda_fused_parallel_aot.lower_fused_residual_rmsnorm_production_candidate(
    residual,
    @luna_cuda_fused_parallel_aot.FusedParallelAotLimits::new(
      maximum_source_bytes=1048576,
      maximum_recipe_bytes=65536,
      maximum_promotion_bytes=65536,
    ),
  ) catch {
    _ => raise ResidualProductionPreparation
  }
  let qwen_fused = prepare_qwen_fused_production_set(plan, layout, set)
  let evidence = @luna_bf16_candidate_export.export_new(
    set,
    arguments.absolute_new_output,
  ) catch {
    _ => raise Publication
  }
  let fused_root = arguments.absolute_new_output + "/reusable-fused-residual"
  write_new_candidate_pair(
    fused_root,
    production.source_bytes(),
    production.recipe_bytes(),
  )
  let partitioned_prefill_root = arguments.absolute_new_output +
    "/reusable-qwen-prefill-partitioned-attention"
  write_new_candidate_pair(
    partitioned_prefill_root,
    qwen_fused.attention_prefill_partitioned.source().bytes(),
    qwen_compiler_partitioned_attention_recipe(
      qwen_fused.first_operation + 3,
      set,
      qwen_fused.attention_prefill_partitioned,
    ),
  )
  let ingress_root = arguments.absolute_new_output + "/reusable-qwen-ingress"
  write_new_candidate_pair(
    ingress_root,
    qwen_fused.postprocess.source_bytes(),
    qwen_fused.postprocess.recipe_bytes(),
  )
  let full_ingress_root = arguments.absolute_new_output +
    "/reusable-qwen-full-ingress"
  write_new_candidate_pair(
    full_ingress_root,
    qwen_fused.full_ingress.source_bytes(),
    qwen_fused.full_ingress.recipe_bytes(),
  )
  let readonly_root = arguments.absolute_new_output +
    "/reusable-qwen-readonly-attention"
  write_new_candidate_pair(
    readonly_root,
    qwen_fused.attention_read.source_bytes(),
    qwen_fused.attention_read.recipe_bytes(),
  )
  let prefill_root = arguments.absolute_new_output +
    "/reusable-qwen-prefill-attention"
  let selected_prefill = qwen_fused.attention_prefill_frontier.selected()
  write_new_candidate_pair(
    prefill_root,
    selected_prefill.source().bytes(),
    qwen_compiler_attention_recipe(
      qwen_fused.first_operation + 3,
      "paged-attention-prefill-functional-tile",
      set,
      selected_prefill,
      frontier_sha256=qwen_fused.attention_prefill_frontier.digest().as_hex(),
      variant_stable_id=selected_prefill
        .functional()
        .plan()
        .candidate()
        .stable_id(),
    ),
  )
  for variant in qwen_fused.attention_prefill_frontier.variants() {
    let stable_id = variant.functional().plan().candidate().stable_id()
    let variant_root = arguments.absolute_new_output +
      "/reusable-qwen-prefill-attention-c" +
      stable_id.to_string()
    write_new_candidate_pair(
      variant_root,
      variant.source().bytes(),
      qwen_compiler_attention_recipe(
        qwen_fused.first_operation + 3,
        "paged-attention-prefill-functional-tile-frontier",
        set,
        variant,
        frontier_sha256=qwen_fused.attention_prefill_frontier.digest().as_hex(),
        variant_stable_id=stable_id,
      ),
    )
  }
  let wide_prefill = qwen_wide_query_prefill_variant(
    qwen_fused.attention_prefill_frontier,
  )
  let wide_prefill_root = arguments.absolute_new_output +
    "/reusable-qwen-prefill-wide-query-attention"
  match wide_prefill {
    Some(variant) =>
      write_new_candidate_pair(
        wide_prefill_root,
        variant.source().bytes(),
        qwen_compiler_attention_recipe(
          qwen_fused.first_operation + 3,
          "paged-attention-prefill-wide-query-functional-tile",
          set,
          variant,
          frontier_sha256=qwen_fused.attention_prefill_frontier
            .digest()
            .as_hex(),
          variant_stable_id=variant.functional().plan().candidate().stable_id(),
        ),
      )
    None => ()
  }
  let decode_root = arguments.absolute_new_output +
    "/reusable-qwen-decode-attention"
  write_new_candidate_pair(
    decode_root,
    qwen_fused.attention_decode.source().bytes(),
    qwen_compiler_attention_recipe(
      qwen_fused.first_operation + 3,
      "paged-attention-decode-functional-tile",
      set,
      qwen_fused.attention_decode,
    ),
  )
  let decode_split_root = arguments.absolute_new_output +
    "/reusable-qwen-decode-split-attention"
  write_new_candidate_pair(
    decode_split_root,
    qwen_fused.attention_decode_split.source_bytes(),
    qwen_fused.attention_decode_split.recipe_bytes(),
  )
  println("schema=lunaflux-qwen3-bf16-candidate-export-cli.v1")
  println("recipe=dense_qwen3_bf16_paged_aot_v12")
  println(
    "weight_route_manifest_sha256=" + arguments.weight_route_manifest_sha256,
  )
  println("model_content_sha256=" + evidence.model_content_sha256())
  println("model_plan_sha256=" + evidence.model_plan_sha256())
  println("target=" + evidence.target())
  println("max_query_rows=" + arguments.max_query_rows.to_string())
  println("max_query_tokens=" + arguments.max_query_tokens.to_string())
  println("operation_count=" + evidence.operation_count().to_string())
  println("candidate_root=" + arguments.absolute_new_output + "/candidate-root")
  println(
    "candidate_inventory=" +
    arguments.absolute_new_output +
    "/candidate.files.sha256#sha256=" +
    evidence.inventory_sha256(),
  )
  println("candidate_set_sha256=" + evidence.candidate_set_sha256())
  println("export_record_sha256=" + evidence.export_record_sha256())
  println("compiler_invoked=0")
  println("device_opened=0")
  println("runtime_authority=0")
  println("physical_readiness=0")
  println("reusable_fused_residual_source=" + fused_root + "/kernel.cu")
  println("reusable_fused_residual_recipe=" + fused_root + "/kernel.recipe")
  println(
    "reusable_fused_residual_candidate_sha256=" +
    production.candidate_digest().as_hex(),
  )
  println(
    "reusable_fused_residual_first_operation=" +
    residual.residual_fallback().operation_id().as_int().to_string(),
  )
  println(
    "reusable_fused_residual_function_symbol=" + production.function_symbol(),
  )
  let residual_launch = production.launch_dimensions()
  println(
    "reusable_fused_residual_launch=" +
    residual_launch.grid_x().to_string() +
    "," +
    residual_launch.grid_y().to_string() +
    "," +
    residual_launch.block_x().to_string() +
    "," +
    residual_launch.shared_memory_bytes().to_string(),
  )
  println("reusable_qwen_ingress_source=" + ingress_root + "/kernel.cu")
  println("reusable_qwen_ingress_recipe=" + ingress_root + "/kernel.recipe")
  println(
    "reusable_qwen_ingress_candidate_sha256=" +
    qwen_fused.postprocess.candidate_digest().as_hex(),
  )
  println(
    "reusable_qwen_ingress_first_operation=" +
    qwen_fused.first_operation.to_string(),
  )
  println(
    "reusable_qwen_ingress_function_symbol=" +
    qwen_fused.postprocess.function_symbol(),
  )
  let ingress_launch = qwen_fused.postprocess.launch_dimensions()
  println(
    "reusable_qwen_ingress_launch=" +
    ingress_launch.grid_x().to_string() +
    "," +
    ingress_launch.grid_y().to_string() +
    "," +
    ingress_launch.block_x().to_string() +
    "," +
    ingress_launch.shared_memory_bytes().to_string(),
  )
  println("reusable_qwen_readonly_source=" + readonly_root + "/kernel.cu")
  println("reusable_qwen_readonly_recipe=" + readonly_root + "/kernel.recipe")
  println(
    "reusable_qwen_readonly_candidate_sha256=" +
    qwen_fused.attention_read.candidate_digest().as_hex(),
  )
  println(
    "reusable_qwen_readonly_operation=" +
    (qwen_fused.first_operation + 3).to_string(),
  )
  println(
    "reusable_qwen_readonly_function_symbol=" +
    qwen_fused.attention_read.function_symbol(),
  )
  let readonly_launch = qwen_fused.attention_read.launch_dimensions()
  println(
    "reusable_qwen_readonly_launch=" +
    readonly_launch.grid_x().to_string() +
    "," +
    readonly_launch.grid_y().to_string() +
    "," +
    readonly_launch.block_x().to_string() +
    "," +
    readonly_launch.shared_memory_bytes().to_string(),
  )
  println("reusable_qwen_prefill_source=" + prefill_root + "/kernel.cu")
  println("reusable_qwen_prefill_recipe=" + prefill_root + "/kernel.recipe")
  println(
    "reusable_qwen_prefill_compilation_sha256=" +
    selected_prefill.digest().as_hex(),
  )
  println(
    "reusable_qwen_prefill_frontier_sha256=" +
    qwen_fused.attention_prefill_frontier.digest().as_hex(),
  )
  println(
    "reusable_qwen_prefill_frontier_variants=" +
    qwen_fused.attention_prefill_frontier.variants().length().to_string(),
  )
  for variant in qwen_fused.attention_prefill_frontier.variants() {
    let stable_id = variant.functional().plan().candidate().stable_id()
    println(
      "reusable_qwen_prefill_variant=" +
      stable_id.to_string() +
      "," +
      variant.source().symbol().to_owned() +
      "," +
      variant.digest().as_hex(),
    )
  }
  match wide_prefill {
    Some(variant) => {
      let (query_rows, key_value_rows, _) = variant
        .functional()
        .plan()
        .candidate()
        .tile()
        .dimensions()
      println(
        "reusable_qwen_prefill_wide_source=" + wide_prefill_root + "/kernel.cu",
      )
      println(
        "reusable_qwen_prefill_wide_recipe=" +
        wide_prefill_root +
        "/kernel.recipe",
      )
      println(
        "reusable_qwen_prefill_wide_tile=" +
        query_rows.to_string() +
        "," +
        key_value_rows.to_string(),
      )
    }
    None => println("reusable_qwen_prefill_wide_source=absent")
  }
  let (prefill_grid_x, prefill_grid_y, prefill_block_x, prefill_shared) = selected_prefill
    .lowering()
    .launch()
  println(
    "reusable_qwen_prefill_launch=" +
    prefill_grid_x.to_string() +
    "," +
    prefill_grid_y.to_string() +
    "," +
    prefill_block_x.to_string() +
    "," +
    prefill_shared.to_string(),
  )
  println("reusable_qwen_decode_source=" + decode_root + "/kernel.cu")
  println("reusable_qwen_decode_recipe=" + decode_root + "/kernel.recipe")
  println(
    "reusable_qwen_decode_compilation_sha256=" +
    qwen_fused.attention_decode.digest().as_hex(),
  )
  let (decode_grid_x, decode_grid_y, decode_block_x, decode_shared) = qwen_fused.attention_decode
    .lowering()
    .launch()
  println(
    "reusable_qwen_decode_launch=" +
    decode_grid_x.to_string() +
    "," +
    decode_grid_y.to_string() +
    "," +
    decode_block_x.to_string() +
    "," +
    decode_shared.to_string(),
  )
  println(
    "reusable_qwen_decode_split_source=" + decode_split_root + "/kernel.cu",
  )
  println(
    "reusable_qwen_decode_split_recipe=" + decode_split_root + "/kernel.recipe",
  )
  println(
    "reusable_qwen_decode_split_candidate_sha256=" +
    qwen_fused.attention_decode_split.source_digest().as_hex(),
  )
  let (partial_symbol, merge_symbol) = qwen_fused.attention_decode_split.function_symbols()
  println("reusable_qwen_decode_split_partial_symbol=" + partial_symbol)
  println("reusable_qwen_decode_split_merge_symbol=" + merge_symbol)
  let (partial_launch, merge_launch) = qwen_fused.attention_decode_split.launch_dimensions()
  println(
    "reusable_qwen_decode_split_launches=" +
    partial_launch.grid_x().to_string() +
    "," +
    partial_launch.grid_y().to_string() +
    "," +
    partial_launch.grid_z().to_string() +
    ";" +
    merge_launch.grid_x().to_string() +
    "," +
    merge_launch.grid_y().to_string() +
    "," +
    merge_launch.grid_z().to_string(),
  )
}

///|
async fn main {
  try run_qwen3_candidate_export(@env.args()) catch {
    error => {
      println("outcome=failed")
      println("error=" + candidate_export_error_name(error))
      abort("Qwen3 BF16 candidate export failed")
    }
  } noraise {
    _ => ()
  }
}
