///|
fn qwen_partitioned_prefill_problem(
  maximum_problem : @luna_attention_strategy.AttentionCompileProblem,
  query_tile : Int,
) -> @luna_attention_strategy.AttentionCompileProblem raise Qwen3Bf16CandidateExportCliError {
  if maximum_problem.phase() != Prefill || query_tile <= 0 {
    raise QwenReadOnlyPreparation
  }
  maximum_problem.with_query_rows(query_tile) catch {
    _ => raise QwenReadOnlyPreparation
  }
}

///|
fn prepare_qwen_compiler_partitioned_prefill(
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  baseline : @luna_cuda_attention_tile_aot.CudaAttentionTileAotCompilation,
  resources? : (Bytes, String, String)? = None,
) -> @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotFrontier raise Qwen3Bf16CandidateExportCliError {
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
    supports_async_copy=set.target().compute_major() >= 8,
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
  let request = @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotRequest::new(
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
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
  let frontier = @luna_cuda_attention_tile_aot.compile_cuda_partitioned_attention_tile_aot_frontier(
    request,
    [],
    [32, 64, 128],
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
  if resources is None {
    return frontier
  }
  @luna_cuda_attention_tile_aot.compile_cuda_partitioned_attention_tile_aot_frontier(
    qwen_partitioned_resource_request(request, frontier, resources),
    [],
    [32, 64, 128],
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
}

///|
fn qwen_partitioned_prefill_variant_root(
  output : String,
  compilation : @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotCompilation,
  index : Int,
) -> String {
  let base = output + "/reusable-qwen-prefill-partitioned-attention"
  if index == 0 {
    base
  } else {
    base +
    "-p" +
    compilation.functional().schedule().key_value_partitions().to_string()
  }
}

///|
fn prepare_qwen_compiler_prefill(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  qkv_index : Int,
  attention_index : Int,
  tuning? : (Bytes, String, String)? = None,
  resources? : (Bytes, String, String)? = None,
  query_metadata? : Bool = false,
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
    allow_alternative_softmax_reduction=true,
    key_value_read_view=DenseCurrentAndPagedHistory,
    required_fragment_ownership=Some(QueryOwnedFragments),
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
  let capabilities = @luna_attention_strategy.AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=100000L,
    max_pipeline_stages=2,
    supports_async_copy=set.target().compute_major() >= 8,
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
  let request = @luna_cuda_attention_tile_aot.CudaAttentionTileAotRequest::new(
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
      query_metadata~,
    ),
    symbol="lunaflux_attention_prefill_tile_compiler_v1",
    maximum_source_bytes=1048576,
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
  let frontier = @luna_cuda_attention_tile_aot.compile_cuda_attention_tile_aot_frontier(
    request, records,
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
  let request = qwen_attention_resource_request(request, frontier, resources)
  let resource_frontier = if resources is None {
    frontier
  } else {
    @luna_cuda_attention_tile_aot.compile_cuda_attention_tile_aot_frontier(
      request,
      records,
      previous=frontier,
    ) catch {
      _ => raise QwenReadOnlyPreparation
    }
  }
  guard tuning is Some((bytes, device, toolchain)) else {
    return resource_frontier
  }
  let observations = @luna_attention_tuning.parse_attention_tuning_records(
    bytes,
    scope=frontier.digest().as_hex() + "\t" + device + "\t" + toolchain,
    backend~,
    target~,
    problem~,
    resource_frontier.variants().map(v => v.functional().plan().candidate())[:],
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
  @luna_cuda_attention_tile_aot.compile_cuda_attention_tile_aot_frontier(
    request,
    observations[:],
    previous=resource_frontier,
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
  resources? : (Bytes, String, String)? = None,
  tuning? : (Bytes, String, String)? = None,
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
  let capabilities = @luna_cuda_attention_autotune.cuda_sm120_decode_compiler_capabilities() catch {
    _ => raise QwenReadOnlyPreparation
  }
  let backend = @luna_cuda_attention_autotune.cuda_sm120_rtx5060ti_backend() catch {
    _ => raise QwenReadOnlyPreparation
  }
  let target = @luna_cuda_attention_autotune.cuda_sm120_rtx5060ti_target() catch {
    _ => raise QwenReadOnlyPreparation
  }
  let records : Array[@luna_attention_strategy.AttentionAutotuneRecord] = []
  let request = @luna_cuda_attention_tile_aot.CudaAttentionTileAotRequest::new(
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
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
  let frontier = @luna_cuda_attention_tile_aot.compile_cuda_attention_tile_aot_frontier(
    request, records,
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
  let request = qwen_attention_resource_request(request, frontier, resources)
  let expanded = if resources is None {
    frontier
  } else {
    @luna_cuda_attention_tile_aot.compile_cuda_attention_tile_aot_frontier(
      request,
      records,
      previous=frontier,
    ) catch {
      _ => raise QwenReadOnlyPreparation
    }
  }
  guard tuning is Some((bytes, device, toolchain)) else { return expanded }
  let observations = @luna_attention_tuning.parse_attention_tuning_records(
    bytes,
    scope=frontier.digest().as_hex() + "\t" + device + "\t" + toolchain,
    backend~,
    target~,
    problem~,
    expanded.variants().map(v => v.functional().plan().candidate())[:],
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
  @luna_cuda_attention_tile_aot.compile_cuda_attention_tile_aot_frontier(
    request,
    observations[:],
    previous=expanded,
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
}
