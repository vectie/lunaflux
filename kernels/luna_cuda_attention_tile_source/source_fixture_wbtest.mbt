///|
fn source_test_lowering(
  supports_async_copy : Bool,
  maximum_shared_memory_bytes : Int64,
  maximum_pipeline_stages : Int32,
  candidate_id~ : Int32? = None,
  dense_current~ : Bool = false,
  affine_current~ : Bool = false,
  partitioned~ : Bool = false,
  approximate_exponential~ : Bool = false,
) -> @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering raise {
  let shape = @luna_attention_strategy.AttentionShape::new(
    query_heads=16,
    kv_heads=4,
    head_dimension=128,
    tokens_per_page=16,
  )
  let problem = @luna_attention_strategy.AttentionCompileProblem::new(
    shape~,
    phase=Prefill,
    numeric_type=AttentionBFloat16,
    query_rows=128,
    context_tokens=4096,
    causal=true,
    allow_alternative_softmax_reduction=candidate_id is Some(_),
    allow_approximate_exponential=approximate_exponential,
    key_value_read_view=if affine_current {
      AffineDenseCurrentAndPagedHistory
    } else if dense_current {
      DenseCurrentAndPagedHistory
    } else {
      PagedCacheOnly
    },
  )
  let capabilities = @luna_attention_strategy.AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=maximum_shared_memory_bytes,
    max_pipeline_stages=maximum_pipeline_stages,
    supports_async_copy~,
    matrix_shape=Some(
      @luna_attention_strategy.AttentionMatrixShape::new(
        rows=16,
        columns=16,
        reduction=16,
      ),
    ),
  )
  let backend = @luna_attention_strategy.AttentionBackendId::new("cuda")
  let target = @luna_attention_strategy.AttentionTargetId::new("sm120-fixture")
  let records = match candidate_id {
    None => []
    Some(stable_id) => {
      guard @luna_attention_strategy.generate_attention_tile_candidates(
          problem, capabilities,
        ).filter(candidate => candidate.stable_id() == stable_id)
        is [candidate] else {
        raise Failure("source-test candidate unavailable")
      }
      [
        @luna_attention_strategy.AttentionAutotuneRecord::new(
          key=@luna_attention_strategy.AttentionAutotuneKey::new(
            backend~,
            target~,
            problem~,
            candidate~,
          ),
          median_latency_ns=1L,
          sample_count=3,
        ),
      ]
    }
  }
  let selected = @luna_attention_strategy.compile_attention_tile_plan(
    records,
    backend~,
    target~,
    problem~,
    capabilities~,
  )
  let program = @luna_attention_tile_ir.compile_attention_tile_ir(selected)
  let optimization = @luna_attention_tile_optimizer.optimize_attention_tile_program(
    program,
    parallel_workgroup_target=if partitioned { 256 } else { 32 },
  )
  let schedule = if partitioned {
    @luna_attention_tile_schedule.schedule_optimized_attention_tile_program(
      program, optimization,
    )
  } else {
    @luna_attention_tile_schedule.schedule_rewritten_attention_tile_program(
      program, optimization,
    )
  }
  @luna_cuda_attention_tile_lowering.lower_attention_tile_schedule_to_cuda(
    program,
    optimization,
    schedule,
    @catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    maximum_shared_memory_bytes,
  )
}

///|
fn source_test_abi() -> CudaPagedAttentionTileAbi raise {
  CudaPagedAttentionTileAbi::new(
    query_heads=16,
    key_value_heads=4,
    head_dimension=128,
    tokens_per_page=16,
    input_row_width=3072,
    page_stride_values=8192L,
    physical_page_capacity=256L,
    maximum_query_rows=32,
    maximum_query_tokens=128,
    maximum_page_entries=256,
  )
}

///|
fn source_test_decode_lowering(
  candidate_id~ : Int32 = 420,
  batch_rows~ : Int32 = 32,
  context_tokens~ : Int32 = 4096,
  alternative~ : Bool = false,
  parallel_workgroup_target~ : Int32 = 32,
  key_value_heads~ : Int32 = 4,
  tokens_per_page~ : Int32 = 16,
) -> @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering raise {
  let shape = @luna_attention_strategy.AttentionShape::new(
    query_heads=16,
    kv_heads=key_value_heads,
    head_dimension=128,
    tokens_per_page~,
  )
  let problem = @luna_attention_strategy.AttentionCompileProblem::new(
    shape~,
    phase=Decode,
    numeric_type=AttentionBFloat16,
    batch_rows~,
    query_rows=1,
    context_tokens~,
    causal=true,
    allow_alternative_softmax_reduction=alternative,
  )
  let capabilities = @luna_attention_strategy.AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=100000L,
    max_pipeline_stages=2,
    matrix_shape=Some(
      @luna_attention_strategy.AttentionMatrixShape::new(
        rows=16,
        columns=16,
        reduction=16,
      ),
    ),
    supports_async_copy=[
      440, 441, 452, 453, 454, 455, 456, 457, 460, 461, 462, 463, 464, 465, 466,
      467, 468, 469, 470, 471, 482, 483,
    ].contains(candidate_id),
    parallel_workgroup_target~,
  )
  guard @luna_attention_strategy.generate_attention_tile_candidates(
      problem, capabilities,
    ).filter(candidate => candidate.stable_id() == candidate_id)
    is [candidate] else {
    raise Failure("decode source-test candidate unavailable")
  }
  let records = [
    @luna_attention_strategy.AttentionAutotuneRecord::new(
      key=@luna_attention_strategy.AttentionAutotuneKey::new(
        backend=@luna_attention_strategy.AttentionBackendId::new("cuda"),
        target=@luna_attention_strategy.AttentionTargetId::new("sm120-fixture"),
        problem~,
        candidate~,
      ),
      median_latency_ns=1L,
      sample_count=3,
    ),
  ]
  let selected = @luna_attention_strategy.compile_attention_tile_plan(
    records,
    backend=@luna_attention_strategy.AttentionBackendId::new("cuda"),
    target=@luna_attention_strategy.AttentionTargetId::new("sm120-fixture"),
    problem~,
    capabilities~,
  )
  let program = @luna_attention_tile_ir.compile_attention_tile_ir(selected)
  let optimized = @luna_attention_tile_optimizer.optimize_attention_tile_program(
    program,
    parallel_workgroup_target~,
  )
  @luna_cuda_attention_tile_lowering.lower_attention_tile_schedule_to_cuda(
    program,
    optimized,
    @luna_attention_tile_schedule.schedule_optimized_attention_tile_program(
      program, optimized,
    ),
    @catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    100000L,
  )
}

///|
fn source_test_partitioned_lowering() -> @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering raise {
  let shape = @luna_attention_strategy.AttentionShape::new(
    query_heads=16,
    kv_heads=4,
    head_dimension=128,
    tokens_per_page=16,
  )
  let problem = @luna_attention_strategy.AttentionCompileProblem::new(
    shape~,
    phase=Prefill,
    numeric_type=AttentionBFloat16,
    query_rows=64,
    context_tokens=4096,
    causal=true,
  )
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
  )
  let plan = @luna_attention_strategy.compile_attention_tile_plan(
    [],
    backend=@luna_attention_strategy.AttentionBackendId::new("cuda"),
    target=@luna_attention_strategy.AttentionTargetId::new("sm120-fixture"),
    problem~,
    capabilities~,
  )
  let program = @luna_attention_tile_ir.compile_attention_tile_ir(plan)
  let optimization = @luna_attention_tile_optimizer.optimize_attention_tile_program(
    program,
  )
  @luna_cuda_attention_tile_lowering.lower_attention_tile_schedule_to_cuda(
    program,
    optimization,
    @luna_attention_tile_schedule.schedule_optimized_attention_tile_program(
      program, optimization,
    ),
    @catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    100000L,
  )
}
