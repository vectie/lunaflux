///|
/// Diagnostic export uses the production Q/KV geometry and bucket ceilings;
/// synthetic timing observations choose a fixture, never enter a release.
fn repair_prefill_lifetime_lowering(
  candidate_id : Int,
  affine_current? : Bool = false,
) -> @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering raise {
  let shape = @luna_attention_strategy.AttentionShape::new(
    query_heads=16,
    kv_heads=8,
    head_dimension=128,
    tokens_per_page=8,
  )
  let problem = @luna_attention_strategy.AttentionCompileProblem::new(
    shape~,
    phase=Prefill,
    numeric_type=AttentionBFloat16,
    batch_rows=32,
    query_rows=2048,
    context_tokens=8192,
    causal=true,
    allow_alternative_softmax_reduction=true,
    key_value_read_view=if affine_current {
      AffineDenseCurrentAndPagedHistory
    } else {
      DenseCurrentAndPagedHistory
    },
  )
  let capabilities = @luna_attention_strategy.AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=100000L,
    max_pipeline_stages=2,
    supports_async_copy=true,
    matrix_shape=Some(
      @luna_attention_strategy.AttentionMatrixShape::new(
        rows=16,
        columns=16,
        reduction=16,
      ),
    ),
  )
  guard @luna_attention_strategy.generate_attention_tile_candidates(
      problem, capabilities,
    ).filter(candidate => candidate.stable_id() == candidate_id)
    is [candidate] else {
    raise Failure("prefill lifetime candidate absent")
  }
  let backend = @luna_attention_strategy.AttentionBackendId::new("cuda")
  let target = @luna_attention_strategy.AttentionTargetId::new("sm121-fixture")
  let record = @luna_attention_strategy.AttentionAutotuneRecord::new(
    key=@luna_attention_strategy.AttentionAutotuneKey::new(
      backend~,
      target~,
      problem~,
      candidate~,
    ),
    median_latency_ns=1L,
    sample_count=3,
  )
  let selected = @luna_attention_strategy.compile_attention_tile_plan(
    [record],
    backend~,
    target~,
    problem~,
    capabilities~,
  )
  let program = @luna_attention_tile_ir.compile_attention_tile_ir(selected)
  let optimization = @luna_attention_tile_optimizer.optimize_attention_tile_program(
    program,
  )
  @luna_cuda_attention_tile_lowering.lower_attention_tile_schedule_to_cuda(
    program,
    optimization,
    @luna_attention_tile_schedule.schedule_rewritten_attention_tile_program(
      program, optimization,
    ),
    @catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=1,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    100000L,
  )
}

///|
fn repair_export_retained_prefill() -> Unit raise {
  let abi = CudaPagedAttentionTileAbi::new(
    query_heads=16,
    key_value_heads=8,
    head_dimension=128,
    tokens_per_page=8,
    input_row_width=4096,
    page_stride_values=8192L,
    physical_page_capacity=16384L,
    maximum_query_rows=32,
    maximum_query_tokens=2048,
    maximum_page_entries=16384,
  )
  for id in [322, 10644, 10645, 20644, 20645] {
    let lowering = repair_prefill_lifetime_lowering(id)
    let source = emit_cuda_attention_tile_source(
      lowering, abi, "lunaflux_attention_prefill_tile_v1", 1048576,
    )
    println("LUNA_SOURCE_BEGIN prefill-lifetime-\{id}")
    println(@utf8.decode(source.bytes()))
    println("LUNA_SOURCE_END")
  }
}
