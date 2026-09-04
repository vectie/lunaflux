///|
fn cuda_lower_test_program_and_schedule(
  causal? : Bool = true,
  query_rows? : Int = 128,
) -> (
  @luna_attention_tile_ir.AttentionTileProgram,
  @luna_attention_tile_optimizer.AttentionTileOptimization,
  @luna_attention_tile_schedule.AttentionTileSchedule,
) raise {
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
    query_rows~,
    context_tokens=4096,
    causal~,
  )
  let capabilities = @luna_attention_strategy.AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=262144L,
    max_pipeline_stages=3,
    supports_async_copy=true,
    matrix_shape=Some(
      @luna_attention_strategy.AttentionMatrixShape::new(
        rows=16,
        columns=16,
        reduction=16,
      ),
    ),
  )
  let selected = @luna_attention_strategy.compile_attention_tile_plan(
    [],
    backend=@luna_attention_strategy.AttentionBackendId::new("cuda"),
    target=@luna_attention_strategy.AttentionTargetId::new("sm120-fixture"),
    problem~,
    capabilities~,
  )
  let program = @luna_attention_tile_ir.compile_attention_tile_ir(selected)
  (
    program,
    @luna_attention_tile_optimizer.optimize_attention_tile_program(program),
    @luna_attention_tile_schedule.schedule_attention_tile_program(program),
  )
}

///|
test "CUDA lowering is the only layer with NVIDIA execution vocabulary" {
  let (program, optimization, schedule) = cuda_lower_test_program_and_schedule()
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let first = lower_attention_tile_schedule_to_cuda(
    program, optimization, schedule, target, 262144L,
  )
  let repeated = lower_attention_tile_schedule_to_cuda(
    program, optimization, schedule, target, 262144L,
  )
  assert_eq(first.digest(), repeated.digest())
  assert_eq(first.canonical_bytes(), repeated.canonical_bytes())
  assert_eq(first.program_digest(), program.digest())
  assert_eq(first.optimization_digest(), optimization.digest())
  assert_eq(first.schedule_digest(), schedule.digest())
  assert_eq(first.compute(), CudaMatrixQkPvMultiplyAccumulate)
  assert_eq(first.copy(), CudaCooperativeSharedLoad)
  assert_eq(first.launch(), (128, 16, 256, 107280L))
  assert_eq(first.key_value_partitions(), 1)
  assert_false(first.hoists_paged_key_value_page_lookup())
  assert_eq(first.tile(), (64, 64, 128))
  assert_eq(first.pipeline(), (1, 16))
  assert_eq(first.digest().as_hex().length(), 64)
}

///|
test "CUDA lowering preserves functional KV partition topology" {
  let (program, optimization, _) = cuda_lower_test_program_and_schedule(
    query_rows=64,
  )
  let schedule = @luna_attention_tile_schedule.schedule_optimized_attention_tile_program(
    program, optimization,
  )
  let lowering = lower_attention_tile_schedule_to_cuda(
    program,
    optimization,
    schedule,
    @catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    262144L,
  )
  assert_eq(lowering.key_value_partitions(), 2)
  assert_true(lowering.hoists_paged_key_value_page_lookup())
  assert_true(lowering.reuses_online_softmax_tile_storage())
  assert_true(
    @utf8.decode(lowering.canonical_bytes()).contains(
      "launch=64,16,2,256,98304",
    ),
  )
}

///|
test "CUDA lowering rejects target shared-memory overcommit" {
  let (program, optimization, schedule) = cuda_lower_test_program_and_schedule()
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  try
    lower_attention_tile_schedule_to_cuda(
      program, optimization, schedule, target, 32768L,
    )
  catch {
    InvalidCudaAttentionTileLowering(SharedMemory) => ()
    _ => fail("unexpected shared-memory rejection")
  } noraise {
    _ => fail("shared-memory overcommit was accepted")
  }
}

///|
test "CUDA lowering rejects optimization from another semantic program" {
  let (program, _, schedule) = cuda_lower_test_program_and_schedule()
  let (_, noncausal_optimization, _) = cuda_lower_test_program_and_schedule(
    causal=false,
  )
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  try
    lower_attention_tile_schedule_to_cuda(
      program, noncausal_optimization, schedule, target, 262144L,
    )
  catch {
    InvalidCudaAttentionTileLowering(InvalidOptimization) => ()
    _ => fail("unexpected optimization mismatch rejection")
  } noraise {
    _ => fail("mismatched functional optimization was accepted")
  }
}

///|
test "decode CUDA lowering splits each grouped query head across subgroups" {
  let shape = @luna_attention_strategy.AttentionShape::new(
    query_heads=16,
    kv_heads=4,
    head_dimension=128,
    tokens_per_page=16,
  )
  let problem = @luna_attention_strategy.AttentionCompileProblem::new(
    shape~,
    phase=Decode,
    numeric_type=AttentionBFloat16,
    batch_rows=32,
    query_rows=1,
    context_tokens=4096,
    causal=true,
  )
  let capabilities = @luna_attention_strategy.AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=262144L,
    max_pipeline_stages=2,
    supports_async_copy=false,
  )
  let selected = @luna_attention_strategy.compile_attention_tile_plan(
    [],
    backend=@luna_attention_strategy.AttentionBackendId::new("cuda"),
    target=@luna_attention_strategy.AttentionTargetId::new("sm120-fixture"),
    problem~,
    capabilities~,
  )
  let program = @luna_attention_tile_ir.compile_attention_tile_ir(selected)
  let lowering = lower_attention_tile_schedule_to_cuda(
    program,
    @luna_attention_tile_optimizer.optimize_attention_tile_program(program),
    @luna_attention_tile_schedule.schedule_attention_tile_program(program),
    @catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    262144L,
  )
  assert_eq(lowering.phase(), Decode)
  assert_eq(lowering.launch(), (32, 4, 256, 34076L))
  assert_eq(lowering.compute(), CudaGroupedQuerySplitSubgroupShuffleDot)
  assert_eq(lowering.copy(), CudaCooperativeSharedLoad)
}
