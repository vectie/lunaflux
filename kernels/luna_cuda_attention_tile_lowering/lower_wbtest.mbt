///|
fn cuda_lower_test_program_and_schedule() -> (
  @luna_attention_tile_ir.AttentionTileProgram,
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
    query_rows=128,
    context_tokens=4096,
    causal=true,
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
    @luna_attention_tile_schedule.schedule_attention_tile_program(program),
  )
}

///|
test "CUDA lowering is the only layer with NVIDIA execution vocabulary" {
  let (program, schedule) = cuda_lower_test_program_and_schedule()
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let first = lower_attention_tile_schedule_to_cuda(
    program, schedule, target, 262144L,
  )
  let repeated = lower_attention_tile_schedule_to_cuda(
    program, schedule, target, 262144L,
  )
  assert_eq(first.digest(), repeated.digest())
  assert_eq(first.canonical_bytes(), repeated.canonical_bytes())
  assert_eq(first.program_digest(), program.digest())
  assert_eq(first.schedule_digest(), schedule.digest())
  assert_eq(first.compute(), CudaMatrixMultiplyAccumulate)
  assert_eq(first.copy(), CudaAsynchronousSharedLoad)
  assert_eq(first.launch(), (8, 16, 256, 208896L))
  assert_eq(first.tile(), (16, 128, 128))
  assert_eq(first.pipeline(), (3, 16))
  assert_eq(first.digest().as_hex().length(), 64)
}

///|
test "CUDA lowering rejects target shared-memory overcommit" {
  let (program, schedule) = cuda_lower_test_program_and_schedule()
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  try
    lower_attention_tile_schedule_to_cuda(program, schedule, target, 49152L)
  catch {
    InvalidCudaAttentionTileLowering(SharedMemory) => ()
    _ => fail("unexpected shared-memory rejection")
  } noraise {
    _ => fail("shared-memory overcommit was accepted")
  }
}
