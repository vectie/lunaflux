///|
fn source_test_lowering(
  supports_async_copy : Bool,
  maximum_shared_memory_bytes : Int64,
  maximum_pipeline_stages : Int,
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
  let selected = @luna_attention_strategy.compile_attention_tile_plan(
    [],
    backend=@luna_attention_strategy.AttentionBackendId::new("cuda"),
    target=@luna_attention_strategy.AttentionTargetId::new("sm120-fixture"),
    problem~,
    capabilities~,
  )
  let program = @luna_attention_tile_ir.compile_attention_tile_ir(selected)
  let schedule = @luna_attention_tile_schedule.schedule_attention_tile_program(
    program,
  )
  @luna_cuda_attention_tile_lowering.lower_attention_tile_schedule_to_cuda(
    program,
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
test "matrix-tiled prefill source is deterministic and consumes the schedule" {
  let lowering = source_test_lowering(false, 100000L, 2)
  assert_eq(lowering.compute(), CudaMatrixMultiplyAccumulate)
  assert_eq(lowering.copy(), CudaCooperativeSharedLoad)
  assert_eq(lowering.tile(), (16, 64, 128))
  let first = emit_cuda_attention_tile_source(
    lowering,
    source_test_abi(),
    "lunaflux_attention_prefill_tile_v1",
    1048576,
  )
  let repeated = emit_cuda_attention_tile_source(
    lowering,
    source_test_abi(),
    "lunaflux_attention_prefill_tile_v1",
    1048576,
  )
  assert_eq(first.digest(), repeated.digest())
  assert_eq(first.bytes(), repeated.bytes())
  assert_eq(first.lowering_digest(), lowering.digest())
  assert_eq(first.digest().as_hex().length(), 64)
  let source = @utf8.decode(first.bytes())
  for
    required in [
      "#define LF_QUERY_TILE 16", "#define LF_KEY_VALUE_TILE 64", "wmma::mma_sync",
      "AttentionTileFoldKeyValueStream", "running_denominator[query] * previous_scale + tile_sum",
      "tile_ordinal < tile_prefix + row_tiles", "const __nv_bfloat16 *key_cache",
      "const __nv_bfloat16 *value_cache",
    ] {
    if required == "AttentionTileFoldKeyValueStream" {
      assert_false(source.contains(required))
    } else {
      assert_true(source.contains(required))
    }
  }
}

///|
test "source emitter rejects an asynchronous schedule until its copy lowering exists" {
  let asynchronous = source_test_lowering(true, 262144L, 3)
  assert_eq(asynchronous.copy(), CudaAsynchronousSharedLoad)
  try
    emit_cuda_attention_tile_source(
      asynchronous,
      source_test_abi(),
      "lunaflux_attention_prefill_tile_v1",
      1048576,
    )
  catch {
    InvalidCudaAttentionTileSource(UnsupportedLowering) => ()
    _ => fail("unexpected asynchronous lowering rejection")
  } noraise {
    _ => fail("unsupported asynchronous source was emitted")
  }
}
