///|
fn source_test_lowering(
  supports_async_copy : Bool,
  maximum_shared_memory_bytes : Int64,
  maximum_pipeline_stages : Int,
  candidate_id? : Int? = None,
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
  )
  let schedule = @luna_attention_tile_schedule.schedule_attention_tile_program(
    program,
  )
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
  candidate_id? : Int = 420,
) -> @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering raise {
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
    max_shared_memory_bytes=100000L,
    max_pipeline_stages=2,
    supports_async_copy=false,
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
  @luna_cuda_attention_tile_lowering.lower_attention_tile_schedule_to_cuda(
    program,
    @luna_attention_tile_optimizer.optimize_attention_tile_program(program),
    @luna_attention_tile_schedule.schedule_attention_tile_program(program),
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
test "compiler direct decode source keeps one subgroup per query head" {
  let lowering = source_test_decode_lowering(candidate_id=400)
  assert_eq(lowering.phase(), Decode)
  assert_eq(lowering.compute(), CudaScalarFma)
  assert_eq(lowering.copy(), CudaDirectGlobalLoad)
  assert_eq(lowering.launch(), (32, 16, 32, 0L))
  let source = @utf8.decode(
    emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "lunaflux_attention_decode_direct_v1",
      1048576,
    ).bytes(),
  )
  for
    required in [
      "const int local_row = (int)blockIdx.x", "query_head = (int)blockIdx.y", "const int key_value_head = query_head / 4",
      "lf_direct_warp_sum", "denominator = __shfl_sync(0xffffffffU, denominator, 0)",
    ] {
    assert_true(source.contains(required))
  }
  assert_false(source.contains("extern __shared__"))
  assert_false(source.contains("__syncthreads"))
}

///|
test "wide dual-matrix prefill source is deterministic and consumes the schedule" {
  let lowering = source_test_lowering(false, 100000L, 2)
  assert_eq(lowering.compute(), CudaMatrixQkPvMultiplyAccumulate)
  assert_eq(lowering.copy(), CudaCooperativeSharedLoad)
  assert_eq(lowering.tile(), (32, 32, 128))
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
      "#define LF_QUERY_TILE 32", "#define LF_KEY_VALUE_TILE 32", "wmma::mma_sync",
      "#define LF_PIPELINE_STAGES 1", "#define LF_SHARED_MEMORY_BYTES 47504", "AttentionTileFoldKeyValueStream",
      "running_denominator[query] * previous_scale + tile_sum", "tile_ordinal < tile_prefix + row_tiles",
      "tile_ordinal += (int)gridDim.x", "const __nv_bfloat16 *key_cache", "const __nv_bfloat16 *value_cache",
    ] {
    if required == "AttentionTileFoldKeyValueStream" {
      assert_false(source.contains(required))
    } else {
      assert_true(source.contains(required))
    }
  }
  assert_true(source.contains("row_count != prefill_count + decode_count"))
  assert_false(source.contains("decode_count != 0"))
}

///|
test "dual-matrix source lowers probability-value multiplication to MMA" {
  let lowering = source_test_lowering(false, 100000L, 2, candidate_id=Some(310))
  assert_eq(lowering.compute(), CudaMatrixQkPvMultiplyAccumulate)
  assert_eq(lowering.tile(), (16, 32, 128))
  assert_eq(lowering.launch(), (128, 16, 256, 31952L))
  let source = @utf8.decode(
    emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "lunaflux_attention_prefill_tile_v1",
      1048576,
    ).bytes(),
  )
  assert_true(source.contains("__nv_bfloat16 *weights"))
  assert_true(source.contains("__float2bfloat16_rn(weight)"))
  assert_true(source.contains("wmma::load_matrix_sync(probability_fragment"))
  assert_true(source.contains("wmma::load_matrix_sync(value_fragment"))
}

///|
test "wide-query prefill reuses one KV tile across two matrix row tiles" {
  let lowering = source_test_lowering(false, 100000L, 2, candidate_id=Some(313))
  assert_eq(lowering.compute(), CudaMatrixQkPvMultiplyAccumulate)
  assert_eq(lowering.tile(), (32, 64, 128))
  assert_eq(lowering.launch(), (128, 16, 256, 70032L))
  let source = @utf8.decode(
    emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "lunaflux_attention_prefill_wide_query_v1",
      1048576,
    ).bytes(),
  )
  assert_true(source.contains("#define LF_QUERY_TILE 32"))
  assert_true(
    source.contains(
      "for (int query_base = 0; query_base < LF_QUERY_TILE; query_base += 16)",
    ),
  )
  assert_true(
    source.contains("weights + query_base * LF_KEY_VALUE_TILE + reduction"),
  )
  assert_true(
    source.contains("scores + query_base * LF_KEY_VALUE_TILE + key_column"),
  )
}

///|
test "source emitter rejects an asynchronous schedule until its copy lowering exists" {
  let asynchronous = source_test_lowering(
    true,
    262144L,
    3,
    candidate_id=Some(301),
  )
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

///|
test "compiler decode source reuses one KV tile across grouped query heads" {
  let lowering = source_test_decode_lowering()
  assert_eq(lowering.phase(), Decode)
  assert_eq(lowering.compute(), CudaGroupedQuerySubgroupShuffleDot)
  assert_eq(lowering.copy(), CudaCooperativeSharedLoad)
  assert_eq(lowering.launch(), (32, 4, 128, 34076L))
  let first = emit_cuda_attention_tile_source(
    lowering,
    source_test_abi(),
    "lunaflux_attention_decode_tile_v1",
    1048576,
  )
  let repeated = emit_cuda_attention_tile_source(
    lowering,
    source_test_abi(),
    "lunaflux_attention_decode_tile_v1",
    1048576,
  )
  assert_eq(first.digest(), repeated.digest())
  assert_eq(first.bytes(), repeated.bytes())
  let source = @utf8.decode(first.bytes())
  for
    required in [
      "#define LF_QUERY_HEADS_PER_BLOCK 4", "key_value_head * LF_QUERY_HEADS_PER_BLOCK + warp",
      "for (int key = 0; key < active_keys; ++key)", "denominator > 0.0f ? accumulator[item] / denominator",
      "denominator = __shfl_sync(0xffffffffU, denominator, 0);", "const __nv_bfloat16 *key_cache",
      "const __nv_bfloat16 *value_cache",
    ] {
    assert_true(source.contains(required))
  }
  assert_false(source.contains("nvcuda::wmma"))
  assert_false(source.contains("partial_numerators"))
}

///|
test "compiler split decode source preserves GQA reuse and key parallelism" {
  let lowering = source_test_decode_lowering(candidate_id=430)
  assert_eq(lowering.phase(), Decode)
  assert_eq(lowering.compute(), CudaGroupedQuerySplitSubgroupShuffleDot)
  assert_eq(lowering.copy(), CudaCooperativeSharedLoad)
  assert_eq(lowering.launch(), (32, 4, 256, 34076L))
  let source = @utf8.decode(
    emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "lunaflux_attention_decode_split_v1",
      1048576,
    ).bytes(),
  )
  for
    required in [
      "#define LF_QUERY_HEADS_PER_BLOCK 4", "#define LF_SPLITS_PER_HEAD 2", "const int query_in_group = warp / LF_SPLITS_PER_HEAD",
      "for (int key = split; key < active_keys; key += LF_SPLITS_PER_HEAD)", "partial_numerators",
      "merged_denominator",
    ] {
    assert_true(source.contains(required))
  }
}
