///|
fn head256_request(
  phase : @luna_attention_strategy.AttentionExecutionPhase,
) -> CudaAttentionTileAotRequest raise {
  let shape = @luna_attention_strategy.AttentionShape::new(
    query_heads=16,
    kv_heads=2,
    head_dimension=256,
    tokens_per_page=16,
  )
  let capabilities = @luna_attention_strategy.AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=49152L,
    max_pipeline_stages=1,
    parallel_workgroup_target=1,
    supports_async_copy=false,
    matrix_shape=Some(
      @luna_attention_strategy.AttentionMatrixShape::new(
        rows=16,
        columns=16,
        reduction=16,
      ),
    ),
  )
  CudaAttentionTileAotRequest::new(
    backend=@luna_attention_strategy.AttentionBackendId::new("cuda"),
    target_id=@luna_attention_strategy.AttentionTargetId::new(
      "sm121-unqualified-software-profile",
    ),
    problem=@luna_attention_strategy.AttentionCompileProblem::new(
      shape~,
      phase~,
      numeric_type=AttentionBFloat16,
      batch_rows=1,
      query_rows=if phase is Decode { 1 } else { 32 },
      context_tokens=4096,
      causal=true,
    ),
    capabilities~,
    device_target=@catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=1,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    maximum_shared_memory_bytes=49152L,
    abi=@luna_cuda_attention_tile_source.CudaPagedAttentionTileAbi::new(
      query_heads=16,
      key_value_heads=2,
      head_dimension=256,
      tokens_per_page=16,
      input_row_width=5120,
      page_stride_values=8192L,
      physical_page_capacity=256L,
      maximum_query_rows=1,
      maximum_query_tokens=32,
      maximum_page_entries=256,
    ),
    symbol="head256_test",
    maximum_source_bytes=2097152,
  )
}

///|
test "256-wide GQA chooses an emitted legal schedule when a static winner lacks codegen" {
  for
    phase in [@luna_attention_strategy.AttentionExecutionPhase::Prefill, Decode] {
    let request = head256_request(phase)
    let result = compile_cuda_attention_tile_aot(request, [])
    assert_true(
      @utf8.decode(result.source().bytes()[:]).contains(
        "#define LF_HEAD_DIMENSION 256",
      ),
    )
    let frontier = compile_cuda_attention_tile_aot_frontier(request, [])
    assert_eq(
      frontier.functional().selected(),
      frontier.selected().functional(),
    )
    assert_eq(frontier.selected(), result)
    if phase is Prefill {
      let functional = @luna_attention_tile_compiler.compile_attention_tile(
        request.functional_request([]),
      )
      assert_eq(functional.plan().candidate().stable_id(), 200)
      assert_true(result.functional().plan().candidate().stable_id() != 200)
      assert_true(
        frontier
        .rejected_candidates()
        .any(entry => entry.0 == 200 && entry.1 == SourceEmission),
      )
    }
  }
}

///|
test "unsupported explicit and measured selections remain errors" {
  let request = head256_request(Prefill)
  assert_true(
    try
      compile_cuda_attention_tile_aot_frontier(
        request,
        [],
        explicit_candidate=200,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  let functional = @luna_attention_tile_compiler.compile_attention_tile(
    request.functional_request([]),
  )
  // Synthetic negative-test record, never exported as hardware evidence.
  let record = @luna_attention_strategy.AttentionAutotuneRecord::new(
    key=@luna_attention_strategy.AttentionAutotuneKey::new(
      backend=request.backend,
      target=request.target_id,
      problem=request.problem,
      candidate=functional.plan().candidate(),
    ),
    median_latency_ns=1L,
    sample_count=3,
  )
  assert_true(
    try compile_cuda_attention_tile_aot(request, [record]) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}
