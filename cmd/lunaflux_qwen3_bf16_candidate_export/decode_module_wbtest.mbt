///|
test "partition helpers cannot collide through global BF16 argument lookup" {
  assert_eq(
    qwen_decode_partition_helpers(
      "lf_matrix_load(p,0,0,128); lf_matrix_load (p); __half_as_ushort(x); my_lf_value; lunaflux_decode; lf_last",
    ),
    "lf_partition_matrix_load(p,0,0,128); lf_partition_matrix_load (p); __half_as_ushort(x); my_lf_value; lunaflux_decode; lf_partition_last",
  )
}

///|
test "decode module resets object and function macro identities" {
  assert_eq(
    qwen_decode_module_macro_reset(
      "#define LF_HEAD_DIMENSION 128\n#define LF_SCORE_COMPONENT(item) (item * 16)\n#define unrelated 1\n",
    ),
    "#undef LF_HEAD_DIMENSION\n#undef LF_SCORE_COMPONENT",
  )
}

///|
test "decode recipe law comes from executable owned and matrix program" {
  let shape = @luna_attention_strategy.AttentionShape::new(
    query_heads=16,
    kv_heads=8,
    head_dimension=128,
    tokens_per_page=8,
  )
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(9),
    max_query_rows=32,
    max_query_tokens=2048,
    max_page_table_entries=16384,
  )
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=1,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  for
    (id, law) in [
      (460, "dual-score-blockwise-f32-probability-v2"),
      (466, "owned4-blockwise-fma-f32-probability-v4"),
      (483, "grouped-head-matrix-bf16-probability-v1"),
    ] {
    let compiled = @luna_cuda_paged_attention_aot.compile_blockwise_decode_split(
      shape,
      profile,
      target,
      16384L,
      16384L,
      4096,
      8,
      16777216L,
      candidate_id=id,
    )
    assert_eq(qwen_decode_physical_numeric_law(compiled.lowering()), Some(law))
  }
}

///|
test "selected serving alias preserves candidate qualified frontier identity" {
  for
    (id, suffix) in [
      (460, "dual_score_blockwise_f32_v2"),
      (466, "owned4_blockwise_fma_f32_v4"),
      (480, "grouped_head_matrix_bf16_v1"),
      (482, "grouped_head_matrix_bf16_v1"),
    ] {
    let qualified = "lunaflux_attention_decode_tile_compiler_v1_c\{id}"
    let request = @luna_cuda_attention_tile_aot.CudaAttentionTileAotRequest::new(
      backend=@luna_attention_strategy.AttentionBackendId::new("cuda"),
      target_id=@luna_attention_strategy.AttentionTargetId::new(
        "sm121-alias-fixture",
      ),
      problem=@luna_attention_strategy.AttentionCompileProblem::new(
        shape=@luna_attention_strategy.AttentionShape::new(
          query_heads=16,
          kv_heads=8,
          head_dimension=128,
          tokens_per_page=8,
        ),
        phase=Decode,
        numeric_type=AttentionBFloat16,
        batch_rows=32,
        query_rows=1,
        context_tokens=4096,
        causal=true,
        allow_alternative_softmax_reduction=true,
      ),
      capabilities=@luna_cuda_attention_autotune.cuda_sm120_decode_compiler_capabilities(),
      device_target=@catalog.DeviceTarget::new(
        compute_major=12,
        compute_minor=1,
        supports_bf16=true,
        supports_cublas_lt=false,
      ),
      maximum_shared_memory_bytes=100000L,
      abi=@luna_cuda_attention_tile_source.CudaPagedAttentionTileAbi::new(
        query_heads=16,
        key_value_heads=8,
        head_dimension=128,
        tokens_per_page=8,
        input_row_width=4096,
        page_stride_values=8192L,
        physical_page_capacity=16L,
        maximum_query_rows=32,
        maximum_query_tokens=128,
        maximum_page_entries=256,
      ),
      symbol=qualified,
      maximum_source_bytes=1048576,
    )
    let compiled = @luna_cuda_attention_tile_aot.compile_cuda_attention_tile_aot_frontier(
      request,
      [],
      explicit_candidate=id,
    ).selected()
    let old_digest = compiled.source().digest()
    let serving_source = qwen_decode_selected_source(compiled)
    assert_eq(compiled.source().symbol(), qualified + "_" + suffix)
    assert_eq(
      serving_source.symbol(),
      "lunaflux_attention_decode_tile_compiler_v1_" + suffix,
    )
    assert_eq(compiled.source().digest(), old_digest)
    assert_true(serving_source.digest() != old_digest)
    assert_eq(serving_source.lowering_digest(), compiled.lowering().digest())
    assert_eq(serving_source.abi(), compiled.source().abi())
    assert_false(@utf8.decode(serving_source.bytes()).contains("_c\{id}_"))
  }
}
