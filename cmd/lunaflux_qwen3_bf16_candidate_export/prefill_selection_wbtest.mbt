///|
test "all exported prefill aliases select query ownership and current history reads" {
  let shape = @luna_attention_strategy.AttentionShape::new(
    query_heads=16,
    kv_heads=8,
    head_dimension=128,
    tokens_per_page=8,
  )
  let frontier = @luna_cuda_attention_tile_aot.compile_cuda_attention_tile_aot_frontier(
    @luna_cuda_attention_tile_aot.CudaAttentionTileAotRequest::new(
      backend=@luna_attention_strategy.AttentionBackendId::new("cuda"),
      target_id=@luna_attention_strategy.AttentionTargetId::new("sm120-fixture"),
      problem=@luna_attention_strategy.AttentionCompileProblem::new(
        shape~,
        phase=Prefill,
        numeric_type=AttentionBFloat16,
        query_rows=128,
        context_tokens=4096,
        causal=true,
        allow_alternative_softmax_reduction=true,
        key_value_read_view=DenseCurrentAndPagedHistory,
        required_fragment_ownership=Some(QueryOwnedFragments),
      ),
      capabilities=@luna_attention_strategy.AttentionCompilerCapabilities::new(
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
      ),
      device_target=@catalog.DeviceTarget::new(
        compute_major=12,
        compute_minor=0,
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
        page_stride_values=16384L,
        physical_page_capacity=8192L,
        maximum_query_rows=32,
        maximum_query_tokens=1024,
        maximum_page_entries=8192,
      ),
      symbol="prefill_selection",
      maximum_source_bytes=1048576,
    ),
    [],
  )
  let selected = frontier.selected()
  assert_eq(selected.functional().plan().candidate().stable_id(), 318)
  assert_true(selected.lowering().query_owned_fragments())
  assert_true(selected.lowering().reads_dense_current_key_value())
  assert_eq(selected.lowering().launch(), (128, 16, 128, 49168L))
  assert_eq(selected.lowering().compiler_register_ceiling(128), 255)
  assert_true(
    frontier.variants().all(v => v.lowering().query_owned_fragments()),
  )
  let source = @utf8.decode(selected.source().bytes())
  assert_false(source.contains("lf_matrix_store_float"))
  assert_false(source.contains("lf_float_index"))
  guard qwen_wide_query_prefill_variant(frontier) is Some(alias_) else {
    fail("missing profile-valid alias")
  }
  assert_eq(alias_.digest(), selected.digest())
  assert_eq(alias_.source().symbol(), selected.source().symbol())
}
