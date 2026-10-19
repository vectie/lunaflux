///|
test "latent decoder removes expanded projections and binds new numerical boundaries" {
  let base = @precision.IndexedSparseAttentionPrecision::new(
    index_head_count=16,
    index_head_dimension=32,
    query_head_count=4,
    key_value_head_count=2,
    query_key_dimension=32,
    value_dimension=32,
    pool_size=4,
    top_k=32,
    append_incomplete_tail=true,
    maximum_queries=17,
    maximum_history=64,
    maximum_workspace_bytes=100000L,
    attention_accumulation=TiledFloat32,
  )
  let projection = @precision.IndexedSparseProjectionPrecision::new(
    base,
    hidden_width=64,
    query_rank=32,
    key_value_rank=64,
    rms_epsilon=1.0e-5,
    index_layer_norm_epsilon=1.0e-5,
    maximum_workspace_bytes=1000000L,
    cache_law=LatentBFloat16,
  )
  let envelope = @precision.HyperConnectionPrecision::new(
    stream_count=4,
    hidden_width=64,
    maximum_rows=17,
    sinkhorn_iterations=20,
    control_epsilon=1.0e-6,
    normalization_epsilon=1.0e-5,
    mix_law=RoundedBf16ControlsAndProducts,
  )
  let source = @indexed_sparse_projection_cuda_source.indexed_sparse_decoder_source(
    @precision.IndexedSparseDecoderPrecision::new(
      projection,
      envelope,
      maximum_state_bytes=1000000L,
      maximum_workspace_bytes=10000000L,
    ),
    symbol_prefix="latent",
  )
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("void latent_projection_q_absorb("))
  assert_true(!text.contains("void latent_projection_kv_b("))
  assert_true(!text.contains("void latent_projection_kv_split("))
  assert_true(text.contains("rsqrtf(32.0f)"))
  assert_true(text.contains("if(row_base>=live)"))
  assert_eq(source.expansion(), Some(("latent_value_expand", (8, 128))))
  if @env.get_env_var("LUNA_LATENT_EXPORT") is Some("1") {
    println("LUNA_LATENT_BEGIN")
    println(text)
    println(
      "static constexpr int absorb_grid=\{source.projection().launch(12).0},expand_grid=\{source.expansion().unwrap().1.0};",
    )
    println(
      "static constexpr int attention_grid=\{source.attention().launch(2).0},attention_workers=\{source.attention().launch(2).1};",
    )
    println(
      "static constexpr unsigned long long attention_scratch=\{source.attention().scratch_bytes()};",
    )
    println("LUNA_LATENT_END")
  }
}
