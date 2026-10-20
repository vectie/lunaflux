///|
test "sparse decoder forwards independent fragment policy and launch geometry" {
  let attention = @precision.IndexedSparseAttentionPrecision::new(
    index_head_count=32, index_head_dimension=128,
    query_head_count=32, key_value_head_count=32,
    query_key_dimension=256, value_dimension=256,
    pool_size=4, top_k=2048, append_incomplete_tail=true,
    maximum_queries=64, maximum_history=8192,
    maximum_workspace_bytes=1073741824L,
    attention_accumulation=TiledFloat32,
  )
  let projection = @precision.IndexedSparseProjectionPrecision::new(
    attention, hidden_width=4096, query_rank=1024, key_value_rank=512,
    rms_epsilon=1.0e-5, index_layer_norm_epsilon=1.0e-5,
    maximum_workspace_bytes=1073741824L, projection_law=ShapeAwareFloat32,
  )
  let envelope = @precision.HyperConnectionPrecision::new(
    stream_count=4, hidden_width=4096, maximum_rows=64,
    sinkhorn_iterations=20, control_epsilon=1.0e-6,
    normalization_epsilon=1.0e-5, mix_law=RoundedBf16ControlsAndProducts,
  )
  let plan = @precision.IndexedSparseDecoderPrecision::new(
    projection, envelope, maximum_state_bytes=1073741824L,
    maximum_workspace_bytes=1073741824L,
  )
  let default = @indexed_sparse_projection_cuda_source.indexed_sparse_decoder_source(
    plan, symbol_prefix="scope",
  )
  let selected = @indexed_sparse_projection_cuda_source.indexed_sparse_decoder_source(
    plan, symbol_prefix="scope",
    row_reuse=@matrix.MatrixRowReuse::new(
      minimum_rows=32, minimum_inner=1536, wide_inner=8192, wide_tile=64,
      fragment_inner=8192, fragment_columns=4096,
    ),
  )
  let source = @utf8.decode(selected.source_bytes())
  assert_true(source.contains("void scope_output("))
  assert_true(source.contains("cp.async.wait_group 1;"))
  assert_true(source.contains("ldmatrix.sync.aligned.m8n8.x4"))
  assert_false(@utf8.decode(default.source_bytes()).contains(
    "cp.async.wait_group 1;",
  ))
  assert_eq(selected.output_launch(), (1024, 128))
  assert_eq(selected.output_launch(), default.output_launch())
  assert_eq(selected.output_operand_alignment(), 16L)
  assert_eq(default.output_operand_alignment(), 2L)
  assert_eq(selected.projection().operand_alignment(0), 2L)
  assert_eq(selected.plan().projection().projection_law(),
    default.plan().projection().projection_law())
  let wider = @indexed_sparse_projection_cuda_source.indexed_sparse_projection_source(
    projection, symbol_prefix="alignment",
    row_reuse=@matrix.MatrixRowReuse::new(
      minimum_rows=32, minimum_inner=1536, wide_inner=8192, wide_tile=64,
      fragment_inner=1024, fragment_columns=4096,
    ),
  )
  assert_eq(wider.operand_alignment(2), 16L)
  assert_eq(wider.operand_alignment(0), 2L)
}
