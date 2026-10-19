///|
test "ordered decoder shares only dead sparse scratch and budgets it once" {
  let slice = GlmDecoderSlice::new(
    @glm53_spec.Glm53FlashSpec::official(),
    first_layer=0,
    end_layer=45,
    maximum_rows=1,
    maximum_sequences=1,
    cache_slots=1,
    maximum_history=64,
    index_layer_norm_epsilon=1.0e-5,
    maximum_layer_state_bytes=10000000L,
    maximum_layer_workspace_bytes=20000000L,
    maximum_layer_expert_bytes=5000000000L,
    accuracy=@precision_ir.Accuracy::new(absolute=0.125, relative=0.125),
  )
  defer slice.close()
  let sparse = slice.layers.filter(layer => layer is Sparse(_))
  assert_eq(sparse.length(), 11)
  let scratch = sparse[0].transient_scratch_bytes()
  assert_true(scratch > 0L)
  assert_eq(slice.scratch_plan.bytes(), scratch)
  assert_eq(slice.scratch_plan.saved_bytes(), scratch * 10L)
  let unshared = slice.layers.fold(init=0L, (n, layer) => {
    n + layer.workspace_bytes()
  })
  assert_eq(
    slice.workspace_bytes(),
    unshared +
    2L * slice.residual_bytes -
    10L * scratch -
    slice.expert_scratch_plan.saved_bytes(),
  )
  assert_true(slice.scratch_buffers.is_empty())
  assert_true(slice.expert_scratch_buffers.is_empty())
  assert_true(slice.expert_scratch_plan.saved_bytes() > 0L)
}

///|
test "full service packed transport is serial across layers not shared producers" {
  let slice = GlmDecoderSlice::new(
    @glm53_spec.Glm53FlashSpec::official(),
    first_layer=0,
    end_layer=45,
    maximum_rows=64,
    maximum_sequences=8,
    cache_slots=8,
    maximum_history=8192,
    paged_history=@history_domain.PagedHistoryDomain::new(
      rows_per_page=16,
      physical_pages=513,
      maximum_context=8192,
      maximum_sequences=8,
    ).unwrap(),
    index_layer_norm_epsilon=1.0e-5,
    maximum_layer_state_bytes=10000000000L,
    maximum_layer_workspace_bytes=10000000000L,
    maximum_layer_expert_bytes=5000000000L,
    accuracy=@precision_ir.Accuracy::new(absolute=0.125, relative=0.125),
    matrix_law=ShapeAwareFloat32,
    sparse_accumulation=TiledFloat32,
    expert_scaled_dot_law=BlockScaledMatrixFloat32(block_columns=16),
    expert_transport=PackedRows,
    expert_fragment_slots=2,
    tensor_group=(0, 2),
  )
  defer slice.close()
  let sizes = slice.layers.map(layer => layer.expert_scratch_bytes())
  let positive = sizes.filter(n => n > 0L)
  assert_true(positive.length() > 1)
  let peak = positive.fold(init=0L, (a, b) => Int64::max(a, b))
  let total = positive.fold(init=0L, (a, b) => a + b)
  assert_eq(slice.expert_scratch_plan.bytes(), peak)
  assert_eq(slice.expert_scratch_plan.saved_bytes(), total - peak)
  inspect(
    (positive.length(), peak, total - peak),
    content="(42, 19957776, 818268816)",
  )
}
