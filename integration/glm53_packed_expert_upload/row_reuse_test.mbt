///|
test "row reuse reaches fused recurrent source and leaves sparse numerics alone" {
  let stages = [16, 32].map(matrix_tile_rows => {
    @glm53_packed_expert_upload.GlmDecoderSlice::new(
      @glm53_spec.Glm53FlashSpec::official(),
      first_layer=3,
      end_layer=5,
      maximum_rows=64,
      maximum_sequences=1,
      cache_slots=1,
      maximum_history=64,
      index_layer_norm_epsilon=1.0e-5,
      maximum_layer_state_bytes=100000000L,
      maximum_layer_workspace_bytes=100000000L,
      maximum_layer_expert_bytes=5000000000L,
      accuracy=@precision_ir.Accuracy::new(absolute=0.125, relative=0.125),
      matrix_law=ShapeAwareFloat32,
      matrix_tile_rows~,
      matrix_reuse_minimum_inner=4096,
      matrix_staging=Synchronous,
      tensor_group=(0, 2),
    )
  })
  defer (for stage in stages {
    stage.close()
  })
  let old = @utf8.decode(stages[0].source_bytes())
  let next = @utf8.decode(stages[1].source_bytes())
  let marker = "void glm_recurrent_4_inputs("
  let fused = next.split(marker).to_array()[1].split("extern \"C\"").to_array()[0]
  assert_true(fused.contains("tile_a[1024],tile_b[1024]"))
  assert_true(next.contains("(warp/2)*16*32+kk"))
  for role in [4, 9] {
    let marker = "void glm_recurrent_4_\{role}("
    assert_eq(
      old.split(marker).to_array()[1].split("extern \"C\"").to_array()[0],
      next.split(marker).to_array()[1].split("extern \"C\"").to_array()[0],
    )
  }
  let sparse = "void glm_sparse_3_projection_q_a("
  assert_eq(
    old.split(sparse).to_array()[1].split("extern \"C\"").to_array()[0],
    next.split(sparse).to_array()[1].split("extern \"C\"").to_array()[0],
  )
  assert_eq(stages[0].workspace_bytes(), stages[1].workspace_bytes())
}
