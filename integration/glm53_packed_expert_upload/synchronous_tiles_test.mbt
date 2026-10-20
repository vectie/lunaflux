///|
test "bounded synchronous tiles reach sparse leaves without enlarging fused KDA scratch" {
  let stages = [false, true].map(enabled => {
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
      matrix_staging=Synchronous,
      sparse_matrix_staging=if enabled {
        WorksetBoundedTiles(
          inner_tile=128,
          maximum_weight_bytes=16777216L,
          maximum_tiles=192,
        )
      } else {
        Synchronous
      },
      expert_scaled_dot_law=BlockScaledMatrixFloat32(block_columns=16),
      tensor_group=(0, 2),
    )
  })
  defer (for s in stages {
    s.close()
  })
  let old = @utf8.decode(stages[0].source_bytes())
  let next = @utf8.decode(stages[1].source_bytes())
  for
    (symbol, k) in [
      ("glm_sparse_3_projection_q_a", 128),
      ("glm_sparse_3_projection_q_b", 32),
      ("glm_sparse_3_projection_kv_a", 128),
      ("glm_sparse_3_projection_kv_b", 32),
      ("glm_sparse_3_output", 32),
    ] {
    let body = next.split("void " + symbol + "(").to_array()[1]
      .split("extern \"C\"")
      .to_array()[0]
    assert_true(body.contains("base_k+=\{k}"))
    assert_true(body.contains("kk+=16"))
  }
  let marker = "void glm_recurrent_4_inputs("
  assert_eq(
    old.split(marker).to_array()[1].split("extern \"C\"").to_array()[0],
    next.split(marker).to_array()[1].split("extern \"C\"").to_array()[0],
  )
  assert_eq(stages[0].workspace_bytes(), stages[1].workspace_bytes())
}
