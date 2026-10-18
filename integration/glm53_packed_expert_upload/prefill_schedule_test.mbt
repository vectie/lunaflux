///|
test "prefill schedules reach both layer families and scale scratch reaches startup budget" {
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
      matrix_staging=Buffered(inner_tile=128),
      sparse_matrix_staging=if enabled {
        Buffered(inner_tile=128)
      } else {
        Synchronous
      },
      expert_scale_partitions=if enabled { 32 } else { 1 },
      expert_scaled_dot_law=BlockScaledMatrixFloat32(block_columns=16),
      tensor_group=(0, 2),
    )
  })
  defer (for s in stages {
    s.close()
  })
  let text = @utf8.decode(stages[1].source_bytes())
  assert_true(text.contains("glm_sparse_3_projection_q_a"))
  assert_true(text.contains("producers*256"))
  let output = text.split("void glm_sparse_3_output(").to_array()
  assert_eq(output.length(), 2)
  assert_true(
    output[1].split("glm_sparse_3_output_publish").to_array()[0].contains(
      "physical operand ring=2, K=128",
    ),
  )
  assert_true(stages[1].workspace_bytes() > stages[0].workspace_bytes())
}
