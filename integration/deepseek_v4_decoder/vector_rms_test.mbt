///|
test "vector RMS arithmetic reaches both envelopes and actual normalization threads" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let selected = @deepseek_v4_decoder.DeepSeekDecoderSlice::new(
    spec,
    first_layer=0,
    end_layer=4,
    maximum_rows=3,
    maximum_context=256,
    maximum_layer_state_bytes=16777216L,
    maximum_layer_workspace_bytes=16777216L,
    maximum_layer_expert_bytes=8589934592L,
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
    control_square_law=VectorizedTreeFloat32(partitions=512, vector_width=4),
    control_root_law=ReciprocalSqrtFloat32,
    control_dot_law=GroupedVectorFloat32(
      groups=18,
      partitions=128,
      vector_width=2,
      publication_partitions=4,
      multiply_add=FusedMultiplyAdd,
    ),
    normalization_square_law=VectorizedTreeFloat32(
      partitions=512,
      vector_width=4,
    ),
  )
  defer selected.close()
  let text = @utf8.decode(selected.source_bytes())
  assert_eq(text.split("float square_vectors[2048]").count(), 9)
  assert_eq(text.split("float dot_vectors[4608]").count(), 9)
  assert_eq(text.split("float dot_publication[4]").count(), 9)
  assert_eq(text.split("blockDim.x != 512").count(), 9)
  assert_eq(
    text
    .split("const float inverse = rsqrtf(__fadd_rn(__fdiv_rn(square_sum")
    .count(),
    9,
  )
}
