///|
test "kernel inspection operation names are stable and complete" {
  let cases : FixedArray[(@plan.OperationKind, String)] = [
    (Embedding, "embedding"),
    (RmsNorm, "rms_norm"),
    (QkvProjection, "qkv_projection"),
    (RotaryEmbedding, "rotary_embedding"),
    (CausalAttention, "causal_attention"),
    (OutputProjection, "output_projection"),
    (ResidualAdd, "residual_add"),
    (GatedMlp, "gated_mlp"),
    (LanguageModelHead, "language_model_head"),
  ]
  for _, pair in cases {
    let (kind, expected) = pair
    assert_eq(operation_kind_name(kind), expected)
  }
}

///|
test "kernel inspection renders exact shape scalars without Debug output" {
  let output = StringBuilder()
  append_shape(output, ProjectionShape(input_width=64, output_width=128))
  inspect(
    output,
    content=(
      #|shape: projection
      #|shape.input_width: 64
      #|shape.output_width: 128
      #|
    ),
  )
  assert_eq(phase_name(FullPrefill), "full_prefill")
  assert_eq(phase_name(FullRecompute), "full_recompute")
}
