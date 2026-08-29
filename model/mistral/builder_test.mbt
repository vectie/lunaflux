///|
fn mistral_test_metadata(
  sliding_window? : Int,
) -> @mistral_spec.MistralModelMetadata raise {
  let spec = @mistral_spec.MistralModelSpec::new(
    vocabulary_size=128,
    hidden_size=16,
    intermediate_size=32,
    layer_count=2,
    attention_head_count=4,
    kv_head_count=2,
    max_position_embeddings=128,
    sliding_window?,
    rope_theta=10000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  @mistral_spec.MistralModelMetadata::from_verified_content(
    spec,
    @spec.ContentDigest::from_sha256(
      "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
}

///|
test "Mistral emits the existing dense decoder operation vocabulary" {
  let model_plan = @mistral.build(mistral_test_metadata(sliding_window=64))
  assert_eq(model_plan.shape_constraints().max_sequence_tokens(), 64)
  assert_eq(model_plan.operations().length(), 21)
  let expected : Array[@plan.OperationKind] = [
    Embedding,
    RmsNorm,
    QkvProjection,
    RotaryEmbedding,
    CausalAttention,
    OutputProjection,
    ResidualAdd,
    RmsNorm,
    GatedMlp,
    ResidualAdd,
    RmsNorm,
    QkvProjection,
    RotaryEmbedding,
    CausalAttention,
    OutputProjection,
    ResidualAdd,
    RmsNorm,
    GatedMlp,
    ResidualAdd,
    RmsNorm,
    LanguageModelHead,
  ]
  for index, operation in model_plan.operations() {
    assert_eq(operation.kind(), expected[index])
  }
}

///|
test "Mistral paged plan is deterministic and architecture-neutral" {
  let metadata = mistral_test_metadata(sliding_window=64)
  let batch = @mistral.MistralPagedBatchEnvelope::new(3)
  let first = @mistral.build_paged(metadata, batch~)
  let second = @mistral.build_paged(metadata, batch~)
  assert_eq(first, second)
  assert_eq(first.shape_constraints().max_batch_rows(), 3)
  assert_eq(first.shape_constraints().max_sequence_tokens(), 64)
  assert_eq(first.kv_execution(), PagedKeyValue)
  for operation in first.operations() {
    assert_false(operation.required_capability().as_int() <= 0)
  }
}

///|
test "Mistral context policy is authenticated by the plan identity" {
  let windowed = @mistral.build(mistral_test_metadata(sliding_window=64))
  let unwindowed = @mistral.build(mistral_test_metadata())
  assert_false(windowed.identity().plan() == unwindowed.identity().plan())
  assert_eq(unwindowed.shape_constraints().max_sequence_tokens(), 128)
}

///|
test "Mistral plan is exactly covered by the existing dense kernel graph" {
  let mistral = mistral_test_metadata(sliding_window=64)
  let source = mistral.spec()
  let dense_spec = @spec.LlamaModelSpec::new(
    vocabulary_size=source.vocabulary_size(),
    hidden_size=source.hidden_size(),
    intermediate_size=source.intermediate_size(),
    layer_count=source.layer_count(),
    attention_head_count=source.attention_head_count(),
    kv_head_count=source.kv_head_count(),
    context_length=source.context_length(),
    rope_theta=source.rope_theta(),
    rms_norm_epsilon=source.rms_norm_epsilon(),
    weight_dtype=source.weight_dtype(),
    kv_dtype=source.kv_dtype(),
  )
  let dense = @spec.LlamaModelMetadata::from_verified_content(
    dense_spec,
    mistral.content_digest(),
  )
  assert_eq(@mistral.build(mistral), @llama.build(dense))
  assert_eq(@mistral.build_paged(mistral), @llama.build_paged(dense))
}
