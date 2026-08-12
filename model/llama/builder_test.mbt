///|
fn llama_test_metadata() -> @spec.LlamaModelMetadata raise {
  let spec = @spec.LlamaModelSpec::new(
    vocabulary_size=128,
    hidden_size=16,
    intermediate_size=32,
    layer_count=2,
    attention_head_count=4,
    kv_head_count=2,
    context_length=64,
    rope_theta=10000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  let content = @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let digest = @spec.PlanDigest::from_sha256(
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  )
  @spec.LlamaModelMetadata::new(spec, @spec.ModelIdentity::new(content, digest))
}

///|
test "dense Llama builder emits complete operations in semantic order" {
  let model_plan = @llama.build(llama_test_metadata())
  let operations = model_plan.operations()
  assert_eq(operations.length(), 21)
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
  for index, operation in operations {
    assert_eq(operation.id().as_int(), index)
    assert_eq(operation.kind(), expected[index])
    assert_eq(operation.outputs()[0].as_int(), index)
  }
}

///|
test "dense Llama dimensions and workspace are explicit and bounded" {
  let model_plan = @llama.build(llama_test_metadata())
  let constraints = model_plan.shape_constraints()
  assert_eq(constraints.max_batch_rows(), 1)
  assert_eq(constraints.max_sequence_tokens(), 64)
  assert_eq(constraints.vocabulary_size(), 128)
  assert_eq(constraints.hidden_size(), 16)
  assert_eq(constraints.intermediate_size(), 32)
  assert_eq(constraints.attention_head_count(), 4)
  assert_eq(constraints.kv_head_count(), 2)
  assert_eq(constraints.head_dimension(), 4)
  assert_eq(model_plan.workspace_bounds().max_temporary_bytes(), 8192L)
  guard model_plan.operations()[2].shape()
    is ProjectionShape(input_width=16, output_width=32) else {
    fail("unexpected QKV projection dimensions")
  }
  guard model_plan.operations()[1].shape()
    is NormalizationShape(width=16, epsilon=0.000001) else {
    fail("unexpected RMS normalization semantics")
  }
  guard model_plan.operations()[3].shape()
    is RotaryShape(
      query_heads=4,
      kv_heads=2,
      head_dimension=4,
      rope_theta=10000.0
    ) else {
    fail("unexpected rotary dimensions or parameters")
  }
  guard model_plan.operations()[20].shape()
    is ProjectionShape(input_width=16, output_width=128) else {
    fail("unexpected language-model head dimensions")
  }
}

///|
test "dense Llama plan construction is deterministic" {
  let first = @llama.build(llama_test_metadata())
  let second = @llama.build(llama_test_metadata())
  assert_eq(first, second)
  assert_eq(first.identity(), llama_test_metadata().identity())
}

///|
test "required semantic capabilities are deduplicated in first-use order" {
  let capabilities = @llama.build(llama_test_metadata()).required_capabilities()
  let expected = [
    @plan.KernelCapabilityId::embedding_lookup(),
    @plan.KernelCapabilityId::rms_norm(),
    @plan.KernelCapabilityId::qkv_projection(),
    @plan.KernelCapabilityId::rotary_embedding(),
    @plan.KernelCapabilityId::causal_attention(),
    @plan.KernelCapabilityId::dense_projection(),
    @plan.KernelCapabilityId::residual_add(),
    @plan.KernelCapabilityId::gated_mlp(),
  ]
  assert_eq(capabilities.length(), expected.length())
  for index, capability in capabilities {
    assert_eq(capability, expected[index])
    assert_eq(capability.as_int(), index + 1)
  }
}
