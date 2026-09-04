///|
fn pinned_qwen3_metadata() -> @qwen3_spec.Qwen3ModelMetadata raise {
  let spec = @qwen3_spec.Qwen3ModelSpec::new(
    vocabulary_size=151936,
    hidden_size=1024,
    intermediate_size=3072,
    layer_count=28,
    attention_head_count=16,
    kv_head_count=8,
    head_dimension=128,
    context_length=40960,
    rope_theta=1000000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
    tied_embeddings=true,
  )
  @qwen3_spec.Qwen3ModelMetadata::from_verified_content(
    spec,
    @spec.ContentDigest::from_sha256(
      "3333333333333333333333333333333333333333333333333333333333333333",
    ),
  )
}

///|
test "Qwen3 plan represents exact wide-head query-key normalization" {
  let metadata = pinned_qwen3_metadata()
  let model_plan = @qwen3.build(metadata)
  assert_eq(model_plan.operations().length(), 283)
  assert_eq(model_plan.shape_constraints().query_width(), 2048)
  assert_eq(model_plan.shape_constraints().kv_width(), 1024)
  assert_eq(model_plan.numeric_binding().tensors().length(), 310)
  let first_qkv = model_plan.operations()[2]
  let first_qk_norm = model_plan.operations()[3]
  let expected_qkv : Array[Int] = [2, 3, 4]
  for index in 1..<4 {
    guard first_qkv.inputs()[index] is Tensor(reference) else {
      fail("Qwen3 projection tensor role drift")
    }
    assert_eq(reference.as_int(), expected_qkv[index - 1])
  }
  guard first_qk_norm.inputs()[1] is Tensor(query_norm) &&
    first_qk_norm.inputs()[2] is Tensor(key_norm) else {
    fail("Qwen3 Q/K norm tensor role drift")
  }
  assert_eq(query_norm.as_int(), 5)
  assert_eq(key_norm.as_int(), 6)
  for layer in 0..<28 {
    let operation = model_plan.operations()[3 + layer * 10]
    assert_eq(operation.kind(), QkRmsNorm)
    guard operation.shape()
      is QkNormalizationShape(
        query_heads=16,
        kv_heads=8,
        head_dimension=128,
        epsilon=0.000001
      ) else {
      fail("Qwen3 Q/K normalization shape drift")
    }
  }
  let head = model_plan.operations()[282]
  guard head.inputs()[1] is Tensor(reference) else {
    fail("Qwen3 tied head is not a tensor")
  }
  assert_eq(reference.as_int(), 0)
}

///|
test "Qwen3 paged plan binds live inputs and semantic capability 11" {
  let batch = @qwen3.Qwen3PagedBatchEnvelope::new(3)
  let model_plan = @qwen3.build_paged(pinned_qwen3_metadata(), batch~)
  assert_eq(model_plan.kv_execution(), PagedKeyValue)
  assert_eq(model_plan.shape_constraints().max_batch_rows(), 3)
  assert_true(
    model_plan
    .required_capabilities()
    .any(capability => capability.as_int() == 11),
  )
  let qk = model_plan.operations()[3]
  assert_eq(qk.runtime_inputs(), [StepCounts])
  let head = model_plan.operations()[282]
  assert_eq(head.kind(), LanguageModelHead)
  assert_eq(head.runtime_inputs(), [StepCounts, QueryRowOffsets])
}
