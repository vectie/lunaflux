///|
fn numeric_semantic_plan(
  model_id : StringView,
  fill : Char,
) -> @deepseek_v4.DeepSeekV4SemanticPlan raise {
  let spec = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    model_id,
  )
  let content = @spec.ContentDigest::from_sha256(fill.to_string().repeat(64))
  @deepseek_v4.build(
    @deepseek_v4_spec.DeepSeekV4ModelMetadata::from_verified_content(
      spec, content,
    ),
  )
}

///|
fn tensor_run_count(
  plan : @deepseek_v4_numeric_plan.DeepSeekV4NumericPlan,
) -> Int {
  let mut total = 0
  for run in plan.tensor_runs() {
    total += run.tensor_count()
  }
  total
}

///|
fn storage_for(
  plan : @deepseek_v4_numeric_plan.DeepSeekV4NumericPlan,
  role : @deepseek_v4_numeric_plan.DeepSeekV4NumericTensorRole,
) -> @deepseek_v4_numeric_plan.DeepSeekV4LogicalStorage? {
  for run in plan.tensor_runs() {
    if run.role() == role {
      return Some(run.storage())
    }
  }
  None
}

///|
fn count_execution(
  plan : @deepseek_v4_numeric_plan.DeepSeekV4NumericPlan,
  expected : @deepseek_v4_numeric_plan.DeepSeekV4OperationExecution,
) -> Int {
  let mut count = 0
  for operation in plan.operations() {
    if operation.execution() == expected {
      count += 1
    }
  }
  count
}

///|
test "all five closed profiles expand to exact manifest cardinalities" {
  let cases = [
    ("deepseek-ai/DeepSeek-V4-Flash", '1', 69187),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", '2', 69189),
    ("deepseek-ai/DeepSeek-V4-Pro", '3', 145116),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", '4', 145118),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", '5', 72317),
  ]
  let operation_counts : Array[Int] = []
  for entry in cases {
    let (model_id, fill, expected) = entry
    let semantic = numeric_semantic_plan(model_id, fill)
    let plan = @deepseek_v4_numeric_plan.build(semantic)
    assert_eq(plan.expected_tensor_count(), expected)
    assert_eq(tensor_run_count(plan), expected)
    assert_eq(plan.identity().content(), semantic.content_digest())
    assert_eq(
      plan.source_plan_digest(),
      semantic.execution_plan().plan_digest(),
    )
    for ordinal, operation in plan.operations() {
      assert_eq(operation.ordinal(), ordinal)
    }
    assert_eq(storage_for(plan, TokenExpertHashInt64), Some(PlainInt64))
    assert_eq(count_execution(plan, RequiresCheckedInt64ToInt32TokenHash), 3)
    operation_counts.push(plan.operations().length())
  }
  assert_eq(operation_counts, [991, 991, 1404, 1404, 1037])
}

///|
test "FP8 FP4 scale and canonical codebook roles remain distinct" {
  let fp4 = @deepseek_v4_numeric_plan.build(
    numeric_semantic_plan("deepseek-ai/DeepSeek-V4-Flash", '6'),
  )
  assert_eq(fp4.fp4_codebook(), Some(CanonicalE2M1))
  assert_eq(
    storage_for(fp4, RoutedExpertFp4Parameter),
    Some(PackedFp4E2M1Block32Parameter),
  )
  assert_eq(
    storage_for(fp4, RoutedExpertBlock32ScaleMetadata),
    Some(Ue8M0Block32ScaleMetadata),
  )
  assert_eq(storage_for(fp4, RoutedExpertFp8Parameter), None)
  assert_eq(count_execution(fp4, UnsupportedPackedFp4E2M1), 44)
  assert_eq(count_execution(fp4, RequiresCheckedInt64ToInt32TokenHash), 3)
  try fp4.require_physical_materialization() catch {
    PhysicalMaterializationUnsupported(encodings) =>
      assert_eq(encodings, [
        BFloat16,
        Float32,
        Int64,
        Fp8E4M3Block128,
        Ue8M0Scale,
        PackedFp4E2M1Block32,
      ])
    _ => fail("FP4 materialization returned the wrong rejection")
  } noraise {
    _ => fail("logical FP4 schema granted a dequantizer")
  }
  let fp8 = @deepseek_v4_numeric_plan.build(
    numeric_semantic_plan("deepseek-ai/DeepSeek-V4-Flash-Base", '7'),
  )
  assert_eq(fp8.fp4_codebook(), None)
  assert_eq(
    storage_for(fp8, RoutedExpertFp8Parameter),
    Some(Fp8E4M3Block128Parameter),
  )
  assert_eq(
    storage_for(fp8, RoutedExpertBlock128ScaleMetadata),
    Some(Ue8M0Block128ScaleMetadata),
  )
  assert_eq(count_execution(fp8, UnsupportedPackedFp4E2M1), 0)
}

///|
test "Flash-0731 operation order retains exact DSpark and compressed semantics" {
  let plan = @deepseek_v4_numeric_plan.build(
    numeric_semantic_plan("deepseek-ai/DeepSeek-V4-Flash-0731", '8'),
  )
  let operations = plan.operations()
  assert_eq(operations[0].scope(), Global)
  assert_eq(operations[0].kind(), TokenEmbedding)
  assert_eq(operations[1].scope(), DecoderLayer(0))
  assert_eq(operations[1].kind(), BlockRmsNorm)
  let mut learned = 0
  let mut deterministic = 0
  let mut dspark_reads = 0
  let mut markov = 0
  let mut confidence = 0
  let mut attention_controls = 0
  let mut feed_forward_controls = 0
  let mut head_reductions = 0
  for operation in operations {
    match operation.kind() {
      LearnedIndexSelection => learned += 1
      DeterministicCompressedSelection => deterministic += 1
      DsparkTargetLayerRead => dspark_reads += 1
      DsparkMarkovAdjustment => markov += 1
      DsparkConfidence => confidence += 1
      HyperConnectionBlockControl(AttentionSite) => attention_controls += 1
      HyperConnectionBlockControl(FeedForwardSite) => feed_forward_controls += 1
      HyperConnectionHeadReduce => head_reductions += 1
      _ => ()
    }
  }
  assert_eq(dspark_reads, 3)
  assert_eq(markov, 1)
  assert_eq(confidence, 1)
  assert_eq(attention_controls, 46)
  assert_eq(feed_forward_controls, 46)
  assert_eq(head_reductions, 2)
  assert_true(learned > 0)
  assert_true(deterministic > 0)
  assert_eq(operations[operations.length() - 2].kind(), FinalRmsNorm)
  assert_eq(operations[operations.length() - 1].kind(), LanguageModelHead)
}

///|
test "logical numeric identities are deterministic and bind profile content and plan" {
  let first = @deepseek_v4_numeric_plan.build(
    numeric_semantic_plan("deepseek-ai/DeepSeek-V4-Flash", '9'),
  )
  let replay = @deepseek_v4_numeric_plan.build(
    numeric_semantic_plan("deepseek-ai/DeepSeek-V4-Flash", '9'),
  )
  let content_mutation = @deepseek_v4_numeric_plan.build(
    numeric_semantic_plan("deepseek-ai/DeepSeek-V4-Flash", 'a'),
  )
  let profile_mutation = @deepseek_v4_numeric_plan.build(
    numeric_semantic_plan("deepseek-ai/DeepSeek-V4-Pro", '9'),
  )
  assert_eq(first, replay)
  assert_true(
    first.logical_manifest_digest() !=
    content_mutation.logical_manifest_digest(),
  )
  assert_true(
    first.numeric_schema_digest() != content_mutation.numeric_schema_digest(),
  )
  assert_true(first.binding_digest() != content_mutation.binding_digest())
  assert_true(first.binding_digest() != profile_mutation.binding_digest())
  inspect(
    first.binding_digest().as_hex(),
    content="1b2c378466a6732fcab009604ad4787ee96a389858739d5ab1a8fa0fbc95864b",
  )
}

///|
test "hostile manifest counts and every physical materialization request fail typed" {
  let plan = @deepseek_v4_numeric_plan.build(
    numeric_semantic_plan("deepseek-ai/DeepSeek-V4-Pro-Base", 'b'),
  )
  try plan.verify_manifest_tensor_count(145117) catch {
    ManifestCardinalityMismatch(expected=145118, actual=145117) => ()
    _ => fail("wrong manifest count returned the wrong error")
  } noraise {
    _ => fail("truncated Pro-Base manifest was admitted")
  }
  try plan.verify_manifest_tensor_count(-1) catch {
    ManifestCardinalityMismatch(expected=145118, actual=-1) => ()
    _ => fail("negative manifest count returned the wrong error")
  } noraise {
    _ => fail("negative manifest count was admitted")
  }
  try plan.require_physical_materialization() catch {
    PhysicalMaterializationUnsupported(encodings) =>
      assert_eq(encodings, [
        BFloat16,
        Float32,
        Int64,
        Fp8E4M3Block128,
        Ue8M0Scale,
      ])
    _ => fail("physical request returned the wrong typed error")
  } noraise {
    _ => fail("logical numeric planning granted physical materialization")
  }
}
