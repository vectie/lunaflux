///|
fn deepseek_kernel_plan(
  model_id : StringView,
  digest_fill : Char,
) -> @deepseek_v4.DeepSeekV4SemanticPlan raise {
  let spec = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    model_id,
  )
  let content = @spec.ContentDigest::from_sha256(
    digest_fill.to_string().repeat(64),
  )
  @deepseek_v4.build(
    @deepseek_v4_spec.DeepSeekV4ModelMetadata::from_verified_content(
      spec, content,
    ),
  )
}

///|
test "all closed DeepSeek profiles project to exact bounded requirement sets" {
  let cases = [
    ("deepseek-ai/DeepSeek-V4-Flash", '1', 32),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", '2', 32),
    ("deepseek-ai/DeepSeek-V4-Pro", '3', 31),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", '4', 31),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", '5', 40),
  ]
  for entry in cases {
    let (model_id, fill, expected_count) = entry
    let semantic = deepseek_kernel_plan(model_id, fill)
    let requirements = @deepseek_v4_kernel_requirements.project(semantic)
    assert_eq(
      requirements.source_identity().content(),
      semantic.content_digest(),
    )
    assert_eq(
      requirements.source_identity().plan(),
      semantic.execution_plan().plan_digest(),
    )
    assert_eq(requirements.requirements().length(), expected_count)
  }
}

///|
test "Flash-0731 projection retains compressed attention MoE mHC and DSpark semantics" {
  let semantic = deepseek_kernel_plan("deepseek-ai/DeepSeek-V4-Flash-0731", '6')
  let requirements = @deepseek_v4_kernel_requirements.project(semantic)
  let entries = requirements.requirements()
  assert_eq(entries[1].operation(), QueryLowRankAProjection(1024))
  assert_eq(entries[3].operation(), QueryKeyRmsNorm(0.000001, 1024, 512))
  assert_eq(entries[4].operation(), QueryLowRankBProjection(1024, 64, 512))
  assert_eq(
    entries[6].operation(),
    HyperConnectionBlockControlV1(4, 20, 0.000001, 0.000001),
  )
  assert_eq(entries[7].operation(), HyperConnectionPreReduceV1(4))
  assert_eq(entries[8].operation(), HyperConnectionPostCombineV1(4))
  assert_eq(
    entries[9].operation(),
    HyperConnectionHeadReduceV1(4, 0.000001, 0.000001),
  )
  assert_eq(entries[10].operation(), SlidingWindowCausalAttention(128))
  assert_eq(entries[11].operation(), LearnedCompressedIndex(4, 64, 128, 512))
  assert_eq(entries[13].operation(), DeterministicCompressedIndex(128))
  assert_eq(
    entries[14].operation(),
    DeterministicWindowCompressedIndexJoinV1(128, 128),
  )
  assert_eq(
    entries[15].operation(),
    NonOverlappingCompressedKeyValuePrepareV1(128, 128),
  )
  assert_eq(entries[16].operation(), SelectedSparseAttentionV1(128, 128))
  assert_eq(
    entries[17].operation(),
    AttentionOutputInverseRotaryV1(160000.0, 65536, 16.0, 32.0, 1.0),
  )
  assert_eq(
    entries[18].operation(),
    AttentionOutputAProjectionV1(1024, 8, 64, 512),
  )
  assert_eq(
    entries[19].operation(),
    AttentionOutputBProjectionV1(1024, 8, 4096),
  )
  assert_eq(entries[20].operation(), ExpertScoreTransform(SqrtSoftplus, 256))
  assert_eq(entries[21].operation(), TokenHashRouting(3, 129280, 6, 256))
  assert_eq(entries[22].operation(), NoAuxiliaryLossTopK(256, 6))
  assert_eq(
    entries[23].operation(),
    SelectedExpertWeightFinalize(256, 6, true, 1.5),
  )
  assert_eq(entries[27].operation(), DsparkTargetLayerRead(40))
  assert_eq(entries[28].operation(), DsparkTargetLayerRead(41))
  assert_eq(entries[29].operation(), DsparkTargetLayerRead(42))
  assert_eq(entries[31].operation(), DsparkNoiseEmbedding(5, 128799))
  assert_eq(entries[32].operation(), AuxiliaryDecoderStage(0))
  assert_eq(entries[35].operation(), AuxiliaryOutputHead)
  assert_eq(entries[36].operation(), DsparkMarkovAdjustment(256))
  assert_eq(entries[37].operation(), DsparkConfidence)
  assert_eq(requirements.capabilities().length(), 32)
  inspect(
    requirements.digest().as_hex(),
    content=(
      #|2a97e33fe76d1466193140c2e2b0a5a45a05aec84ca83536eacab2255b42e257
    ),
  )
}

///|
test "DeepSeek projection does not claim compatibility with dense generic kernels" {
  let requirements = @deepseek_v4_kernel_requirements.project(
    deepseek_kernel_plan("deepseek-ai/DeepSeek-V4-Flash", '7'),
  )
  try requirements.require_current_generic_capabilities() catch {
    UnsupportedGenericKernelOperation(1, QueryLowRankAProjection(1024)) => ()
    _ => fail("DeepSeek returned the wrong dense-kernel rejection")
  } noraise {
    _ => fail("DeepSeek advanced semantics were weakened to dense kernels")
  }
}

///|
test "profile and content changes perturb projected identity and digest" {
  let flash = @deepseek_v4_kernel_requirements.project(
    deepseek_kernel_plan("deepseek-ai/DeepSeek-V4-Flash", '8'),
  )
  let pro = @deepseek_v4_kernel_requirements.project(
    deepseek_kernel_plan("deepseek-ai/DeepSeek-V4-Pro", '8'),
  )
  let other_content = @deepseek_v4_kernel_requirements.project(
    deepseek_kernel_plan("deepseek-ai/DeepSeek-V4-Flash", '9'),
  )
  assert_true(flash.source_identity() != pro.source_identity())
  assert_true(flash.digest() != pro.digest())
  assert_true(flash.source_identity() != other_content.source_identity())
  assert_true(flash.digest() != other_content.digest())
}
