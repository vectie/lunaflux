///|
fn official_sparse_attention_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is SelectedSparseAttentionV1(128, 128) {
      return ordinal
    }
  }
  abort("official profile lost selected sparse-attention requirement")
}

///|
test "all five official profiles lower exact inert selected sparse attention" {
  let profiles = [
    ("deepseek-ai/DeepSeek-V4-Flash", 'a', 64),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", 'b', 64),
    ("deepseek-ai/DeepSeek-V4-Pro", 'c', 128),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", 'd', 128),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", 'e', 64),
  ]
  for profile in profiles {
    let (model_id, fill, expected_heads) = profile
    let requirements = official_deepseek_requirements(model_id, fill)
    let ordinal = official_sparse_attention_ordinal(requirements)
    let candidate = @advanced_decoder_cuda_aot.lower_selected_sparse_attention_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=2,
      maximum_batches=1,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
    assert_eq(candidate.operation(), SelectedSparseAttentionV1(128, 128))
    assert_eq(candidate.capability(), SelectedSparseAttentionV1Aot)
    assert_eq(candidate.query_head_count(), expected_heads)
    assert_eq(candidate.head_dimension(), 512)
    assert_eq(candidate.window_size(), 128)
    assert_eq(candidate.compression_ratio(), 128)
    assert_eq(candidate.selected_width(), 8320)
    assert_eq(candidate.maximum_key_slots(), 1056768)
    assert_eq(candidate.maximum_batches(), 1)
    assert_eq(candidate.operands().length(), 8)
    assert_eq(candidate.operands()[0].role(), StepCountsInput)
    assert_eq(candidate.operands()[1].role(), SparseAttentionQueryInput)
    assert_eq(candidate.operands()[2].role(), SparseAttentionQueryBatchIdsInput)
    assert_eq(candidate.operands()[3].role(), SparseAttentionKeyValueInput)
    assert_eq(candidate.operands()[3].byte_count(), 1082130432L)
    assert_eq(candidate.operands()[4].role(), SparseAttentionCacheLengthsInput)
    assert_eq(candidate.operands()[5].role(), SparseAttentionSinkInput)
    assert_eq(
      candidate.operands()[6].role(),
      SparseAttentionSelectedIndicesInput,
    )
    assert_eq(candidate.operands()[6].byte_count(), 66560L)
    assert_eq(candidate.operands()[7].role(), SparseAttentionOutput)
    assert_false(candidate.manifest_bindable())
  }
}

///|
test "official selected sparse attention rejects adjacent phase and two-batch KV overflow" {
  let requirements = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Flash-0731", 'f',
  )
  let ordinal = official_sparse_attention_ordinal(requirements)
  try
    @advanced_decoder_cuda_aot.lower_selected_sparse_attention_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal - 1,
      maximum_tokens=2,
      maximum_batches=1,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
  catch {
    UnsupportedRequirement(actual) => assert_eq(actual, ordinal - 1)
    _ => fail("unexpected adjacent phase rejection")
  } noraise {
    _ => fail("KV-preparation gap lowered as sparse attention")
  }
  try
    @advanced_decoder_cuda_aot.lower_selected_sparse_attention_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=2,
      maximum_batches=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
  catch {
    SizeOverflow => ()
    _ => fail("unexpected official KV byte overflow rejection")
  } noraise {
    _ => fail("over-2GiB borrowed KV operand admitted")
  }
}
