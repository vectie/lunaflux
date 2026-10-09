///|
fn sparse_attention_test_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is SelectedSparseAttentionV1(_, _) {
      return ordinal
    }
  }
  abort("selected sparse-attention requirement missing")
}

///|
test "selected sparse attention lowers exact shared-KV eight-operand ABI" {
  let requirements = aot_test_requirements(
    attention_head_count=2,
    attention_head_dimension=8,
    maximum_position_count=1024,
    deterministic_compression_ratio=128,
  )
  let ordinal = sparse_attention_test_ordinal(requirements)
  let candidate = lower_selected_sparse_attention_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    maximum_batches=1,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.family(), SharedKeyValueSelectedAttentionBf16)
  assert_eq(
    candidate.numerical_contract(),
    SerialOrderedF32SinkDenominatorBf16OutputV1,
  )
  assert_eq(candidate.operation(), SelectedSparseAttentionV1(64, 128))
  assert_eq(candidate.capability(), SelectedSparseAttentionV1Aot)
  assert_eq(candidate.query_head_count(), 2)
  assert_eq(candidate.head_dimension(), 8)
  assert_eq(candidate.selected_width(), 72)
  assert_eq(candidate.maximum_key_slots(), 1032)
  assert_eq(candidate.dimensions().grid_x(), 32)
  assert_eq(candidate.dimensions().block_x(), 1)
  let operands = candidate.operands()
  assert_eq(operands.length(), 8)
  assert_eq(operands[0].role(), StepCountsInput)
  assert_eq(operands[0].byte_count(), 20L)
  assert_eq(operands[1].role(), SparseAttentionQueryInput)
  assert_eq(operands[1].byte_count(), 1024L)
  assert_eq(operands[2].role(), SparseAttentionQueryBatchIdsInput)
  assert_eq(operands[2].byte_count(), 128L)
  assert_eq(operands[3].role(), SparseAttentionKeyValueInput)
  assert_eq(operands[3].byte_count(), 16512L)
  assert_eq(operands[4].role(), SparseAttentionCacheLengthsInput)
  assert_eq(operands[4].byte_count(), 4L)
  assert_eq(operands[5].role(), SparseAttentionSinkInput)
  assert_eq(operands[5].byte_count(), 8L)
  assert_eq(operands[6].role(), SparseAttentionSelectedIndicesInput)
  assert_eq(operands[6].byte_count(), 9216L)
  assert_eq(operands[7].role(), SparseAttentionOutput)
  assert_eq(operands[7].byte_count(), 1024L)
  assert_false(candidate.manifest_bindable())
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("float maximum = attention_sink[head]"))
  assert_false(source.contains("inverse_rope"))
  let repeated = lower_selected_sparse_attention_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    maximum_batches=1,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.source_digest(), repeated.source_digest())
  assert_eq(candidate.recipe_digest(), repeated.recipe_digest())
  let foreign = lower_selected_sparse_attention_cuda_aot_candidate(
    aot_test_requirements(
      content_fill='c',
      maximum_position_count=1024,
      deterministic_compression_ratio=128,
    ),
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    maximum_batches=1,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.source_digest(), foreign.source_digest())
  assert_true(candidate.recipe_digest() != foreign.recipe_digest())
}

///|
test "selected sparse attention rejects adjacent ordinal target compiler and bounds" {
  let requirements = aot_test_requirements(deterministic_compression_ratio=128)
  let ordinal = sparse_attention_test_ordinal(requirements)
  for hostile_ordinal in [ordinal - 1, ordinal + 1] {
    try
      lower_selected_sparse_attention_cuda_aot_candidate(
        requirements,
        requirement_ordinal=hostile_ordinal,
        maximum_tokens=2,
        maximum_batches=1,
        target=aot_test_target(),
        compiler=aot_test_compiler(),
        limits=aot_test_limits(),
      )
    catch {
      UnsupportedRequirement(actual) => assert_eq(actual, hostile_ordinal)
      _ => fail("unexpected adjacent-ordinal error")
    } noraise {
      _ => fail("adjacent phase lowered as selected sparse attention")
    }
  }
  try
    lower_selected_sparse_attention_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=2,
      maximum_batches=1,
      target=aot_test_target(bf16=false),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("unexpected target error")
  } noraise {
    _ => fail("non-BF16 target admitted")
  }
  try
    lower_selected_sparse_attention_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=2,
      maximum_batches=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(fused_multiply_add=true),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("unexpected compiler error")
  } noraise {
    _ => fail("FMA compiler admitted")
  }
  try
    lower_selected_sparse_attention_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=2,
      maximum_batches=3,
      target=aot_test_target(),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("unexpected batch-bound error")
  } noraise {
    _ => fail("more batches than rows admitted")
  }
}
