///|
fn sparse_attention_test_operand(
  role : AdvancedDecoderLaunchOperandRole,
  byte_count : Int64,
  alignment : Int64,
) -> AdvancedDecoderLaunchOperand {
  try! AdvancedDecoderLaunchOperand::new(role~, byte_count~, alignment~)
}

///|
fn sparse_attention_test_operands() -> Array[AdvancedDecoderLaunchOperand] {
  [
    sparse_attention_test_operand(StepCountsInput, 20L, 4L),
    sparse_attention_test_operand(SparseAttentionQueryInput, 512L, 16L),
    sparse_attention_test_operand(SparseAttentionQueryBatchIdsInput, 8L, 4L),
    sparse_attention_test_operand(SparseAttentionKeyValueInput, 132096L, 16L),
    sparse_attention_test_operand(SparseAttentionCacheLengthsInput, 4L, 4L),
    sparse_attention_test_operand(SparseAttentionSinkInput, 8L, 4L),
    sparse_attention_test_operand(SparseAttentionSelectedIndicesInput, 576L, 4L),
    sparse_attention_test_operand(SparseAttentionOutput, 512L, 16L),
  ]
}

///|
test "selected sparse attention artifact ABI binds roles geometry and bytes" {
  let requirements = deterministic_index_test_requirements()
  let dimensions = try! @launch_contract.AotLaunchDimensions::new(
    grid_x=2,
    grid_y=1,
    grid_z=1,
    block_x=1,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  assert_eq(expected_operand_roles(SelectedSparseAttentionV1(64, 128)), [
    StepCountsInput,
    SparseAttentionQueryInput,
    SparseAttentionQueryBatchIdsInput,
    SparseAttentionKeyValueInput,
    SparseAttentionCacheLengthsInput,
    SparseAttentionSinkInput,
    SparseAttentionSelectedIndicesInput,
    SparseAttentionOutput,
  ])
  assert_true(
    operands_are_exact(
      SelectedSparseAttentionV1(64, 128),
      ReadOnlyArray::from_array(sparse_attention_test_operands()),
      dimensions,
      requirements.geometry(),
      8,
    ),
  )
}

///|
test "selected sparse attention artifact ABI rejects hostile role bytes and launch" {
  let requirements = deterministic_index_test_requirements()
  let dimensions = try! @launch_contract.AotLaunchDimensions::new(
    grid_x=2,
    grid_y=1,
    grid_z=1,
    block_x=1,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  let wrong_role = sparse_attention_test_operands()
  wrong_role[6] = sparse_attention_test_operand(RoutingIndicesInput, 576L, 4L)
  assert_false(
    operands_are_exact(
      SelectedSparseAttentionV1(64, 128),
      ReadOnlyArray::from_array(wrong_role),
      dimensions,
      requirements.geometry(),
      8,
    ),
  )
  let wrong_bytes = sparse_attention_test_operands()
  wrong_bytes[3] = sparse_attention_test_operand(
    SparseAttentionKeyValueInput,
    132080L,
    16L,
  )
  assert_false(
    operands_are_exact(
      SelectedSparseAttentionV1(64, 128),
      ReadOnlyArray::from_array(wrong_bytes),
      dimensions,
      requirements.geometry(),
      8,
    ),
  )
  let wrong_launch = try! @launch_contract.AotLaunchDimensions::new(
    grid_x=2,
    grid_y=1,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  assert_false(
    operands_are_exact(
      SelectedSparseAttentionV1(64, 128),
      ReadOnlyArray::from_array(sparse_attention_test_operands()),
      wrong_launch,
      requirements.geometry(),
      8,
    ),
  )
}
