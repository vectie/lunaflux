///|
pub fn AttentionMultiLaunchLimits::new(
  maximum_workspace_bytes~ : Int64,
  maximum_symbol_bytes~ : Int,
  maximum_operands_per_launch~ : Int,
) -> AttentionMultiLaunchLimits raise AttentionMultiLaunchContractError {
  if maximum_workspace_bytes <= 0L ||
    maximum_symbol_bytes <= 0 ||
    maximum_symbol_bytes > 4096 ||
    maximum_operands_per_launch <= 0 ||
    maximum_operands_per_launch > 32 {
    raise InvalidAttentionMultiLaunchContract(Limits)
  }
  { maximum_workspace_bytes, maximum_symbol_bytes, maximum_operands_per_launch }
}

///|
pub fn AttentionDecodeWorkspaceContract::new(
  scalar~ : AttentionWorkspaceScalar,
  split_count~ : Int,
  partial_count~ : Int64,
  maximum_offset_bytes~ : Int64,
  denominator_offset_bytes~ : Int64,
  numerator_offset_bytes~ : Int64,
  total_bytes~ : Int64,
  alignment~ : Int64,
  limits~ : AttentionMultiLaunchLimits,
) -> AttentionDecodeWorkspaceContract raise AttentionMultiLaunchContractError {
  if scalar != F32 ||
    split_count <= 0 ||
    partial_count <= 0L ||
    partial_count > 1152921504606846975L ||
    maximum_offset_bytes != 0L ||
    denominator_offset_bytes != partial_count * 4L ||
    numerator_offset_bytes != partial_count * 8L ||
    total_bytes <= numerator_offset_bytes ||
    total_bytes > limits.maximum_workspace_bytes ||
    alignment < 4L ||
    alignment > 4096L ||
    (alignment & (alignment - 1L)) != 0L ||
    maximum_offset_bytes % alignment != 0L ||
    denominator_offset_bytes % 4L != 0L ||
    numerator_offset_bytes % 4L != 0L ||
    total_bytes % 4L != 0L {
    raise InvalidAttentionMultiLaunchContract(Workspace)
  }
  {
    scalar,
    split_count,
    partial_count,
    maximum_offset_bytes,
    denominator_offset_bytes,
    numerator_offset_bytes,
    total_bytes,
    alignment,
  }
}

///|
pub fn AttentionOrderedLaunchInput::new(
  stage~ : AttentionDecodeLaunchStage,
  entry_point~ : @catalog.AotKernelEntryPoint,
  function_symbol~ : String,
  dimensions~ : @launch_contract.AotLaunchDimensions,
  operands~ : ArrayView[@launch_contract.AotOperand],
) -> AttentionOrderedLaunchInput raise AttentionMultiLaunchContractError {
  if function_symbol.is_empty() ||
    function_symbol.contains("\n") ||
    function_symbol.contains("\r") ||
    operands.is_empty() ||
    operands.length() > 32 {
    raise InvalidAttentionMultiLaunchContract(Launch)
  }
  {
    stage,
    entry_point,
    function_symbol,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
  }
}

///|
fn validate_workspace_operand(
  operand : @launch_contract.AotOperand,
  workspace : AttentionDecodeWorkspaceContract,
) -> Unit raise AttentionMultiLaunchContractError {
  if operand.role() != Workspace ||
    operand.byte_count() != workspace.total_bytes ||
    operand.alignment() != workspace.alignment {
    raise MismatchedAttentionMultiLaunchContract(Workspace)
  }
}

///|
fn validate_partial_operands(
  operands : ReadOnlyArray[@launch_contract.AotOperand],
  workspace : AttentionDecodeWorkspaceContract,
) -> Unit raise AttentionMultiLaunchContractError {
  if operands.length() != 10 ||
    operands[0].role() != RuntimeInput(StepCounts) ||
    operands[1].role() != RuntimeInput(QueryPositions) ||
    operands[2].role() != RuntimeInput(QueryRowOffsets) ||
    operands[3].role() != RuntimeInput(SequenceLengths) ||
    operands[4].role() != RuntimeInput(PageTableOffsets) ||
    operands[5].role() != RuntimeInput(PageTablePageIndices) ||
    !(operands[6].role() is ActivationInput(_)) ||
    !(operands[7].role() is PersistentKvComponent(_, Key)) ||
    !(operands[8].role() is PersistentKvComponent(_, Value)) {
    raise MismatchedAttentionMultiLaunchContract(Operand)
  }
  validate_workspace_operand(operands[9], workspace)
}

///|
fn validate_merge_operands(
  operands : ReadOnlyArray[@launch_contract.AotOperand],
  workspace : AttentionDecodeWorkspaceContract,
) -> Unit raise AttentionMultiLaunchContractError {
  if operands.length() != 4 ||
    operands[0].role() != RuntimeInput(StepCounts) ||
    operands[1].role() != RuntimeInput(QueryRowOffsets) ||
    !(operands[3].role() is ActivationOutput(_)) {
    raise MismatchedAttentionMultiLaunchContract(Operand)
  }
  validate_workspace_operand(operands[2], workspace)
}

///|
pub fn admit_attention_decode_multilaunch_contract(
  operation_id~ : @plan.OperationId,
  profile~ : @launch_contract.PagedKernelProfile,
  target~ : @catalog.DeviceTarget,
  workspace~ : AttentionDecodeWorkspaceContract,
  partial~ : AttentionOrderedLaunchInput,
  merge~ : AttentionOrderedLaunchInput,
  limits~ : AttentionMultiLaunchLimits,
) -> AttentionDecodeMultiLaunchContract raise AttentionMultiLaunchContractError {
  if !target.supports_bf16() {
    raise InvalidAttentionMultiLaunchContract(Target)
  }
  if partial.stage != Partial || merge.stage != Merge {
    raise MismatchedAttentionMultiLaunchContract(Launch)
  }
  if partial.function_symbol.length() > limits.maximum_symbol_bytes ||
    merge.function_symbol.length() > limits.maximum_symbol_bytes ||
    partial.operands.length() > limits.maximum_operands_per_launch ||
    merge.operands.length() > limits.maximum_operands_per_launch {
    raise InvalidAttentionMultiLaunchContract(Limits)
  }
  let partial_family = partial.entry_point.family()
  let merge_family = merge.entry_point.family()
  if partial_family.module_digest() != merge_family.module_digest() ||
    partial_family.family_id() != merge_family.family_id() ||
    partial.entry_point.entry_point_id() == merge.entry_point.entry_point_id() {
    raise MismatchedAttentionMultiLaunchContract(Module)
  }
  if partial.function_symbol == merge.function_symbol {
    raise MismatchedAttentionMultiLaunchContract(Symbol)
  }
  if partial.dimensions.grid_x() != profile.max_query_rows() ||
    merge.dimensions.grid_x() != profile.max_query_rows() ||
    partial.dimensions.grid_y() != merge.dimensions.grid_y() ||
    partial.dimensions.grid_z() != workspace.split_count ||
    merge.dimensions.grid_z() != 1 ||
    partial.dimensions.block_x() != merge.dimensions.block_x() ||
    partial.dimensions.block_y() != merge.dimensions.block_y() ||
    partial.dimensions.block_z() != merge.dimensions.block_z() ||
    partial.dimensions.shared_memory_bytes() !=
    merge.dimensions.shared_memory_bytes() {
    raise MismatchedAttentionMultiLaunchContract(Launch)
  }
  validate_partial_operands(partial.operands, workspace)
  validate_merge_operands(merge.operands, workspace)
  let launches : Array[AttentionOrderedLaunchContract] = [
    {
      ordinal: 0,
      stage: partial.stage,
      entry_point: partial.entry_point,
      function_symbol: partial.function_symbol,
      dimensions: partial.dimensions,
      operands: partial.operands,
    },
    {
      ordinal: 1,
      stage: merge.stage,
      entry_point: merge.entry_point,
      function_symbol: merge.function_symbol,
      dimensions: merge.dimensions,
      operands: merge.operands,
    },
  ]
  {
    operation_id,
    profile,
    target,
    workspace,
    launches: ReadOnlyArray::from_array(launches),
  }
}
