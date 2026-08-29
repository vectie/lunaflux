///|
const TP_INT32_BYTES : Int64 = 4L

///|
const TP_STEP_COUNTS_BYTES : Int64 = 20L

///|
const TP_BF16_BYTES : Int64 = 2L

///|
fn checked_product(
  left : Int64,
  right : Int64,
  profile : Int,
  operation : Int,
) -> Int64 raise TensorParallelLaunchContractError {
  if left <= 0L || right <= 0L || left > TP_MAX_INT64 / right {
    raise invalid(CapacityOverflow, profile~, operation~)
  }
  left * right
}

///|
fn runtime_bytes(
  role : @plan.OperationRuntimeInput,
  profile : @launch_contract.PagedKernelProfile,
  profile_index : Int,
  operation : Int,
) -> Int64 raise TensorParallelLaunchContractError {
  match role {
    StepCounts => TP_STEP_COUNTS_BYTES
    QueryPositions =>
      checked_product(
        Int64::from_int(profile.max_query_tokens()),
        TP_INT32_BYTES,
        profile_index,
        operation,
      )
    QueryRowOffsets | PageTableOffsets =>
      checked_product(
        Int64::from_int(profile.max_query_rows()) + 1L,
        TP_INT32_BYTES,
        profile_index,
        operation,
      )
    SequenceLengths =>
      checked_product(
        Int64::from_int(profile.max_query_rows()),
        TP_INT32_BYTES,
        profile_index,
        operation,
      )
    PageTablePageIndices =>
      checked_product(
        Int64::from_int(profile.max_page_table_entries()),
        TP_INT32_BYTES,
        profile_index,
        operation,
      )
  }
}

///|
fn activation_bytes(
  profile : @launch_contract.PagedKernelProfile,
  width : Int,
  profile_index : Int,
  operation : Int,
) -> Int64 raise TensorParallelLaunchContractError {
  let elements = checked_product(
    Int64::from_int(profile.max_query_tokens()),
    Int64::from_int(width),
    profile_index,
    operation,
  )
  checked_product(elements, TP_BF16_BYTES, profile_index, operation)
}

///|
fn push_operand(
  output : Array[TensorParallelOperand],
  role : TensorParallelOperandRole,
  byte_count : Int64,
  alignment : Int64,
  profile : Int,
  operation : Int,
) -> Unit raise TensorParallelLaunchContractError {
  if byte_count <= 0L ||
    alignment <= 0L ||
    alignment > 4096L ||
    (alignment & (alignment - 1L)) != 0L {
    raise invalid(CapacityOverflow, profile~, operation~)
  }
  output.push({ role, byte_count, alignment })
}

///|
fn build_operands(
  operation : @plan.PlanOperation,
  binding : @catalog.KernelBinding,
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  layout : @device_layout.DeviceKvLayout,
  profile : @launch_contract.PagedKernelProfile,
  profile_index : Int,
  geometry : @tensor_parallel_plan.TensorParallelOperationPlan,
  maximum : Int,
) -> ReadOnlyArray[TensorParallelOperand] raise TensorParallelLaunchContractError {
  let operation_index = operation.id().as_int()
  let output : Array[TensorParallelOperand] = []
  for runtime_input in operation.runtime_inputs() {
    push_operand(
      output,
      RuntimeInput(runtime_input),
      runtime_bytes(runtime_input, profile, profile_index, operation_index),
      TP_INT32_BYTES,
      profile_index,
      operation_index,
    )
  }
  push_operand(
    output,
    RankConstants(version=TP_RANK_CONSTANTS_VERSION),
    TP_RANK_CONSTANTS_BYTES,
    TP_RANK_CONSTANTS_ALIGNMENT,
    profile_index,
    operation_index,
  )
  for input in operation.inputs() {
    match input {
      ModelInput(reference) =>
        push_operand(
          output,
          ExternalInput(reference),
          checked_product(
            Int64::from_int(profile.max_query_tokens()),
            TP_INT32_BYTES,
            profile_index,
            operation_index,
          ),
          TP_INT32_BYTES,
          profile_index,
          operation_index,
        )
      Tensor(reference) =>
        push_operand(
          output,
          WeightInput(reference),
          rank_weight_bytes(rank_plan, reference),
          rank_plan.alignment_bytes(),
          profile_index,
          operation_index,
        )
      Activation(reference) =>
        push_operand(
          output,
          ActivationInput(reference),
          activation_bytes(
            profile,
            geometry.input_width(),
            profile_index,
            operation_index,
          ),
          rank_plan.alignment_bytes(),
          profile_index,
          operation_index,
        )
    }
  }
  for reference in operation.outputs() {
    push_operand(
      output,
      ActivationOutput(reference),
      activation_bytes(
        profile,
        geometry.output_width(),
        profile_index,
        operation_index,
      ),
      rank_plan.alignment_bytes(),
      profile_index,
      operation_index,
    )
  }
  match operation.persistent_state() {
    NoPersistentState => ()
    KvCacheReadWrite(layer) => {
      push_operand(
        output,
        PersistentKvComponent(layer, Key),
        layout.component_stride_bytes(),
        layout.segment_alignment_bytes(),
        profile_index,
        operation_index,
      )
      push_operand(
        output,
        PersistentKvComponent(layer, Value),
        layout.component_stride_bytes(),
        layout.segment_alignment_bytes(),
        profile_index,
        operation_index,
      )
    }
  }
  if binding.workspace_bytes() > 0L {
    push_operand(
      output,
      Workspace,
      binding.workspace_bytes(),
      binding.workspace_alignment(),
      profile_index,
      operation_index,
    )
  }
  if output.is_empty() || output.length() > maximum {
    raise invalid(
      OperandCount,
      profile=profile_index,
      operation=operation_index,
    )
  }
  ReadOnlyArray::from_array(output)
}

///|
test "operand capacity multiplication rejects overflow" {
  try ignore(checked_product(TP_MAX_INT64, 2L, 0, 0)) catch {
    InvalidTensorParallelLaunchContract(rule=CapacityOverflow, ..) => ()
    _ => fail("unexpected operand-overflow rejection")
  } noraise {
    _ => fail("expected operand-overflow rejection")
  }
}
