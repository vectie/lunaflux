///|
const PAGED_V4_TOKEN_ID_BYTES : Int64 = 4L

///|
const PAGED_V4_VALUE_ALIGNMENT : Int64 = 16L

///|
fn paged_v4_execution(
  model_plan : @plan.ModelPlan,
  operation : @plan.OperationId,
) -> @numeric_contract.OperationExecutionContract raise LaunchContractError {
  guard model_plan.numeric_binding().operations().get(operation.as_int())
    is Some(binding) &&
    binding.operation() == operation else {
    raise NumericBindingMismatch(operation)
  }
  binding.execution()
}

///|
fn paged_v4_tensor(
  model_plan : @plan.ModelPlan,
  operation : @plan.OperationId,
  reference : @plan.TensorRef,
) -> @plan.PlanTensor raise LaunchContractError {
  guard model_plan.tensor(reference) is Some(tensor) &&
    tensor.reference() == reference else {
    raise InvalidOperand(operation, NumericTensorMismatch)
  }
  tensor
}

///|
fn paged_v4_tensor_bytes(
  tensor : @plan.PlanTensor,
  operation : @plan.OperationId,
) -> Int64 raise LaunchContractError {
  @numeric_contract.storage_byte_length(
    tensor.storage().dtype(),
    tensor.shape(),
  ) catch {
    _ => raise InvalidOperand(operation, NumericByteCountMismatch)
  }
}

///|
fn paged_v4_scale_tensor(
  model_plan : @plan.ModelPlan,
  operation : @plan.OperationId,
  weight : @plan.TensorRef,
  scale : @plan.ScaleTensorRef,
) -> @plan.PlanTensor raise LaunchContractError {
  let parameter = paged_v4_tensor(model_plan, operation, weight)
  if parameter.role() != ParameterTensor || parameter.scale() != Some(scale) {
    raise InvalidOperand(operation, NumericTensorMismatch)
  }
  let metadata = paged_v4_tensor(model_plan, operation, scale.tensor())
  if metadata.role() != ScaleMetadataTensor ||
    metadata.scale() is Some(_) ||
    metadata.zero_point() is Some(_) ||
    metadata.codebook() is Some(_) {
    raise InvalidOperand(operation, NumericTensorMismatch)
  }
  metadata
}

///|
fn validate_paged_v4_parameter(
  model_plan : @plan.ModelPlan,
  operation : @plan.OperationId,
  execution : @numeric_contract.OperationExecutionContract,
  reference : @plan.TensorRef,
) -> @plan.PlanTensor raise LaunchContractError {
  let tensor = paged_v4_tensor(model_plan, operation, reference)
  if tensor.role() != ParameterTensor {
    raise InvalidOperand(operation, NumericTensorMismatch)
  }
  if tensor.zero_point() is Some(_) || tensor.codebook() is Some(_) {
    raise InvalidOperand(operation, UnsupportedNumericMetadata)
  }
  if !execution.accepts_storage(tensor.storage()) {
    raise InvalidOperand(operation, NumericTensorMismatch)
  }
  if tensor.storage().requires_scale() != (tensor.scale() is Some(_)) {
    raise InvalidOperand(operation, NumericTensorMismatch)
  }
  tensor
}

///|
fn canonical_paged_v4_roles(
  model_plan : @plan.ModelPlan,
  operation : @plan.PlanOperation,
  binding : @catalog.KernelBinding,
) -> Array[AotOperandRole] raise LaunchContractError {
  let execution = paged_v4_execution(model_plan, operation.id())
  let roles : Array[AotOperandRole] = []
  for runtime in operation.runtime_inputs() {
    roles.push(RuntimeInput(runtime))
  }
  for input in operation.inputs() {
    match input {
      ModelInput(reference) => roles.push(ExternalInput(reference))
      Activation(reference) => roles.push(ActivationInput(reference))
      Tensor(reference) => {
        let tensor = validate_paged_v4_parameter(
          model_plan,
          operation.id(),
          execution,
          reference,
        )
        roles.push(WeightInput(reference))
        match tensor.scale() {
          Some(scale) => {
            ignore(
              paged_v4_scale_tensor(
                model_plan,
                operation.id(),
                reference,
                scale,
              ),
            )
            roles.push(WeightScaleInput(reference, scale))
          }
          None => ()
        }
      }
    }
  }
  for output in operation.outputs() {
    roles.push(ActivationOutput(output))
  }
  match operation.persistent_state() {
    NoPersistentState => ()
    KvCacheReadWrite(layer) => {
      roles.push(PersistentKvComponent(layer, Key))
      roles.push(PersistentKvComponent(layer, Value))
    }
  }
  if binding.workspace_bytes() > 0L {
    roles.push(Workspace)
  }
  roles
}

///|
fn paged_v4_compute_byte_width(
  dtype : @numeric_contract.ComputeDType,
  operation : @plan.OperationId,
) -> Int64 raise LaunchContractError {
  if dtype == @numeric_contract.ComputeDType::bfloat16() {
    2L
  } else if dtype == @numeric_contract.ComputeDType::f8_e4m3_finite() ||
    dtype == @numeric_contract.ComputeDType::i8() {
    1L
  } else if dtype == @numeric_contract.ComputeDType::f32() {
    4L
  } else {
    raise NumericBindingMismatch(operation)
  }
}

///|
fn paged_v4_checked_value_bytes(
  rows : Int,
  width : Int64,
  element_bytes : Int64,
  operation : @plan.OperationId,
) -> Int64 raise LaunchContractError {
  let maximum = 9223372036854775807L
  let row_count = rows.to_int64()
  if row_count <= 0L ||
    width <= 0L ||
    element_bytes <= 0L ||
    row_count > maximum / width ||
    row_count * width > maximum / element_bytes {
    raise InvalidOperand(operation, NumericByteCountMismatch)
  }
  row_count * width * element_bytes
}

///|
fn paged_v4_activation_input_widths(
  operation : @plan.PlanOperation,
) -> Array[Int64] raise LaunchContractError {
  match (operation.kind(), operation.shape()) {
    (Embedding, EmbeddingShape(..)) => []
    (RmsNorm, NormalizationShape(width~, ..)) => [width.to_int64()]
    (QkvProjection, ProjectionShape(input_width~, ..)) =>
      [input_width.to_int64()]
    (RotaryEmbedding, RotaryShape(query_heads~, kv_heads~, head_dimension~, ..)) =>
      [(query_heads + 2 * kv_heads).to_int64() * head_dimension.to_int64()]
    (CausalAttention, AttentionShape(query_heads~, kv_heads~, head_dimension~)) =>
      [(query_heads + 2 * kv_heads).to_int64() * head_dimension.to_int64()]
    (OutputProjection | LanguageModelHead, ProjectionShape(input_width~, ..)) =>
      [input_width.to_int64()]
    (ResidualAdd, ElementwiseShape(width~)) =>
      [width.to_int64(), width.to_int64()]
    (GatedMlp, GatedMlpShape(hidden_size~, ..)) => [hidden_size.to_int64()]
    _ => raise NumericBindingMismatch(operation.id())
  }
}

///|
fn paged_v4_activation_input_width(
  operation : @plan.PlanOperation,
  reference : @plan.ActivationRef,
) -> Int64 raise LaunchContractError {
  let widths = paged_v4_activation_input_widths(operation)
  let mut activation_index = 0
  for input in operation.inputs() {
    match input {
      Activation(candidate) => {
        if candidate == reference {
          guard widths.get(activation_index) is Some(width) else {
            raise NumericBindingMismatch(operation.id())
          }
          return width
        }
        activation_index += 1
      }
      ModelInput(_) | Tensor(_) => ()
    }
  }
  raise InvalidOperand(operation.id(), NumericTensorMismatch)
}

///|
fn paged_v4_activation_output_width(
  operation : @plan.PlanOperation,
  reference : @plan.ActivationRef,
) -> Int64 raise LaunchContractError {
  if !operation.outputs().contains(reference) {
    raise InvalidOperand(operation.id(), NumericTensorMismatch)
  }
  match (operation.kind(), operation.shape()) {
    (Embedding, EmbeddingShape(width~, ..)) => width.to_int64()
    (RmsNorm, NormalizationShape(width~, ..)) => width.to_int64()
    (QkvProjection, ProjectionShape(output_width~, ..)) =>
      output_width.to_int64()
    (RotaryEmbedding, RotaryShape(query_heads~, kv_heads~, head_dimension~, ..)) =>
      (query_heads + 2 * kv_heads).to_int64() * head_dimension.to_int64()
    (CausalAttention, AttentionShape(query_heads~, head_dimension~, ..)) =>
      query_heads.to_int64() * head_dimension.to_int64()
    (OutputProjection | LanguageModelHead, ProjectionShape(output_width~, ..)) =>
      output_width.to_int64()
    (ResidualAdd, ElementwiseShape(width~)) => width.to_int64()
    (GatedMlp, GatedMlpShape(hidden_size~, ..)) => hidden_size.to_int64()
    _ => raise NumericBindingMismatch(operation.id())
  }
}

///|
fn validate_paged_v4_operand_set(
  model_plan : @plan.ModelPlan,
  operation : @plan.PlanOperation,
  binding : @catalog.KernelBinding,
  profile : PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  operands : ReadOnlyArray[AotOperand],
  maximum : Int,
) -> Unit raise LaunchContractError {
  if operands.is_empty() || operands.length() > maximum {
    raise OperandCountExceeded(operation=operation.id(), maximum~)
  }
  let expected = canonical_paged_v4_roles(model_plan, operation, binding)
  if operands.length() != expected.length() {
    raise InvalidOperand(operation.id(), MissingSemanticRole)
  }
  let execution = paged_v4_execution(model_plan, operation.id())
  for index, operand in operands {
    if operand.role != expected[index] {
      raise InvalidOperand(operation.id(), PositionalRoleMismatch)
    }
    match operand.role {
      RuntimeInput(role) =>
        if operand.byte_count != paged_runtime_bytes(role, profile) ||
          operand.alignment != PAGED_V3_INT32_ALIGNMENT {
          raise InvalidOperand(operation.id(), RuntimeInputSizeMismatch)
        }
      ExternalInput(TokenIds) =>
        if operand.byte_count !=
          profile.max_query_tokens.to_int64() * PAGED_V4_TOKEN_ID_BYTES ||
          operand.alignment != PAGED_V3_INT32_ALIGNMENT {
          raise InvalidOperand(operation.id(), NumericByteCountMismatch)
        }
      WeightInput(reference) => {
        let tensor = validate_paged_v4_parameter(
          model_plan,
          operation.id(),
          execution,
          reference,
        )
        if operand.byte_count != paged_v4_tensor_bytes(tensor, operation.id()) ||
          operand.alignment != PAGED_V4_VALUE_ALIGNMENT {
          raise InvalidOperand(operation.id(), NumericByteCountMismatch)
        }
      }
      WeightScaleInput(weight, scale) => {
        let tensor = paged_v4_scale_tensor(
          model_plan,
          operation.id(),
          weight,
          scale,
        )
        if operand.byte_count != paged_v4_tensor_bytes(tensor, operation.id()) ||
          operand.alignment != PAGED_V4_VALUE_ALIGNMENT {
          raise InvalidOperand(operation.id(), NumericByteCountMismatch)
        }
      }
      ActivationInput(reference) => {
        let expected_bytes = paged_v4_checked_value_bytes(
          profile.max_query_tokens,
          paged_v4_activation_input_width(operation, reference),
          paged_v4_compute_byte_width(
            execution.activation_input(),
            operation.id(),
          ),
          operation.id(),
        )
        if operand.byte_count != expected_bytes ||
          operand.alignment != PAGED_V4_VALUE_ALIGNMENT {
          raise InvalidOperand(operation.id(), NumericByteCountMismatch)
        }
      }
      ActivationOutput(reference) => {
        let expected_bytes = paged_v4_checked_value_bytes(
          profile.max_query_tokens,
          paged_v4_activation_output_width(operation, reference),
          paged_v4_compute_byte_width(execution.output(), operation.id()),
          operation.id(),
        )
        if operand.byte_count != expected_bytes ||
          operand.alignment != PAGED_V4_VALUE_ALIGNMENT {
          raise InvalidOperand(operation.id(), NumericByteCountMismatch)
        }
      }
      PersistentKvComponent(_, _) =>
        if operand.byte_count != layout.component_stride_bytes() ||
          operand.alignment != layout.segment_alignment_bytes() {
          raise InvalidOperand(operation.id(), PersistentKvMismatch)
        }
      Workspace =>
        if operand.byte_count != binding.workspace_bytes() ||
          operand.alignment != binding.workspace_alignment() {
          raise InvalidOperand(operation.id(), WorkspaceMismatch)
        }
    }
  }
}
