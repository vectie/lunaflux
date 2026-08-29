///|
fn numeric_shape_is_1d(
  shape : @numeric_contract.NumericShape,
  first : Int,
) -> Bool {
  shape.rank() == 1 && shape.dimension(0) == Some(Int64::from_int(first))
}

///|
fn numeric_shape_is_2d(
  shape : @numeric_contract.NumericShape,
  first : Int,
  second : Int,
) -> Bool {
  shape.rank() == 2 &&
  shape.dimension(0) == Some(Int64::from_int(first)) &&
  shape.dimension(1) == Some(Int64::from_int(second))
}

///|
fn valid_numeric_tensor_shape(
  operation : PlanOperation,
  input_index : Int,
  shape : @numeric_contract.NumericShape,
  constraints : ShapeConstraints,
) -> Bool {
  let hidden = constraints.hidden_size()
  let intermediate = constraints.intermediate_size()
  let vocabulary = constraints.vocabulary_size()
  let query_width = constraints.query_width()
  let kv_width = constraints.kv_width()
  match (operation.kind(), input_index) {
    (Embedding, 1) => numeric_shape_is_2d(shape, vocabulary, hidden)
    (RmsNorm, 1) => numeric_shape_is_1d(shape, hidden)
    (QkvProjection, 1) => numeric_shape_is_2d(shape, query_width, hidden)
    (QkvProjection, 2 | 3) => numeric_shape_is_2d(shape, kv_width, hidden)
    (QkRmsNorm, 1 | 2) =>
      numeric_shape_is_1d(shape, constraints.head_dimension())
    (OutputProjection, 1) => numeric_shape_is_2d(shape, hidden, query_width)
    (GatedMlp, 1 | 2) => numeric_shape_is_2d(shape, intermediate, hidden)
    (GatedMlp, 3) => numeric_shape_is_2d(shape, hidden, intermediate)
    (LanguageModelHead, 1) => numeric_shape_is_2d(shape, vocabulary, hidden)
    _ => false
  }
}

///|
fn validate_numeric_tensor_operand(
  operation : PlanOperation,
  input_index : Int,
  reference : TensorRef,
  execution : @numeric_contract.OperationExecutionContract,
  tensors : ReadOnlyArray[PlanTensor],
  constraints : ShapeConstraints,
  used_parameters : Array[Bool],
) -> Unit raise PlanValidationError {
  guard tensors.get(reference.as_int()) is Some(tensor) &&
    tensor.reference() == reference else {
    raise InvalidGraph(MissingTensor)
  }
  if tensor.role() != ParameterTensor {
    raise InvalidGraph(TensorRoleMismatch)
  }
  if !execution.accepts_storage(tensor.storage()) {
    raise InvalidGraph(NumericTensorType)
  }
  if !valid_numeric_tensor_shape(
      operation,
      input_index,
      tensor.shape(),
      constraints,
    ) {
    raise InvalidGraph(NumericTensorShape)
  }
  used_parameters[reference.as_int()] = true
}

///|
fn valid_activation_scale_workspace(
  operation : PlanOperation,
  execution : @numeric_contract.OperationExecutionContract,
  compound_plan : Bool,
) -> Bool {
  let policy = execution.activation_scale()
  if policy ==
    @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_f32_v1() ||
    policy ==
    @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_bound_expf_f32_v2() {
    return operation.kind() == GatedMlp && operation.workspace_bytes() == 8L
  }
  if policy ==
    @numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1() {
    if compound_plan {
      return match operation.kind() {
        QkvProjection | OutputProjection | LanguageModelHead =>
          operation.workspace_bytes() == 4L
        _ => false
      }
    }
    // V1 admitted this policy independently of implementation scratch size.
    // Preserve those plan bytes; the v2 builder supplies the stronger 4-byte
    // simple-projection requirement without reinterpreting legacy plans.
    return true
  }
  true
}

///|
fn validate_operation_numeric(
  operation : PlanOperation,
  execution : @numeric_contract.OperationExecutionContract,
  tensors : ReadOnlyArray[PlanTensor],
  produced : ArrayView[@numeric_contract.ComputeDType],
  constraints : ShapeConstraints,
  used_parameters : Array[Bool],
  compound_plan : Bool,
) -> Unit raise PlanValidationError {
  if !valid_activation_scale_workspace(operation, execution, compound_plan) {
    raise InvalidGraph(NumericOperationBinding)
  }
  let mut has_activation_input = false
  let mut has_tensor_input = false
  for input_index, input in operation.inputs() {
    match input {
      Activation(reference) => {
        has_activation_input = true
        guard produced.get(reference.as_int()) is Some(dtype) &&
          dtype == execution.activation_input() else {
          raise InvalidGraph(NumericActivationType)
        }
      }
      Tensor(reference) => {
        has_tensor_input = true
        validate_numeric_tensor_operand(
          operation, input_index, reference, execution, tensors, constraints, used_parameters,
        )
      }
      ModelInput(_) => ()
    }
  }
  let activation_declared = execution.activation_input() !=
    @numeric_contract.ComputeDType::absent()
  let tensor_declared = execution.tensor_input() !=
    @numeric_contract.ComputeDType::absent()
  if has_activation_input != activation_declared ||
    has_tensor_input != tensor_declared {
    raise InvalidGraph(NumericOperationBinding)
  }
}

///|
fn mark_numeric_metadata(
  referenced : Array[Bool],
  reference : TensorRef,
) -> Unit {
  referenced[reference.as_int()] = true
}

///|
fn validate_complete_numeric_tensor_use(
  tensors : ReadOnlyArray[PlanTensor],
  used_parameters : ArrayView[Bool],
) -> Unit raise PlanValidationError {
  let referenced_metadata : Array[Bool] = []
  for _ in 0..<tensors.length() {
    referenced_metadata.push(false)
  }
  for index, tensor in tensors {
    if tensor.role() == ParameterTensor && used_parameters[index] {
      match tensor.scale() {
        Some(reference) =>
          mark_numeric_metadata(referenced_metadata, reference.tensor())
        None => ()
      }
      match tensor.zero_point() {
        Some(reference) =>
          mark_numeric_metadata(referenced_metadata, reference.tensor())
        None => ()
      }
      match tensor.codebook() {
        Some(reference) =>
          mark_numeric_metadata(referenced_metadata, reference.tensor())
        None => ()
      }
    }
  }
  for index, tensor in tensors {
    if tensor.role() == ParameterTensor {
      if !used_parameters[index] {
        raise InvalidGraph(UnusedParameterTensor)
      }
    } else if !referenced_metadata[index] {
      raise InvalidGraph(UnreferencedNumericMetadata)
    }
  }
}
