///|
fn selected_i8_weight_only_operation_v1(
  selected : ArrayView[@plan.OperationId],
  id : @plan.OperationId,
) -> Bool {
  selected.contains(id)
}

///|
fn i8_weight_only_operation_kind_allowed_v1(kind : @plan.OperationKind) -> Bool {
  kind is (QkvProjection | OutputProjection | GatedMlp | LanguageModelHead)
}

///|
fn validate_synthetic_i8_weight_only_selection_v1(
  operations : ArrayView[@plan.PlanOperation],
  selected : ArrayView[@plan.OperationId],
) -> Unit {
  for selected_index, id in selected {
    for previous_index in 0..<selected_index {
      if selected[previous_index] == id {
        abort("synthetic I8 v1 selection repeats one operation ID")
      }
    }
    let mut matches = 0
    let mut allowed = false
    for operation in operations {
      if operation.id() == id {
        matches += 1
        allowed = i8_weight_only_operation_kind_allowed_v1(operation.kind())
      }
    }
    if matches != 1 {
      abort("synthetic I8 v1 selected operation ID must exist exactly once")
    }
    if !allowed {
      abort("synthetic I8 v1 selected operation kind is not reviewed")
    }
  }
}

///|
fn i8_weight_only_operation_execution() -> @numeric_contract.OperationExecutionContract {
  @numeric_contract.OperationExecutionContract::symmetric_i8_weight_only_v1()
}

///|
/// Builds a mixed BF16/symmetric-I8 weight-only schema for synthetic graph
/// tests. Selection is operation-atomic: every tensor operand of a selected
/// operation is I8 and owns one distinct `[out_channels]` plain-F32 scale.
/// Selection is closed to QKV projection, output projection, gated MLP, and
/// language-model head v1 operations. Every requested ID must be unique and
/// identify exactly one operation; future kinds are never inferred eligible
/// from activation or tensor shape.
pub fn synthetic_symmetric_i8_weight_only_numeric_schema_v1(
  operations : ArrayView[@plan.PlanOperation],
  i8_operations : ArrayView[@plan.OperationId],
) -> @numeric_contract.ModelNumericSchema {
  validate_synthetic_i8_weight_only_selection_v1(operations, i8_operations)
  let mut tensor_count = 0
  for operation in operations {
    let (_, _, operation_tensor_count) = operation_presence(operation)
    tensor_count = tensor_count.max(operation_tensor_count)
  }
  let tensor_shapes : Array[@numeric_contract.NumericShape?] = []
  let tensor_modes : Array[Int] = []
  for _ in 0..<tensor_count {
    tensor_shapes.push(None)
    tensor_modes.push(0)
  }
  for operation in operations {
    let selected = selected_i8_weight_only_operation_v1(
      i8_operations,
      operation.id(),
    )
    let (has_activation, has_tensor, _) = operation_presence(operation)
    if selected && (!has_activation || !has_tensor) {
      abort("synthetic I8 weight-only operation lacks activation or weight")
    }
    let mode = if selected { 2 } else { 1 }
    for input_index, input in operation.inputs() {
      match input {
        Tensor(reference) => {
          let index = reference.as_int()
          let shape = tensor_shape_for_input(operation, input_index)
          if selected && shape.rank() != 2 {
            abort("synthetic I8 weight-only tensor is not a matrix")
          }
          match tensor_shapes[index] {
            Some(previous) =>
              if previous != shape || tensor_modes[index] != mode {
                abort("synthetic tensor has incompatible numeric uses")
              }
            None => {
              tensor_shapes[index] = Some(shape)
              tensor_modes[index] = mode
            }
          }
        }
        Activation(_) | ModelInput(_) => ()
      }
    }
  }
  let tensors : Array[@numeric_contract.NumericTensorSchema] = []
  let bf16_storage = plain_bf16_storage()
  let i8_storage = @numeric_contract.TensorStorageContract::symmetric_i8_weight_only_v1()
  let mut scale_count = 0
  for index in 0..<tensor_count {
    guard tensor_shapes[index] is Some(shape) else {
      abort("synthetic numeric fixture declared an unused tensor")
    }
    if tensor_modes[index] == 2 {
      tensors.push(
        try! @numeric_contract.NumericTensorSchema::new(
          ordinal=@numeric_contract.NumericTensorOrdinal::new(index),
          role=@numeric_contract.NumericTensorRole::parameter(),
          shape~,
          storage=i8_storage,
          scale=@numeric_contract.ScaleTensorOrdinal::new(
            @numeric_contract.NumericTensorOrdinal::new(
              tensor_count + scale_count,
            ),
          ),
        ),
      )
      scale_count += 1
    } else {
      tensors.push(
        try! @numeric_contract.NumericTensorSchema::new(
          ordinal=@numeric_contract.NumericTensorOrdinal::new(index),
          role=@numeric_contract.NumericTensorRole::parameter(),
          shape~,
          storage=bf16_storage,
        ),
      )
    }
  }
  let scale_storage = @numeric_contract.TensorStorageContract::symmetric_i8_weight_only_scale_v1()
  let mut next_scale = 0
  for index in 0..<tensor_count {
    if tensor_modes[index] == 2 {
      guard tensor_shapes[index] is Some(shape) &&
        shape.dimension(0) is Some(output_channels) else {
        abort("synthetic I8 weight-only matrix has no output axis")
      }
      tensors.push(
        try! @numeric_contract.NumericTensorSchema::new(
          ordinal=@numeric_contract.NumericTensorOrdinal::new(
            tensor_count + next_scale,
          ),
          role=@numeric_contract.NumericTensorRole::scale_metadata(),
          shape=@numeric_contract.NumericShape::new([output_channels]),
          storage=scale_storage,
        ),
      )
      next_scale += 1
    }
  }
  let bindings : Array[@numeric_contract.NumericOperationSchema] = []
  for operation in operations {
    let (has_activation, has_tensor, _) = operation_presence(operation)
    let selected = selected_i8_weight_only_operation_v1(
      i8_operations,
      operation.id(),
    )
    let execution = if selected {
      i8_weight_only_operation_execution()
    } else {
      try! @numeric_contract.OperationExecutionContract::new(
        activation_input=if has_activation {
          @numeric_contract.ComputeDType::bfloat16()
        } else {
          @numeric_contract.ComputeDType::absent()
        },
        activation_compute=if has_activation {
          @numeric_contract.ComputeDType::bfloat16()
        } else {
          @numeric_contract.ComputeDType::absent()
        },
        activation_scale=@numeric_contract.ActivationScalePolicy::absent_exact_v1(),
        tensor_input=if has_tensor {
          @numeric_contract.ComputeDType::bfloat16()
        } else {
          @numeric_contract.ComputeDType::absent()
        },
        accumulator=@numeric_contract.AccumulatorDType::f32(),
        output=@numeric_contract.ComputeDType::bfloat16(),
        conversion=@numeric_contract.NumericConversion::exact_v1(),
        order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
      )
    }
    let ordinal = try! @numeric_contract.NumericOperationOrdinal::new(
      operation.id().as_int(),
    )
    bindings.push(
      @numeric_contract.NumericOperationSchema::new(ordinal~, execution~),
    )
  }
  try! @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors~,
    operations=bindings,
  )
}
