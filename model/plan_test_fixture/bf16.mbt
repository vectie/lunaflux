///|
fn plain_bf16_storage() -> @numeric_contract.TensorStorageContract {
  try! @numeric_contract.TensorStorageContract::new(
    dtype=@numeric_contract.StorageDType::bfloat16(),
    layout=@numeric_contract.StorageLayout::dense_row_major_v1(),
    encoding=@numeric_contract.StorageEncoding::plain_v1(),
    scale_granularity=@numeric_contract.ScaleGranularity::absent(),
    zero_point=@numeric_contract.ZeroPointMode::absent(),
    codebook_layout=@numeric_contract.CodebookLayout::absent(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
  )
}

///|
fn operation_presence(operation : @plan.PlanOperation) -> (Bool, Bool, Int) {
  let mut has_activation = false
  let mut has_tensor = false
  let mut tensor_count = 0
  for input in operation.inputs() {
    match input {
      Activation(_) => has_activation = true
      Tensor(reference) => {
        has_tensor = true
        tensor_count = tensor_count.max(reference.as_int() + 1)
      }
      ModelInput(_) => ()
    }
  }
  (has_activation, has_tensor, tensor_count)
}

///|
fn tensor_shape_for_input(
  operation : @plan.PlanOperation,
  input_index : Int,
) -> @numeric_contract.NumericShape {
  let dimensions : Array[Int64] = match
    (operation.kind(), operation.shape(), input_index) {
    (Embedding, EmbeddingShape(vocabulary_size~, width~), 1) =>
      [Int64::from_int(vocabulary_size), Int64::from_int(width)]
    (RmsNorm, NormalizationShape(width~, ..), 1) => [Int64::from_int(width)]
    (QkvProjection, ProjectionShape(input_width~, ..), 1) =>
      [Int64::from_int(input_width), Int64::from_int(input_width)]
    (QkvProjection, ProjectionShape(input_width~, output_width~), 2 | 3) =>
      [
        Int64::from_int((output_width - input_width) / 2),
        Int64::from_int(input_width),
      ]
    (
      OutputProjection
      | LanguageModelHead,
      ProjectionShape(input_width~, output_width~),
      1,
    ) => [Int64::from_int(output_width), Int64::from_int(input_width)]
    (GatedMlp, GatedMlpShape(hidden_size~, intermediate_size~), 1 | 2) =>
      [Int64::from_int(intermediate_size), Int64::from_int(hidden_size)]
    (GatedMlp, GatedMlpShape(hidden_size~, intermediate_size~), 3) =>
      [Int64::from_int(hidden_size), Int64::from_int(intermediate_size)]
    // Invalid-role fixtures must reach the plan's earlier structural gate.
    // Every valid tensor position matches one exact shape above.
    _ => []
  }
  try! @numeric_contract.NumericShape::new(dimensions)
}

///|
/// Builds a complete BF16 numeric schema for an intentionally synthetic test
/// graph. Stored tensors derive exact semantic shapes from their operation
/// positions; family builders still declare their own complete real schema.
pub fn synthetic_bf16_numeric_schema(
  operations : ArrayView[@plan.PlanOperation],
) -> @numeric_contract.ModelNumericSchema {
  let mut tensor_count = 0
  for operation in operations {
    let (_, _, operation_tensor_count) = operation_presence(operation)
    tensor_count = tensor_count.max(operation_tensor_count)
  }
  let tensor_shapes : Array[@numeric_contract.NumericShape?] = []
  for _ in 0..<tensor_count {
    tensor_shapes.push(None)
  }
  for operation in operations {
    for input_index, input in operation.inputs() {
      match input {
        Tensor(reference) => {
          let shape = tensor_shape_for_input(operation, input_index)
          match tensor_shapes[reference.as_int()] {
            Some(previous) =>
              if previous != shape {
                abort("one synthetic tensor has incompatible semantic shapes")
              }
            None => tensor_shapes[reference.as_int()] = Some(shape)
          }
        }
        Activation(_) | ModelInput(_) => ()
      }
    }
  }
  let tensors : Array[@numeric_contract.NumericTensorSchema] = []
  let storage = plain_bf16_storage()
  for index in 0..<tensor_count {
    guard tensor_shapes[index] is Some(shape) else {
      abort("synthetic numeric fixture declared an unused tensor")
    }
    tensors.push(
      try! @numeric_contract.NumericTensorSchema::new(
        ordinal=@numeric_contract.NumericTensorOrdinal::new(index),
        role=@numeric_contract.NumericTensorRole::parameter(),
        shape~,
        storage~,
      ),
    )
  }
  let bindings : Array[@numeric_contract.NumericOperationSchema] = []
  for operation in operations {
    let (has_activation, has_tensor, _) = operation_presence(operation)
    let execution = try! @numeric_contract.OperationExecutionContract::new(
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
