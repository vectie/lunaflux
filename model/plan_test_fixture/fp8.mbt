///|
fn synthetic_fp8_storage() -> @numeric_contract.TensorStorageContract {
  try! @numeric_contract.TensorStorageContract::new(
    dtype=@numeric_contract.StorageDType::f8_e4m3_finite(),
    layout=@numeric_contract.StorageLayout::dense_row_major_v1(),
    encoding=@numeric_contract.StorageEncoding::scaled_v1(),
    scale_granularity=@numeric_contract.ScaleGranularity::per_tensor(),
    zero_point=@numeric_contract.ZeroPointMode::absent(),
    codebook_layout=@numeric_contract.CodebookLayout::absent(),
    conversion=@numeric_contract.NumericConversion::quantized_v1(),
  )
}

///|
fn synthetic_f32_storage() -> @numeric_contract.TensorStorageContract {
  try! @numeric_contract.TensorStorageContract::new(
    dtype=@numeric_contract.StorageDType::f32(),
    layout=@numeric_contract.StorageLayout::dense_row_major_v1(),
    encoding=@numeric_contract.StorageEncoding::plain_v1(),
    scale_granularity=@numeric_contract.ScaleGranularity::absent(),
    zero_point=@numeric_contract.ZeroPointMode::absent(),
    codebook_layout=@numeric_contract.CodebookLayout::absent(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
  )
}

///|
fn selected_fp8_operation(
  selected : ArrayView[@plan.OperationId],
  id : @plan.OperationId,
) -> Bool {
  selected.contains(id)
}

///|
/// Builds a mixed BF16/finite-E4M3 W8A8 schema for synthetic graph tests.
///
/// FP8 operation weights receive one explicit scalar F32 metadata tensor each.
/// A parameter shared between BF16 and FP8 operations is rejected because one
/// tensor cannot have two storage contracts.
pub fn synthetic_fp8_w8a8_numeric_schema(
  operations : ArrayView[@plan.PlanOperation],
  fp8_operations : ArrayView[@plan.OperationId],
) -> @numeric_contract.ModelNumericSchema {
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
    let mode = if selected_fp8_operation(fp8_operations, operation.id()) {
      2
    } else {
      1
    }
    for input_index, input in operation.inputs() {
      match input {
        Tensor(reference) => {
          let index = reference.as_int()
          let shape = tensor_shape_for_input(operation, input_index)
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
  let fp8_storage = synthetic_fp8_storage()
  let bf16_storage = plain_bf16_storage()
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
          storage=fp8_storage,
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
  let scale_storage = synthetic_f32_storage()
  for scale_index in 0..<scale_count {
    tensors.push(
      try! @numeric_contract.NumericTensorSchema::new(
        ordinal=@numeric_contract.NumericTensorOrdinal::new(
          tensor_count + scale_index,
        ),
        role=@numeric_contract.NumericTensorRole::scale_metadata(),
        shape=@numeric_contract.NumericShape::new([]),
        storage=scale_storage,
      ),
    )
  }
  let bindings : Array[@numeric_contract.NumericOperationSchema] = []
  for operation in operations {
    let (has_activation, has_tensor, _) = operation_presence(operation)
    let fp8 = selected_fp8_operation(fp8_operations, operation.id())
    if fp8 && (!has_activation || !has_tensor) {
      abort("synthetic W8A8 operation lacks activation or tensor input")
    }
    let graph_activation = if has_activation {
      @numeric_contract.ComputeDType::bfloat16()
    } else {
      @numeric_contract.ComputeDType::absent()
    }
    let execution = try! @numeric_contract.OperationExecutionContract::new(
      activation_input=graph_activation,
      activation_compute=if fp8 {
        @numeric_contract.ComputeDType::f8_e4m3_finite()
      } else {
        graph_activation
      },
      activation_scale=if fp8 {
        @numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1()
      } else {
        @numeric_contract.ActivationScalePolicy::absent_exact_v1()
      },
      tensor_input=if has_tensor {
        if fp8 {
          @numeric_contract.ComputeDType::f8_e4m3_finite()
        } else {
          @numeric_contract.ComputeDType::bfloat16()
        }
      } else {
        @numeric_contract.ComputeDType::absent()
      },
      accumulator=@numeric_contract.AccumulatorDType::f32(),
      output=@numeric_contract.ComputeDType::bfloat16(),
      conversion=if fp8 {
        @numeric_contract.NumericConversion::quantized_v1()
      } else {
        @numeric_contract.NumericConversion::exact_v1()
      },
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
