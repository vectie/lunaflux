///|
fn plain_storage(
  dtype : @numeric_contract.StorageDType,
) -> @numeric_contract.TensorStorageContract {
  try! @numeric_contract.TensorStorageContract::new(
    dtype~,
    layout=@numeric_contract.StorageLayout::dense_row_major_v1(),
    encoding=@numeric_contract.StorageEncoding::plain_v1(),
    scale_granularity=@numeric_contract.ScaleGranularity::absent(),
    zero_point=@numeric_contract.ZeroPointMode::absent(),
    codebook_layout=@numeric_contract.CodebookLayout::absent(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
  )
}

///|
fn exact_execution(
  dtype : @numeric_contract.ComputeDType,
  has_activation : Bool,
  has_tensor : Bool,
) -> @numeric_contract.OperationExecutionContract {
  try! @numeric_contract.OperationExecutionContract::new(
    activation_input=if has_activation {
      dtype
    } else {
      @numeric_contract.ComputeDType::absent()
    },
    activation_compute=if has_activation {
      dtype
    } else {
      @numeric_contract.ComputeDType::absent()
    },
    activation_scale=@numeric_contract.ActivationScalePolicy::absent_exact_v1(),
    tensor_input=if has_tensor {
      dtype
    } else {
      @numeric_contract.ComputeDType::absent()
    },
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=dtype,
    conversion=@numeric_contract.NumericConversion::exact_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
}

///|
fn bf16_tensor_execution() -> @numeric_contract.OperationExecutionContract {
  exact_execution(@numeric_contract.ComputeDType::bfloat16(), true, true)
}

///|
fn bf16_activation_execution() -> @numeric_contract.OperationExecutionContract {
  exact_execution(@numeric_contract.ComputeDType::bfloat16(), true, false)
}

///|
fn bf16_parameter_execution() -> @numeric_contract.OperationExecutionContract {
  exact_execution(@numeric_contract.ComputeDType::bfloat16(), false, true)
}

///|
fn f32_tensor_execution() -> @numeric_contract.OperationExecutionContract {
  exact_execution(@numeric_contract.ComputeDType::f32(), true, true)
}

///|
fn bf16_f32_to_f32_tensor_execution() -> @numeric_contract.OperationExecutionContract {
  try! @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::bfloat16(),
    activation_compute=@numeric_contract.ComputeDType::f32(),
    activation_scale=@numeric_contract.ActivationScalePolicy::absent_exact_v1(),
    tensor_input=@numeric_contract.ComputeDType::f32(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::f32(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
}

///|
/// BF16 activation storage with an exact F32 compute seam and one BF16 output
/// conversion. KDA's F32 log-decay and recurrent-state side inputs are bound by
/// the launch ABI because the generic operation contract has one activation
/// input dtype.
fn bf16_f32_activation_execution() -> @numeric_contract.OperationExecutionContract {
  try! @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::bfloat16(),
    activation_compute=@numeric_contract.ComputeDType::f32(),
    activation_scale=@numeric_contract.ActivationScalePolicy::absent_exact_v1(),
    tensor_input=@numeric_contract.ComputeDType::absent(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
}

///|
fn bf16_f32_tensor_execution() -> @numeric_contract.OperationExecutionContract {
  try! @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::bfloat16(),
    activation_compute=@numeric_contract.ComputeDType::f32(),
    activation_scale=@numeric_contract.ActivationScalePolicy::absent_exact_v1(),
    tensor_input=@numeric_contract.ComputeDType::bfloat16(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
}

///|
/// The official router consumes an F32 activation view while retaining a
/// logical BF16 gate matrix. This is distinct from the following F32
/// correction-bias and top-k operation.
fn f32_router_projection_execution() -> @numeric_contract.OperationExecutionContract {
  try! @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::f32(),
    activation_compute=@numeric_contract.ComputeDType::f32(),
    activation_scale=@numeric_contract.ActivationScalePolicy::absent_exact_v1(),
    tensor_input=@numeric_contract.ComputeDType::bfloat16(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::f32(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
}
