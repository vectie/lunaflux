///|
fn llama_bf16_storage() -> @numeric_contract.TensorStorageContract {
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
fn append_llama_numeric_tensor(
  tensors : Array[@numeric_contract.NumericTensorSchema],
  dimensions : ArrayView[Int64],
) -> Unit {
  tensors.push(
    try! @numeric_contract.NumericTensorSchema::new(
      ordinal=@numeric_contract.NumericTensorOrdinal::new(tensors.length()),
      role=@numeric_contract.NumericTensorRole::parameter(),
      shape=@numeric_contract.NumericShape::new(dimensions),
      storage=llama_bf16_storage(),
    ),
  )
}

///|
fn append_llama_numeric_operation(
  operations : Array[@numeric_contract.NumericOperationSchema],
  has_activation : Bool,
  has_tensor : Bool,
) -> Unit {
  let ordinal = try! @numeric_contract.NumericOperationOrdinal::new(
    operations.length(),
  )
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
  operations.push(
    @numeric_contract.NumericOperationSchema::new(ordinal~, execution~),
  )
}

///|
/// Derives the complete reference-free BF16 tensor and operation schema used
/// by the dense Llama plan builder and its authenticated plan identity.
pub fn LlamaModelSpec::numeric_schema(
  self : LlamaModelSpec,
) -> @numeric_contract.ModelNumericSchema {
  let tensors : Array[@numeric_contract.NumericTensorSchema] = []
  let hidden = Int64::from_int(self.hidden_size())
  let intermediate = Int64::from_int(self.intermediate_size())
  let vocabulary = Int64::from_int(self.vocabulary_size())
  let kv_width = Int64::from_int(self.kv_head_count() * self.head_dimension())
  append_llama_numeric_tensor(tensors, [vocabulary, hidden])
  for _ in 0..<self.layer_count() {
    append_llama_numeric_tensor(tensors, [hidden])
    append_llama_numeric_tensor(tensors, [hidden, hidden])
    append_llama_numeric_tensor(tensors, [kv_width, hidden])
    append_llama_numeric_tensor(tensors, [kv_width, hidden])
    append_llama_numeric_tensor(tensors, [hidden, hidden])
    append_llama_numeric_tensor(tensors, [hidden])
    append_llama_numeric_tensor(tensors, [intermediate, hidden])
    append_llama_numeric_tensor(tensors, [intermediate, hidden])
    append_llama_numeric_tensor(tensors, [hidden, intermediate])
  }
  append_llama_numeric_tensor(tensors, [hidden])
  append_llama_numeric_tensor(tensors, [vocabulary, hidden])
  let operations : Array[@numeric_contract.NumericOperationSchema] = []
  append_llama_numeric_operation(operations, false, true)
  for _ in 0..<self.layer_count() {
    append_llama_numeric_operation(operations, true, true)
    append_llama_numeric_operation(operations, true, true)
    append_llama_numeric_operation(operations, true, false)
    append_llama_numeric_operation(operations, true, false)
    append_llama_numeric_operation(operations, true, true)
    append_llama_numeric_operation(operations, true, false)
    append_llama_numeric_operation(operations, true, true)
    append_llama_numeric_operation(operations, true, true)
    append_llama_numeric_operation(operations, true, false)
  }
  append_llama_numeric_operation(operations, true, true)
  append_llama_numeric_operation(operations, true, true)
  try! @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors~,
    operations~,
  )
}
