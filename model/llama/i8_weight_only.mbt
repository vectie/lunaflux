///|
/// Bounded policy-admission failures for dense-Llama symmetric-I8 weight-only
/// plan construction.
pub(all) suberror LlamaI8WeightOnlyPolicyError {
  TiedEmbeddingsUnsupported
} derive(Debug, Eq)

///|
/// Opaque all-or-nothing policy for the first production dense-Llama
/// weight-only format.
///
/// This policy has no tensor or operation selection surface. Every eligible
/// projection is quantized, while every ineligible parameter remains BF16.
#valtype
pub struct LlamaI8WeightOnlyPolicy {
  priv version : Int
} derive(Debug, Eq)

///|
/// Admits the exact untied-embedding symmetric-I8 weight-only v1 policy.
///
/// Tied embeddings fail closed because the shared matrix would be required to
/// satisfy both the BF16 embedding contract and the I8 language-model-head
/// contract. This constructor performs no quantization or model allocation.
pub fn LlamaI8WeightOnlyPolicy::new(
  tied_embeddings~ : Bool,
) -> LlamaI8WeightOnlyPolicy raise LlamaI8WeightOnlyPolicyError {
  if tied_embeddings {
    raise TiedEmbeddingsUnsupported
  }
  { version: 1 }
}

///|
/// Returns the selected-model untied v1 policy.
pub fn LlamaI8WeightOnlyPolicy::untied_v1() -> LlamaI8WeightOnlyPolicy {
  try! LlamaI8WeightOnlyPolicy::new(tied_embeddings=false)
}

///|
fn LlamaI8WeightOnlyPolicy::validate(self : LlamaI8WeightOnlyPolicy) -> Unit {
  if self.version != 1 {
    abort("invalid internal dense-Llama I8 weight-only policy")
  }
}

///|
fn i8_llama_plain_bf16_storage() -> @numeric_contract.TensorStorageContract {
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
fn append_i8_llama_bf16_parameter(
  tensors : Array[@numeric_contract.NumericTensorSchema],
  dimensions : ArrayView[Int64],
) -> Unit {
  tensors.push(
    try! @numeric_contract.NumericTensorSchema::new(
      ordinal=@numeric_contract.NumericTensorOrdinal::new(tensors.length()),
      role=@numeric_contract.NumericTensorRole::parameter(),
      shape=@numeric_contract.NumericShape::new(dimensions),
      storage=i8_llama_plain_bf16_storage(),
    ),
  )
}

///|
fn append_i8_llama_quantized_parameter(
  tensors : Array[@numeric_contract.NumericTensorSchema],
  dimensions : ArrayView[Int64],
  scale_ordinal : Int,
) -> Unit {
  tensors.push(
    try! @numeric_contract.NumericTensorSchema::new(
      ordinal=@numeric_contract.NumericTensorOrdinal::new(tensors.length()),
      role=@numeric_contract.NumericTensorRole::parameter(),
      shape=@numeric_contract.NumericShape::new(dimensions),
      storage=@numeric_contract.TensorStorageContract::symmetric_i8_weight_only_v1(),
      scale=@numeric_contract.ScaleTensorOrdinal::new(
        @numeric_contract.NumericTensorOrdinal::new(scale_ordinal),
      ),
    ),
  )
}

///|
fn append_i8_llama_scale(
  tensors : Array[@numeric_contract.NumericTensorSchema],
  output_channels : Int64,
) -> Unit {
  tensors.push(
    try! @numeric_contract.NumericTensorSchema::new(
      ordinal=@numeric_contract.NumericTensorOrdinal::new(tensors.length()),
      role=@numeric_contract.NumericTensorRole::scale_metadata(),
      shape=@numeric_contract.NumericShape::new([output_channels]),
      storage=@numeric_contract.TensorStorageContract::symmetric_i8_weight_only_scale_v1(),
    ),
  )
}

///|
fn i8_llama_bf16_execution(
  has_activation : Bool,
  has_tensor : Bool,
) -> @numeric_contract.OperationExecutionContract {
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

///|
fn append_i8_llama_operation(
  operations : Array[@numeric_contract.NumericOperationSchema],
  execution : @numeric_contract.OperationExecutionContract,
) -> Unit {
  operations.push(
    @numeric_contract.NumericOperationSchema::new(
      ordinal=try! @numeric_contract.NumericOperationOrdinal::new(
        operations.length(),
      ),
      execution~,
    ),
  )
}

///|
fn build_i8_llama_numeric_schema(
  spec : @spec.LlamaModelSpec,
) -> @numeric_contract.ModelNumericSchema {
  let layer_count = spec.layer_count()
  let parameter_count = 3 + 9 * layer_count
  let scale_count = 1 + 7 * layer_count
  let tensors : Array[@numeric_contract.NumericTensorSchema] = []
  let hidden = Int64::from_int(spec.hidden_size())
  let intermediate = Int64::from_int(spec.intermediate_size())
  let vocabulary = Int64::from_int(spec.vocabulary_size())
  let kv_width = Int64::from_int(spec.kv_head_count() * spec.head_dimension())
  let mut next_scale = 0
  append_i8_llama_bf16_parameter(tensors, [vocabulary, hidden])
  for _ in 0..<layer_count {
    append_i8_llama_bf16_parameter(tensors, [hidden])
    append_i8_llama_quantized_parameter(
      tensors,
      [hidden, hidden],
      parameter_count + next_scale,
    )
    next_scale += 1
    append_i8_llama_quantized_parameter(
      tensors,
      [kv_width, hidden],
      parameter_count + next_scale,
    )
    next_scale += 1
    append_i8_llama_quantized_parameter(
      tensors,
      [kv_width, hidden],
      parameter_count + next_scale,
    )
    next_scale += 1
    append_i8_llama_quantized_parameter(
      tensors,
      [hidden, hidden],
      parameter_count + next_scale,
    )
    next_scale += 1
    append_i8_llama_bf16_parameter(tensors, [hidden])
    append_i8_llama_quantized_parameter(
      tensors,
      [intermediate, hidden],
      parameter_count + next_scale,
    )
    next_scale += 1
    append_i8_llama_quantized_parameter(
      tensors,
      [intermediate, hidden],
      parameter_count + next_scale,
    )
    next_scale += 1
    append_i8_llama_quantized_parameter(
      tensors,
      [hidden, intermediate],
      parameter_count + next_scale,
    )
    next_scale += 1
  }
  append_i8_llama_bf16_parameter(tensors, [hidden])
  append_i8_llama_quantized_parameter(
    tensors,
    [vocabulary, hidden],
    parameter_count + next_scale,
  )
  next_scale += 1
  if tensors.length() != parameter_count || next_scale != scale_count {
    abort("dense-Llama I8 parameter count drift")
  }
  for _ in 0..<layer_count {
    append_i8_llama_scale(tensors, hidden)
    append_i8_llama_scale(tensors, kv_width)
    append_i8_llama_scale(tensors, kv_width)
    append_i8_llama_scale(tensors, hidden)
    append_i8_llama_scale(tensors, intermediate)
    append_i8_llama_scale(tensors, intermediate)
    append_i8_llama_scale(tensors, hidden)
  }
  append_i8_llama_scale(tensors, vocabulary)
  if tensors.length() != parameter_count + scale_count {
    abort("dense-Llama I8 scale count drift")
  }
  let operations : Array[@numeric_contract.NumericOperationSchema] = []
  append_i8_llama_operation(operations, i8_llama_bf16_execution(false, true))
  let i8_execution = @numeric_contract.OperationExecutionContract::symmetric_i8_weight_only_v1()
  for _ in 0..<layer_count {
    append_i8_llama_operation(operations, i8_llama_bf16_execution(true, true))
    append_i8_llama_operation(operations, i8_execution)
    append_i8_llama_operation(operations, i8_llama_bf16_execution(true, false))
    append_i8_llama_operation(operations, i8_llama_bf16_execution(true, false))
    append_i8_llama_operation(operations, i8_execution)
    append_i8_llama_operation(operations, i8_llama_bf16_execution(true, false))
    append_i8_llama_operation(operations, i8_llama_bf16_execution(true, true))
    append_i8_llama_operation(operations, i8_execution)
    append_i8_llama_operation(operations, i8_llama_bf16_execution(true, false))
  }
  append_i8_llama_operation(operations, i8_llama_bf16_execution(true, true))
  append_i8_llama_operation(operations, i8_execution)
  if operations.length() != 3 + 9 * layer_count {
    abort("dense-Llama I8 operation count drift")
  }
  try! @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors~,
    operations~,
  )
}

///|
/// Builds the immutable full-context dense-Llama plan with the exact
/// symmetric-I8 weight-only v1 representation.
///
/// Eligible QKV, output-projection, gated-MLP, and language-model-head
/// matrices are all I8 and each owns a distinct F32 per-output-channel scale.
/// Embeddings, normalization weights, activations, outputs, and KV remain
/// BF16. No runtime quantization or partial selection is performed.
pub fn build_symmetric_i8_weight_only_v1(
  metadata : @spec.LlamaModelMetadata,
  policy : LlamaI8WeightOnlyPolicy,
) -> @plan.ModelPlan raise @plan.PlanValidationError {
  policy.validate()
  build_with_execution(
    metadata,
    StatelessFullContext,
    1,
    build_i8_llama_numeric_schema(metadata.spec()),
    LegacyOperationWorkspaceV1,
  )
}

///|
/// Builds the paged-KV dense-Llama graph with the same exact I8 storage and
/// execution policy. The optional batch envelope remains part of plan identity.
pub fn build_paged_symmetric_i8_weight_only_v1(
  metadata : @spec.LlamaModelMetadata,
  policy : LlamaI8WeightOnlyPolicy,
  batch? : LlamaPagedBatchEnvelope = LlamaPagedBatchEnvelope::single_row(),
) -> @plan.ModelPlan raise @plan.PlanValidationError {
  policy.validate()
  build_with_execution(
    metadata,
    PagedKeyValue,
    batch.max_batch_rows(),
    build_i8_llama_numeric_schema(metadata.spec()),
    LegacyOperationWorkspaceV1,
  )
}
