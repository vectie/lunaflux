///|
/// Bounded policy-admission failures for the homogeneous dense-Llama finite
/// E4M3 W8A8 representation.
pub(all) suberror LlamaFp8W8A8PolicyError {
  TiedFp8EmbeddingsUnsupported
} derive(Debug, Eq)

///|
/// Opaque all-or-nothing policy for dense-Llama finite-E4M3 W8A8 plans.
///
/// Every eligible projection is FP8 with its own scalar F32 weight scale.
/// Embeddings and normalization parameters remain BF16. Activations for every
/// FP8 projection use the canonical dynamic per-tensor F32 scale policy.
#valtype
pub struct LlamaFp8W8A8Policy {
  priv version : Int
} derive(Debug, Eq)

///|
/// Opaque dense-Llama finite-E4M3 W8A8 v2 policy.
///
/// V2 authenticates the gated MLP's distinct external-input and post-SiLU
/// scales. A separate type prevents replay into a v1 builder.
#valtype
pub struct LlamaFp8W8A8V2Policy {
  priv version : Int
} derive(Debug, Eq)

///|
/// Admits the exact untied-embedding finite-E4M3 W8A8 v1 policy.
///
/// Tied embeddings fail closed because one shared matrix cannot simultaneously
/// satisfy the BF16 lookup and FP8 language-model-head storage contracts.
pub fn LlamaFp8W8A8Policy::new(
  tied_embeddings~ : Bool,
) -> LlamaFp8W8A8Policy raise LlamaFp8W8A8PolicyError {
  if tied_embeddings {
    raise TiedFp8EmbeddingsUnsupported
  }
  { version: 1 }
}

///|
/// Returns the canonical untied finite-E4M3 W8A8 v1 policy.
pub fn LlamaFp8W8A8Policy::untied_v1() -> LlamaFp8W8A8Policy {
  try! LlamaFp8W8A8Policy::new(tied_embeddings=false)
}

///|
/// Admits the exact untied-embedding finite-E4M3 W8A8 v2 policy.
pub fn LlamaFp8W8A8V2Policy::new(
  tied_embeddings~ : Bool,
) -> LlamaFp8W8A8V2Policy raise LlamaFp8W8A8PolicyError {
  if tied_embeddings {
    raise TiedFp8EmbeddingsUnsupported
  }
  { version: 2 }
}

///|
/// Returns the canonical untied finite-E4M3 W8A8 v2 policy.
pub fn LlamaFp8W8A8V2Policy::untied_v2() -> LlamaFp8W8A8V2Policy {
  try! LlamaFp8W8A8V2Policy::new(tied_embeddings=false)
}

///|
fn LlamaFp8W8A8Policy::validate(self : LlamaFp8W8A8Policy) -> Unit {
  if self.version != 1 {
    abort("invalid internal dense-Llama FP8 W8A8 policy")
  }
}

///|
fn LlamaFp8W8A8V2Policy::validate(self : LlamaFp8W8A8V2Policy) -> Unit {
  if self.version != 2 {
    abort("invalid internal dense-Llama FP8 W8A8 v2 policy")
  }
}

///|
fn fp8_llama_plain_storage(
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
fn fp8_llama_weight_storage() -> @numeric_contract.TensorStorageContract {
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
fn append_fp8_llama_bf16_parameter(
  tensors : Array[@numeric_contract.NumericTensorSchema],
  dimensions : ArrayView[Int64],
  storage : @numeric_contract.TensorStorageContract,
) -> Unit {
  tensors.push(
    try! @numeric_contract.NumericTensorSchema::new(
      ordinal=@numeric_contract.NumericTensorOrdinal::new(tensors.length()),
      role=@numeric_contract.NumericTensorRole::parameter(),
      shape=@numeric_contract.NumericShape::new(dimensions),
      storage~,
    ),
  )
}

///|
fn append_fp8_llama_quantized_parameter(
  tensors : Array[@numeric_contract.NumericTensorSchema],
  dimensions : ArrayView[Int64],
  storage : @numeric_contract.TensorStorageContract,
  scale_ordinal : Int,
) -> Unit {
  tensors.push(
    try! @numeric_contract.NumericTensorSchema::new(
      ordinal=@numeric_contract.NumericTensorOrdinal::new(tensors.length()),
      role=@numeric_contract.NumericTensorRole::parameter(),
      shape=@numeric_contract.NumericShape::new(dimensions),
      storage~,
      scale=@numeric_contract.ScaleTensorOrdinal::new(
        @numeric_contract.NumericTensorOrdinal::new(scale_ordinal),
      ),
    ),
  )
}

///|
fn append_fp8_llama_scale(
  tensors : Array[@numeric_contract.NumericTensorSchema],
) -> Unit {
  tensors.push(
    try! @numeric_contract.NumericTensorSchema::new(
      ordinal=@numeric_contract.NumericTensorOrdinal::new(tensors.length()),
      role=@numeric_contract.NumericTensorRole::scale_metadata(),
      shape=@numeric_contract.NumericShape::new([]),
      storage=fp8_llama_plain_storage(@numeric_contract.StorageDType::f32()),
    ),
  )
}

///|
fn fp8_llama_exact_execution(
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
fn fp8_llama_projection_execution(
  activation_scale : @numeric_contract.ActivationScalePolicy,
) -> @numeric_contract.OperationExecutionContract {
  try! @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::bfloat16(),
    activation_compute=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    activation_scale~,
    tensor_input=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::quantized_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
}

///|
fn append_fp8_llama_operation(
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
fn build_fp8_llama_numeric_schema(
  spec : @spec.LlamaModelSpec,
  compound_gated : Bool,
) -> @numeric_contract.ModelNumericSchema {
  let layer_count = spec.layer_count()
  let parameter_count = 3 + 9 * layer_count
  let scale_count = 1 + 7 * layer_count
  let tensors : Array[@numeric_contract.NumericTensorSchema] = []
  let bf16 = fp8_llama_plain_storage(@numeric_contract.StorageDType::bfloat16())
  let fp8 = fp8_llama_weight_storage()
  let hidden = Int64::from_int(spec.hidden_size())
  let intermediate = Int64::from_int(spec.intermediate_size())
  let vocabulary = Int64::from_int(spec.vocabulary_size())
  let kv_width = Int64::from_int(spec.kv_head_count() * spec.head_dimension())
  let mut next_scale = 0
  append_fp8_llama_bf16_parameter(tensors, [vocabulary, hidden], bf16)
  for _ in 0..<layer_count {
    append_fp8_llama_bf16_parameter(tensors, [hidden], bf16)
    append_fp8_llama_quantized_parameter(
      tensors,
      [hidden, hidden],
      fp8,
      parameter_count + next_scale,
    )
    next_scale += 1
    for _ in 0..<2 {
      append_fp8_llama_quantized_parameter(
        tensors,
        [kv_width, hidden],
        fp8,
        parameter_count + next_scale,
      )
      next_scale += 1
    }
    append_fp8_llama_quantized_parameter(
      tensors,
      [hidden, hidden],
      fp8,
      parameter_count + next_scale,
    )
    next_scale += 1
    append_fp8_llama_bf16_parameter(tensors, [hidden], bf16)
    for _ in 0..<2 {
      append_fp8_llama_quantized_parameter(
        tensors,
        [intermediate, hidden],
        fp8,
        parameter_count + next_scale,
      )
      next_scale += 1
    }
    append_fp8_llama_quantized_parameter(
      tensors,
      [hidden, intermediate],
      fp8,
      parameter_count + next_scale,
    )
    next_scale += 1
  }
  append_fp8_llama_bf16_parameter(tensors, [hidden], bf16)
  append_fp8_llama_quantized_parameter(
    tensors,
    [vocabulary, hidden],
    fp8,
    parameter_count + next_scale,
  )
  next_scale += 1
  if tensors.length() != parameter_count || next_scale != scale_count {
    abort("dense-Llama FP8 parameter count drift")
  }
  for _ in 0..<scale_count {
    append_fp8_llama_scale(tensors)
  }
  let operations : Array[@numeric_contract.NumericOperationSchema] = []
  append_fp8_llama_operation(operations, fp8_llama_exact_execution(false, true))
  let projection = fp8_llama_projection_execution(
    @numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1(),
  )
  let gated = if compound_gated {
    fp8_llama_projection_execution(
      @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_bound_expf_f32_v2(),
    )
  } else {
    projection
  }
  for _ in 0..<layer_count {
    append_fp8_llama_operation(
      operations,
      fp8_llama_exact_execution(true, true),
    )
    append_fp8_llama_operation(operations, projection)
    for _ in 0..<2 {
      append_fp8_llama_operation(
        operations,
        fp8_llama_exact_execution(true, false),
      )
    }
    append_fp8_llama_operation(operations, projection)
    append_fp8_llama_operation(
      operations,
      fp8_llama_exact_execution(true, false),
    )
    append_fp8_llama_operation(
      operations,
      fp8_llama_exact_execution(true, true),
    )
    append_fp8_llama_operation(operations, gated)
    append_fp8_llama_operation(
      operations,
      fp8_llama_exact_execution(true, false),
    )
  }
  append_fp8_llama_operation(operations, fp8_llama_exact_execution(true, true))
  append_fp8_llama_operation(operations, projection)
  if operations.length() != parameter_count ||
    tensors.length() != parameter_count + scale_count {
    abort("dense-Llama FP8 schema count drift")
  }
  try! @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors~,
    operations~,
  )
}

///|
/// Builds an immutable full-context dense-Llama plan with the complete finite
/// E4M3 W8A8 v1 numeric schema.
///
/// The result authenticates numeric identity and graph/schema correspondence.
/// It is not a kernel route, device admission, executable, or readiness claim.
pub fn build_finite_e4m3_fp8_w8a8_v1(
  metadata : @spec.LlamaModelMetadata,
  policy : LlamaFp8W8A8Policy,
) -> @plan.ModelPlan raise @plan.PlanValidationError {
  policy.validate()
  build_with_execution(
    metadata,
    StatelessFullContext,
    1,
    build_fp8_llama_numeric_schema(metadata.spec(), false),
    LegacyOperationWorkspaceV1,
  )
}

///|
/// Builds the paged-KV graph with the same exact finite-E4M3 W8A8 v1 schema.
/// The batch envelope remains part of the authenticated plan identity.
pub fn build_paged_finite_e4m3_fp8_w8a8_v1(
  metadata : @spec.LlamaModelMetadata,
  policy : LlamaFp8W8A8Policy,
  batch? : LlamaPagedBatchEnvelope = LlamaPagedBatchEnvelope::single_row(),
) -> @plan.ModelPlan raise @plan.PlanValidationError {
  policy.validate()
  build_with_execution(
    metadata,
    PagedKeyValue,
    batch.max_batch_rows(),
    build_fp8_llama_numeric_schema(metadata.spec(), false),
    LegacyOperationWorkspaceV1,
  )
}

///|
/// Builds the full-context dense-Llama finite-E4M3 W8A8 v2 plan.
///
/// Simple projections own one 4-byte dynamic-scale workspace. Every gated MLP
/// owns the compound two-stage policy and one exact 8-byte workspace. This is
/// inert model-plan evidence and grants no kernel or device readiness.
pub fn build_finite_e4m3_fp8_w8a8_v2(
  metadata : @spec.LlamaModelMetadata,
  policy : LlamaFp8W8A8V2Policy,
) -> @plan.ModelPlan raise @plan.PlanValidationError {
  policy.validate()
  build_with_execution(
    metadata,
    StatelessFullContext,
    1,
    build_fp8_llama_numeric_schema(metadata.spec(), true),
    Fp8DynamicActivationWorkspaceV2,
  )
}

///|
/// Builds the paged-KV dense-Llama finite-E4M3 W8A8 v2 plan with an exact
/// authenticated batch envelope and the same 4/8-byte workspace policy.
pub fn build_paged_finite_e4m3_fp8_w8a8_v2(
  metadata : @spec.LlamaModelMetadata,
  policy : LlamaFp8W8A8V2Policy,
  batch? : LlamaPagedBatchEnvelope = LlamaPagedBatchEnvelope::single_row(),
) -> @plan.ModelPlan raise @plan.PlanValidationError {
  policy.validate()
  build_with_execution(
    metadata,
    PagedKeyValue,
    batch.max_batch_rows(),
    build_fp8_llama_numeric_schema(metadata.spec(), true),
    Fp8DynamicActivationWorkspaceV2,
  )
}
