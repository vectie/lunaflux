///|
fn exact_production_bf16_storage(
  storage : @numeric_contract.TensorStorageContract,
) -> Bool {
  storage.dtype() == @numeric_contract.StorageDType::bfloat16() &&
  storage.encoding() == @numeric_contract.StorageEncoding::plain_v1() &&
  storage.scale_granularity() == @numeric_contract.ScaleGranularity::absent() &&
  storage.zero_point() == @numeric_contract.ZeroPointMode::absent() &&
  storage.codebook_layout() == @numeric_contract.CodebookLayout::absent() &&
  storage.conversion() == @numeric_contract.NumericConversion::exact_v1()
}

///|
fn production_execution_is_fp8(
  execution : @numeric_contract.OperationExecutionContract,
) -> Bool {
  let fp8 = @numeric_contract.ComputeDType::f8_e4m3_finite()
  execution.activation_input() == fp8 ||
  execution.activation_compute() == fp8 ||
  execution.tensor_input() == fp8 ||
  execution.output() == fp8
}

///|
fn exact_production_bf16_execution(
  execution : @numeric_contract.OperationExecutionContract,
) -> Bool {
  let absent = @numeric_contract.ComputeDType::absent()
  let bf16 = @numeric_contract.ComputeDType::bfloat16()
  let activation = execution.activation_input()
  (activation == absent || activation == bf16) &&
  execution.activation_compute() == activation &&
  execution.activation_scale() ==
  @numeric_contract.ActivationScalePolicy::absent_exact_v1() &&
  (execution.tensor_input() == absent || execution.tensor_input() == bf16) &&
  execution.accumulator() == @numeric_contract.AccumulatorDType::f32() &&
  execution.output() == bf16 &&
  execution.conversion() == @numeric_contract.NumericConversion::exact_v1() &&
  execution.order() ==
  @numeric_contract.OperationOrderPolicy::strict_declared_order_v1()
}

///|
fn validate_production_tensor_policy(
  plan : @plan.ModelPlan,
) -> (Int, Int) raise I8InertKernelCapabilityAdmissionError {
  let tensors = plan.numeric_binding().tensors()
  let scale_uses : Array[Int] = Array::make(tensors.length(), 0)
  let mut i8_parameter_count = 0
  let mut scale_tensor_count = 0
  for index, tensor in tensors {
    if tensor.reference().as_int() != index {
      raise I8InertKernelCapabilityRejected(ProductionTensorStorage)
    }
    if tensor.role() == ParameterTensor {
      if exact_i8_weight_storage(tensor.storage()) {
        guard tensor.scale() is Some(scale) &&
          tensor.zero_point() is None &&
          tensor.codebook() is None &&
          tensors.get(scale.tensor().as_int()) is Some(scale_tensor) &&
          scale_tensor.role() == ScaleMetadataTensor &&
          exact_i8_scale_storage(scale_tensor.storage()) &&
          tensor.shape().rank() == 2 &&
          tensor.shape().dimension(0) is Some(output_channels) &&
          scale_tensor.shape().rank() == 1 &&
          scale_tensor.shape().dimension(0) == Some(output_channels) else {
          raise I8InertKernelCapabilityRejected(ProductionTensorStorage)
        }
        let scale_index = scale.tensor().as_int()
        if scale_uses[scale_index] != 0 {
          raise I8InertKernelCapabilityRejected(ProductionScaleCoverage)
        }
        scale_uses[scale_index] = 1
        i8_parameter_count += 1
      } else if !exact_production_bf16_storage(tensor.storage()) ||
        tensor.scale() is Some(_) ||
        tensor.zero_point() is Some(_) ||
        tensor.codebook() is Some(_) {
        raise I8InertKernelCapabilityRejected(ProductionTensorStorage)
      }
    } else if tensor.role() == ScaleMetadataTensor {
      if !exact_i8_scale_storage(tensor.storage()) ||
        tensor.scale() is Some(_) ||
        tensor.zero_point() is Some(_) ||
        tensor.codebook() is Some(_) {
        raise I8InertKernelCapabilityRejected(ProductionTensorStorage)
      }
      scale_tensor_count += 1
    } else {
      // Affine zero-point and codebook metadata are deliberately outside the
      // first production symmetric-I8 policy.
      raise I8InertKernelCapabilityRejected(ProductionTensorStorage)
    }
  }
  for index, tensor in tensors {
    if tensor.role() == ScaleMetadataTensor && scale_uses[index] != 1 {
      raise I8InertKernelCapabilityRejected(ProductionScaleCoverage)
    }
  }
  if i8_parameter_count <= 0 || scale_tensor_count != i8_parameter_count {
    raise I8InertKernelCapabilityRejected(ProductionScaleCoverage)
  }
  (i8_parameter_count, scale_tensor_count)
}

///|
fn validate_production_operation_policy(
  plan : @plan.ModelPlan,
) -> Unit raise I8InertKernelCapabilityAdmissionError {
  let operations = plan.operations()
  let numeric = plan.numeric_binding().operations()
  if operations.length() != numeric.length() {
    raise I8InertKernelCapabilityRejected(NumericSchema)
  }
  for index, operation in operations {
    let execution = numeric[index].execution()
    if production_execution_is_fp8(execution) {
      raise I8InertKernelCapabilityRejected(MixedFp8Execution)
    }
    let mut has_parameter = false
    let mut has_i8 = false
    let mut has_bf16 = false
    for input in operation.inputs() {
      if input is Tensor(reference) && plan.tensor(reference) is Some(tensor) {
        has_parameter = true
        if exact_i8_weight_storage(tensor.storage()) {
          has_i8 = true
        } else if exact_production_bf16_storage(tensor.storage()) {
          has_bf16 = true
        } else {
          raise I8InertKernelCapabilityRejected(ProductionTensorStorage)
        }
      }
    }
    let eligible = i8_operation_kind_allowed(operation.kind())
    if eligible {
      // Policy v1 is one reviewed full-coverage profile. Arbitrary partial
      // quantization would need a distinct policy version and accuracy proof.
      if !has_parameter ||
        !has_i8 ||
        has_bf16 ||
        !exact_i8_weight_only_execution(execution) {
        raise I8InertKernelCapabilityRejected(ProductionExecution)
      }
    } else if has_i8 ||
      exact_i8_weight_only_execution(execution) ||
      !exact_production_bf16_execution(execution) {
      raise I8InertKernelCapabilityRejected(ProductionExecution)
    }
  }
}

///|
fn require_production_kind(
  operations : ReadOnlyArray[@plan.PlanOperation],
  index : Int,
  kind : @plan.OperationKind,
  layer : @plan.KvLayerId?,
) -> Unit raise I8InertKernelCapabilityAdmissionError {
  guard operations.get(index) is Some(operation) &&
    operation.id().as_int() == index &&
    operation.kind() == kind &&
    operation.kv_layer() == layer else {
    raise I8InertKernelCapabilityRejected(ProductionGraph)
  }
}

///|
fn validate_production_llama_graph(
  plan : @plan.ModelPlan,
) -> Unit raise I8InertKernelCapabilityAdmissionError {
  let operations = plan.operations()
  let layer_count = plan.kv_cache_geometry().layer_count()
  let mut index = 0
  require_production_kind(operations, index, Embedding, None)
  index += 1
  for layer_index in 0..<layer_count {
    let layer = @plan.KvLayerId::new(layer_index) catch {
      _ => raise I8InertKernelCapabilityRejected(ProductionGraph)
    }
    require_production_kind(operations, index, RmsNorm, None)
    index += 1
    require_production_kind(operations, index, QkvProjection, None)
    index += 1
    require_production_kind(operations, index, RotaryEmbedding, Some(layer))
    index += 1
    require_production_kind(operations, index, CausalAttention, Some(layer))
    index += 1
    require_production_kind(operations, index, OutputProjection, None)
    index += 1
    require_production_kind(operations, index, ResidualAdd, None)
    index += 1
    require_production_kind(operations, index, RmsNorm, None)
    index += 1
    require_production_kind(operations, index, GatedMlp, None)
    index += 1
    require_production_kind(operations, index, ResidualAdd, None)
    index += 1
  }
  require_production_kind(operations, index, RmsNorm, None)
  index += 1
  require_production_kind(operations, index, LanguageModelHead, None)
  index += 1
  if index != operations.length() {
    raise I8InertKernelCapabilityRejected(ProductionGraph)
  }
}

///|
/// Admits the complete initial production Llama symmetric-I8 policy.
///
/// The returned value is a pure pre-hardware authorization. The catalog-only
/// admission is rederived inside this function so callers cannot substitute a
/// weaker admission. FP8 tensors or execution contracts, affine zero points,
/// codebooks, orphan scales, shared scales, and unsupported I8 operation kinds
/// all fail before artifact, filesystem, loader, or device work can begin.
pub fn admit_production_llama_symmetric_i8_weight_only_v1(
  plan : @plan.ModelPlan,
  device_numeric : @device.DeviceNumericCapability,
  resolved_v4 : @catalog.ResolvedKernelCatalog,
  limits : I8InertKernelCapabilityAdmissionLimits,
) -> I8ProductionPolicyAdmission raise I8InertKernelCapabilityAdmissionError {
  let admission = admit_catalog_only_symmetric_i8_weight_only_v1(
    plan, device_numeric, resolved_v4, limits,
  )
  validate_production_llama_graph(plan)
  validate_production_operation_policy(plan)
  let (i8_parameter_count, scale_tensor_count) = validate_production_tensor_policy(
    plan,
  )
  { admission, i8_parameter_count, scale_tensor_count }
}

///|
/// Performs the complete model-only production-I8 policy check without
/// minting authority. Manifest loaders use this before opening a filesystem
/// capability so FP8, affine, codebook, orphan-scale, and mixed-execution
/// requests fail before any external work.
pub fn preflight_production_llama_symmetric_i8_weight_only_v1(
  plan : @plan.ModelPlan,
) -> Unit raise I8InertKernelCapabilityAdmissionError {
  if plan.kv_execution() != PagedKeyValue {
    raise I8InertKernelCapabilityRejected(ExecutionScope)
  }
  validate_production_llama_graph(plan)
  validate_production_operation_policy(plan)
  ignore(validate_production_tensor_policy(plan))
}

///|
pub fn I8ProductionPolicyAdmission::catalog_admission(
  self : I8ProductionPolicyAdmission,
) -> I8InertKernelCapabilityAdmission {
  self.admission
}

///|
pub fn I8ProductionPolicyAdmission::model_identity(
  self : I8ProductionPolicyAdmission,
) -> @spec.ModelIdentity {
  self.admission.model_identity()
}

///|
pub fn I8ProductionPolicyAdmission::numeric_schema_digest(
  self : I8ProductionPolicyAdmission,
) -> @numeric_contract.ModelNumericSchemaDigest {
  self.admission.numeric_schema_digest()
}

///|
pub fn I8ProductionPolicyAdmission::device_numeric_digest(
  self : I8ProductionPolicyAdmission,
) -> @device.DeviceNumericCapabilityDigest {
  self.admission.device_numeric_digest()
}

///|
pub fn I8ProductionPolicyAdmission::digest(
  self : I8ProductionPolicyAdmission,
) -> I8InertKernelCapabilityAdmissionDigest {
  // The complete schema digest is already in the catalog admission. The
  // production policy has one exact deterministic interpretation of it.
  self.admission.digest()
}

///|
pub fn I8ProductionPolicyAdmission::i8_parameter_count(
  self : I8ProductionPolicyAdmission,
) -> Int {
  self.i8_parameter_count
}

///|
pub fn I8ProductionPolicyAdmission::scale_tensor_count(
  self : I8ProductionPolicyAdmission,
) -> Int {
  self.scale_tensor_count
}
