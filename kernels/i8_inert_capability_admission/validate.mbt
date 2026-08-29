///|
fn exact_i8_weight_only_execution(
  execution : @numeric_contract.OperationExecutionContract,
) -> Bool {
  execution ==
  @numeric_contract.OperationExecutionContract::symmetric_i8_weight_only_v1()
}

///|
fn exact_i8_weight_storage(
  storage : @numeric_contract.TensorStorageContract,
) -> Bool {
  storage ==
  @numeric_contract.TensorStorageContract::symmetric_i8_weight_only_v1()
}

///|
fn exact_i8_scale_storage(
  storage : @numeric_contract.TensorStorageContract,
) -> Bool {
  storage ==
  @numeric_contract.TensorStorageContract::symmetric_i8_weight_only_scale_v1()
}

///|
fn i8_operation_kind_allowed(kind : @plan.OperationKind) -> Bool {
  kind is (QkvProjection | OutputProjection | GatedMlp | LanguageModelHead)
}

///|
fn operation_has_i8_tensor(
  plan : @plan.ModelPlan,
  operation : @plan.PlanOperation,
) -> Bool {
  for input in operation.inputs() {
    match input {
      Tensor(reference) =>
        match plan.numeric_binding().tensors().get(reference.as_int()) {
          Some(tensor) =>
            if tensor.storage().dtype() == @numeric_contract.StorageDType::i8() {
              return true
            }
          None => ()
        }
      ModelInput(_) | Activation(_) => ()
    }
  }
  false
}

///|
fn validate_i8_catalog_binding(
  operation : @plan.PlanOperation,
  execution : @numeric_contract.OperationExecutionContract,
  binding : @catalog.KernelBinding,
) -> @catalog.AotKernelEntryPoint raise I8InertKernelCapabilityAdmissionError {
  if binding.operation_id() != operation.id() ||
    binding.capability() != operation.required_capability() {
    raise I8InertKernelCapabilityRejected(BindingOrder)
  }
  if binding.semantic_version() != @catalog.KernelSemanticVersion::v4() ||
    binding.operation_execution_digest() != Some(execution.digest()) {
    raise I8InertKernelCapabilityRejected(OperationExecution)
  }
  guard binding.implementation() is ContentAddressedAot(family) &&
    binding.aot_entry_point() is Some(entry_point) &&
    entry_point.family() == family else {
    raise I8InertKernelCapabilityRejected(AotBinding)
  }
  if binding.workspace_bytes() < operation.workspace_bytes() ||
    binding.workspace_alignment() <= 0L {
    raise I8InertKernelCapabilityRejected(Workspace)
  }
  entry_point
}

///|
fn validate_i8_weight_reference(
  plan : @plan.ModelPlan,
  reference : @plan.TensorRef,
) -> Unit raise I8InertKernelCapabilityAdmissionError {
  guard plan.numeric_binding().tensors().get(reference.as_int()) is Some(weight) &&
    weight.reference() == reference &&
    weight.role() == ParameterTensor &&
    exact_i8_weight_storage(weight.storage()) &&
    weight.shape().rank() == 2 &&
    weight.shape().dimension(0) is Some(output_channels) &&
    weight.scale() is Some(scale_reference) else {
    raise I8InertKernelCapabilityRejected(MixedWeightStorage)
  }
  let scale_index = scale_reference.tensor().as_int()
  guard plan.numeric_binding().tensors().get(scale_index) is Some(scale) &&
    scale.reference() == scale_reference.tensor() &&
    scale.role() == ScaleMetadataTensor &&
    exact_i8_scale_storage(scale.storage()) &&
    scale.shape().rank() == 1 &&
    scale.shape().dimension(0) == Some(output_channels) &&
    scale.shape().element_count() == output_channels else {
    raise I8InertKernelCapabilityRejected(ScaleBinding)
  }
}

///|
fn count_i8_weight_references(
  plan : @plan.ModelPlan,
  operation : @plan.PlanOperation,
) -> Int raise I8InertKernelCapabilityAdmissionError {
  let mut count = 0
  for input in operation.inputs() {
    match input {
      Tensor(reference) => {
        validate_i8_weight_reference(plan, reference)
        count += 1
      }
      ModelInput(_) | Activation(_) => ()
    }
  }
  if count == 0 {
    raise I8InertKernelCapabilityRejected(MixedWeightStorage)
  }
  count
}

///|
fn validate_i8_scale_uniqueness(
  plan : @plan.ModelPlan,
  operation : @plan.PlanOperation,
  scale_seen : Array[Bool],
) -> Unit raise I8InertKernelCapabilityAdmissionError {
  for input in operation.inputs() {
    match input {
      Tensor(reference) => {
        validate_i8_weight_reference(plan, reference)
        guard plan.numeric_binding().tensors().get(reference.as_int())
          is Some(weight) &&
          weight.scale() is Some(scale_reference) else {
          raise I8InertKernelCapabilityRejected(ScaleBinding)
        }
        let scale_index = scale_reference.tensor().as_int()
        guard scale_seen.get(scale_index) is Some(false) else {
          if scale_seen.get(scale_index) is Some(true) {
            raise I8InertKernelCapabilityRejected(DuplicateScale)
          }
          raise I8InertKernelCapabilityRejected(ScaleBinding)
        }
        scale_seen[scale_index] = true
      }
      ModelInput(_) | Activation(_) => ()
    }
  }
}

///|
fn collect_i8_weight_evidence(
  plan : @plan.ModelPlan,
  operation : @plan.PlanOperation,
) -> ReadOnlyArray[I8InertWeightEvidence] raise I8InertKernelCapabilityAdmissionError {
  let weights : Array[I8InertWeightEvidence] = []
  for input in operation.inputs() {
    match input {
      Tensor(reference) => {
        validate_i8_weight_reference(plan, reference)
        guard plan.numeric_binding().tensors().get(reference.as_int())
          is Some(weight) &&
          weight.shape().dimension(0) is Some(output_channels) &&
          weight.scale() is Some(scale_reference) &&
          plan
          .numeric_binding()
          .tensors()
          .get(scale_reference.tensor().as_int())
          is Some(scale) else {
          raise I8InertKernelCapabilityRejected(ScaleBinding)
        }
        weights.push({
          tensor: reference,
          storage: weight.storage().digest(),
          shape: weight.shape(),
          scale: scale_reference,
          scale_storage: scale.storage().digest(),
          scale_shape: scale.shape(),
          output_channel_count: output_channels,
        })
      }
      ModelInput(_) | Activation(_) => ()
    }
  }
  ReadOnlyArray::from_array(weights)
}
