///|
priv struct V2NumericSelection {
  operation_kind : @plan.OperationKind
  policy : @numeric_contract.ActivationScalePolicy
  scale_kind : Fp8StagedDynamicScaleKind
  workspace_bytes : Int64
}

///|
priv struct SelectedV2LaunchContract {
  source_version : Fp8V2LaunchContractSourceVersion
  entry_point : @catalog.AotKernelEntryPoint
  dimensions : @launch_contract.AotLaunchDimensions
  operands : ReadOnlyArray[@launch_contract.AotOperand]
  workspace_bytes : Int64
  workspace_alignment : Int64
}

///|
fn validate_v2_numeric_execution(
  plan : @plan.ModelPlan,
  entry : @numeric_capability_manifest.Fp8KernelManifestEntry,
) -> V2NumericSelection raise Fp8LaunchAbiError {
  let index = entry.operation().as_int()
  guard plan.operations().get(index) is Some(operation) &&
    operation.id() == entry.operation() &&
    plan.numeric_binding().operations().get(index) is Some(binding) &&
    binding.operation() == entry.operation() else {
    raise InvalidFp8LaunchAbi(ManifestEntry)
  }
  let execution = binding.execution()
  if execution.digest() != entry.execution_digest() {
    raise InvalidFp8LaunchAbi(ExecutionDigest)
  }
  if execution.activation_input() != @numeric_contract.ComputeDType::bfloat16() ||
    execution.activation_compute() !=
    @numeric_contract.ComputeDType::f8_e4m3_finite() ||
    execution.tensor_input() != @numeric_contract.ComputeDType::f8_e4m3_finite() ||
    execution.accumulator() != @numeric_contract.AccumulatorDType::f32() ||
    execution.output() != @numeric_contract.ComputeDType::bfloat16() ||
    execution.conversion() !=
    @numeric_contract.NumericConversion::quantized_v1() ||
    execution.order() !=
    @numeric_contract.OperationOrderPolicy::strict_declared_order_v1() {
    raise InvalidFp8LaunchAbi(UnsupportedScaleMode)
  }
  let policy = execution.activation_scale()
  let selection = if policy ==
    @numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1() {
    if !(operation.kind()
      is (QkvProjection | OutputProjection | LanguageModelHead)) {
      raise InvalidFp8LaunchAbi(OperationKind)
    }
    {
      operation_kind: operation.kind(),
      policy,
      scale_kind: SimpleExternalFiniteE4M3F32V2,
      workspace_bytes: 4L,
    }
  } else if policy ==
    @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_f32_v1() ||
    policy ==
    @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_bound_expf_f32_v2() {
    if operation.kind() != GatedMlp {
      raise InvalidFp8LaunchAbi(OperationKind)
    }
    {
      operation_kind: operation.kind(),
      policy,
      scale_kind: GatedExternalAndPostSiluFiniteE4M3F32V2,
      workspace_bytes: 8L,
    }
  } else {
    raise InvalidFp8LaunchAbi(ActivationScalePolicy)
  }
  if operation.workspace_bytes() != selection.workspace_bytes {
    raise InvalidFp8LaunchAbi(WorkspaceContract)
  }
  if entry.weights().is_empty() {
    raise InvalidFp8LaunchAbi(UnsupportedScaleMode)
  }
  for weight in entry.weights() {
    guard plan.tensor(weight.tensor()) is Some(parameter) &&
      parameter.scale() == Some(weight.scale()) &&
      parameter.storage().digest() == weight.storage_digest() &&
      plan.tensor(weight.scale().tensor()) is Some(scale) &&
      scale.storage().digest() == weight.scale_storage_digest() else {
      raise InvalidFp8LaunchAbi(NumericSchema)
    }
  }
  selection
}

///|
fn exact_v2_workspace_operand(
  operands : ReadOnlyArray[@launch_contract.AotOperand],
  expected_bytes : Int64,
  expected_alignment : Int64,
) -> Bool {
  let mut count = 0
  let mut exact = false
  for index, operand in operands {
    if operand.role() == Workspace {
      count += 1
      exact = index == operands.length() - 1 &&
        operand.byte_count() == expected_bytes &&
        operand.alignment() == expected_alignment
    }
  }
  count == 1 && exact
}

///|
fn exact_v2_weight_scale_operands(
  entry : @numeric_capability_manifest.Fp8KernelManifestEntry,
  operands : ReadOnlyArray[@launch_contract.AotOperand],
) -> Bool {
  let mut weight_index = 0
  let mut pending_scale = false
  for operand in operands {
    match operand.role() {
      WeightInput(weight) => {
        if pending_scale ||
          weight_index >= entry.weights().length() ||
          entry.weights()[weight_index].tensor() != weight {
          return false
        }
        pending_scale = true
      }
      WeightScaleInput(weight, scale) => {
        if !pending_scale ||
          weight_index >= entry.weights().length() ||
          entry.weights()[weight_index].tensor() != weight ||
          entry.weights()[weight_index].scale() != scale {
          return false
        }
        pending_scale = false
        weight_index += 1
      }
      _ => if pending_scale { return false }
    }
  }
  !pending_scale && weight_index == entry.weights().length()
}

///|
fn select_v2_paged_numeric_contract(
  plan : @plan.ModelPlan,
  manifest : @numeric_capability_manifest.Fp8KernelCapabilityManifest,
  launches : Fp8LaunchContractSource,
  entry : @numeric_capability_manifest.Fp8KernelManifestEntry,
  shape : Fp8ExactExecutionShape,
  workspace_bytes : Int64,
) -> SelectedV2LaunchContract raise Fp8LaunchAbiError {
  guard launches is PagedV4(contracts) else {
    raise InvalidFp8LaunchAbi(LaunchContract)
  }
  let selected = select_paged_v4_contract(
    plan, manifest, contracts, entry, shape,
  )
  guard contracts.contract(entry.profile(), entry.operation()) is Some(contract) else {
    raise InvalidFp8LaunchAbi(LaunchContract)
  }
  let operands = contract.operands()
  if selected.workspace_bytes != workspace_bytes ||
    selected.workspace_alignment != 4L ||
    !exact_v2_workspace_operand(operands, workspace_bytes, 4L) ||
    !exact_v2_weight_scale_operands(entry, operands) {
    raise InvalidFp8LaunchAbi(WorkspaceContract)
  }
  {
    source_version: PagedNumericV4,
    entry_point: selected.entry_point,
    dimensions: contract.dimensions(),
    operands,
    workspace_bytes: selected.workspace_bytes,
    workspace_alignment: selected.workspace_alignment,
  }
}

///|
fn derive_v2_stage_regions(
  policy : @numeric_contract.ActivationScalePolicy,
  workspace : Fp8LaunchStorageRegion,
) -> ReadOnlyArray[Fp8ActivationScaleStageRegion] raise Fp8LaunchAbiError {
  let count = policy.stage_count()
  if count.to_int64() != workspace.byte_length / 4L || count < 1 || count > 2 {
    raise InvalidFp8LaunchAbi(ActivationScalePolicy)
  }
  let result : Array[Fp8ActivationScaleStageRegion] = []
  for ordinal in 0..<count {
    guard policy.stage_at(ordinal) is Some(stage) else {
      raise InvalidFp8LaunchAbi(ActivationScaleStage)
    }
    result.push({
      stage,
      ordinal,
      region: {
        storage: workspace.storage,
        storage_bytes: workspace.storage_bytes,
        offset: workspace.offset + ordinal.to_int64() * 4L,
        byte_length: 4L,
        alignment: 4L,
      },
    })
  }
  ReadOnlyArray::from_array(result)
}

///|
fn validate_v2_workspace(
  scale_tensors : ReadOnlyArray[Fp8ScaleTensorStorageRegion],
  workspace : Fp8LaunchStorageRegion,
  expected_bytes : Int64,
  expected_alignment : Int64,
) -> Unit raise Fp8LaunchAbiError {
  if workspace.byte_length != expected_bytes ||
    workspace.alignment != expected_alignment ||
    expected_alignment != 4L {
    raise InvalidFp8LaunchAbi(WorkspaceContract)
  }
  for index, tensor in scale_tensors {
    if tensor.region.byte_length != 4L || tensor.region.alignment < 4L {
      raise InvalidFp8LaunchAbi(WorkspaceContract)
    }
    validate_distinct_regions(tensor.region, workspace)
    for previous in 0..<index {
      validate_distinct_regions(scale_tensors[previous].region, tensor.region)
    }
  }
}

///|
fn validate_v2_scale_operand_regions(
  operands : ReadOnlyArray[@launch_contract.AotOperand],
  scale_tensors : ReadOnlyArray[Fp8ScaleTensorStorageRegion],
) -> Unit raise Fp8LaunchAbiError {
  let mut scale_index = 0
  for operand in operands {
    guard operand.role() is WeightScaleInput(_, _) else { continue }
    if scale_index >= scale_tensors.length() ||
      scale_tensors[scale_index].region.byte_length != operand.byte_count() ||
      scale_tensors[scale_index].region.alignment < operand.alignment() {
      raise InvalidFp8LaunchAbi(BindingMismatch)
    }
    scale_index += 1
  }
  if scale_index != scale_tensors.length() {
    raise InvalidFp8LaunchAbi(BindingMismatch)
  }
}

///|
/// Admits the version-2 whole-Workspace FP8 launch ABI. The caller supplies
/// one exact 4- or 8-byte workspace region; ordered 4-byte stage cells are
/// derived from authenticated policy and never accepted independently.
pub fn admit_staged_dynamic_activation_scale_launch_v2(
  plan : @plan.ModelPlan,
  kernels : @numeric_capability_manifest.AdmittedFp8KernelArtifacts,
  launches : Fp8LaunchContractSource,
  operation~ : @plan.OperationId,
  shape~ : Fp8ExactExecutionShape,
  weight_scale_regions~ : ArrayView[Fp8LaunchStorageRegion],
  workspace~ : Fp8LaunchStorageRegion,
) -> Fp8StagedDynamicLaunchAbi raise Fp8LaunchAbiError {
  let manifest = kernels.manifest()
  if manifest.model_identity() != plan.identity() {
    raise InvalidFp8LaunchAbi(ModelIdentity)
  }
  if manifest.numeric_schema_digest() != plan.numeric_binding().digest() {
    raise InvalidFp8LaunchAbi(NumericSchema)
  }
  let entry = find_manifest_entry(manifest, operation, shape.profile())
  let numeric = validate_v2_numeric_execution(plan, entry)
  let scale_tensor_regions = bind_scale_tensor_regions(
    entry, weight_scale_regions,
  )
  let selected = select_v2_paged_numeric_contract(
    plan,
    manifest,
    launches,
    entry,
    shape,
    numeric.workspace_bytes,
  )
  validate_v2_workspace(
    scale_tensor_regions,
    workspace,
    numeric.workspace_bytes,
    4L,
  )
  validate_v2_scale_operand_regions(selected.operands, scale_tensor_regions)
  let stages = derive_v2_stage_regions(numeric.policy, workspace)
  let digest = encode_staged_abi_v2(
    plan,
    manifest,
    entry,
    selected,
    numeric.operation_kind,
    numeric.policy,
    shape,
    scale_tensor_regions,
    workspace,
    stages,
  )
  {
    version: Fp8LaunchAbiVersion::v2(),
    scope: shape.scope(),
    plan_identity: plan.identity(),
    numeric_schema: plan.numeric_binding().digest(),
    manifest_digest: manifest.digest(),
    target: manifest.target(),
    profile: entry.profile(),
    operation: entry.operation(),
    operation_kind: numeric.operation_kind,
    execution: entry.execution_digest(),
    module_digest: entry.module_digest(),
    entry_point: selected.entry_point,
    launch_source_version: selected.source_version,
    launch_dimensions: selected.dimensions,
    raw_operands: selected.operands,
    shape,
    policy: numeric.policy,
    scale_kind: numeric.scale_kind,
    weight_scales: entry.weights(),
    scale_tensor_regions,
    workspace,
    stages,
    digest,
  }
}
