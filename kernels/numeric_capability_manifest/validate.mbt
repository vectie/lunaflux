///|
fn exact_fp8_w8a8_execution(
  operation : @plan.PlanOperation,
  execution : @numeric_contract.OperationExecutionContract,
) -> Bool {
  let policy = execution.activation_scale()
  let scale_policy_matches = (
      policy ==
      @numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1() &&
      operation.kind()
      is (QkvProjection | OutputProjection | GatedMlp | LanguageModelHead)
    ) ||
    (
      policy ==
      @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_f32_v1() &&
      operation.kind() == GatedMlp &&
      operation.workspace_bytes() == 8L
    ) ||
    (
      policy ==
      @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_bound_expf_f32_v2() &&
      operation.kind() == GatedMlp &&
      operation.workspace_bytes() == 8L
    )
  execution.activation_input() == @numeric_contract.ComputeDType::bfloat16() &&
  execution.activation_compute() ==
  @numeric_contract.ComputeDType::f8_e4m3_finite() &&
  scale_policy_matches &&
  execution.tensor_input() == @numeric_contract.ComputeDType::f8_e4m3_finite() &&
  execution.accumulator() == @numeric_contract.AccumulatorDType::f32() &&
  execution.output() == @numeric_contract.ComputeDType::bfloat16() &&
  execution.conversion() == @numeric_contract.NumericConversion::quantized_v1() &&
  execution.order() ==
  @numeric_contract.OperationOrderPolicy::strict_declared_order_v1()
}

///|
fn mentions_fp8_execution(
  execution : @numeric_contract.OperationExecutionContract,
) -> Bool {
  execution.activation_compute() ==
  @numeric_contract.ComputeDType::f8_e4m3_finite() ||
  execution.tensor_input() == @numeric_contract.ComputeDType::f8_e4m3_finite() ||
  execution.activation_scale() ==
  @numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1() ||
  execution.activation_scale() ==
  @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_f32_v1() ||
  execution.activation_scale() ==
  @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_bound_expf_f32_v2()
}

///|
fn exact_fp8_weight_storage(
  storage : @numeric_contract.TensorStorageContract,
) -> Bool {
  storage.dtype() == @numeric_contract.StorageDType::f8_e4m3_finite() &&
  storage.layout() == @numeric_contract.StorageLayout::dense_row_major_v1() &&
  storage.encoding() == @numeric_contract.StorageEncoding::scaled_v1() &&
  storage.scale_granularity() ==
  @numeric_contract.ScaleGranularity::per_tensor() &&
  storage.zero_point() == @numeric_contract.ZeroPointMode::absent() &&
  storage.codebook_layout() == @numeric_contract.CodebookLayout::absent() &&
  storage.conversion() == @numeric_contract.NumericConversion::quantized_v1()
}

///|
fn exact_f32_scale_storage(
  storage : @numeric_contract.TensorStorageContract,
) -> Bool {
  storage.dtype() == @numeric_contract.StorageDType::f32() &&
  storage.layout() == @numeric_contract.StorageLayout::dense_row_major_v1() &&
  storage.encoding() == @numeric_contract.StorageEncoding::plain_v1() &&
  storage.scale_granularity() == @numeric_contract.ScaleGranularity::absent() &&
  storage.zero_point() == @numeric_contract.ZeroPointMode::absent() &&
  storage.codebook_layout() == @numeric_contract.CodebookLayout::absent() &&
  storage.conversion() == @numeric_contract.NumericConversion::exact_v1()
}

///|
fn fp8_weight_bindings(
  plan : @plan.ModelPlan,
  operation : @plan.PlanOperation,
) -> ReadOnlyArray[Fp8KernelWeightBinding] raise Fp8KernelManifestError {
  let weights : Array[Fp8KernelWeightBinding] = []
  for input in operation.inputs() {
    match input {
      Tensor(reference) => {
        guard plan.numeric_binding().tensors().get(reference.as_int())
          is Some(weight) &&
          weight.reference() == reference &&
          weight.role() == ParameterTensor &&
          exact_fp8_weight_storage(weight.storage()) &&
          weight.scale() is Some(scale) else {
          raise InvalidFp8KernelManifest(ExecutionContract)
        }
        let scale_reference = scale.tensor()
        guard plan.numeric_binding().tensors().get(scale_reference.as_int())
          is Some(scale_tensor) &&
          scale_tensor.reference() == scale_reference &&
          scale_tensor.role() == ScaleMetadataTensor &&
          exact_f32_scale_storage(scale_tensor.storage()) else {
          raise InvalidFp8KernelManifest(ExecutionContract)
        }
        weights.push({
          tensor: reference,
          storage: weight.storage().digest(),
          scale,
          scale_storage: scale_tensor.storage().digest(),
        })
      }
      ModelInput(_) | Activation(_) => ()
    }
  }
  if weights.is_empty() {
    raise InvalidFp8KernelManifest(ExecutionContract)
  }
  ReadOnlyArray::from_array(weights)
}

///|
fn validate_base(
  plan : @plan.ModelPlan,
  device_numeric : @device.DeviceNumericCapability,
  target : @catalog.DeviceTarget,
  resolved : @catalog.ResolvedKernelCatalog,
  artifacts : @artifact.KernelArtifactBundle,
  limits : Fp8KernelManifestLimits,
) -> Array[ExpectedFp8Operation] raise Fp8KernelManifestError {
  if device_numeric.version() != @device.DeviceNumericCapabilityVersion::v1() ||
    device_numeric.feature() != FiniteFp8E4M3W8A8F32AccumulateBf16OutputV1 {
    raise InvalidFp8KernelManifest(DeviceCapability)
  }
  if !target.matches_numeric_capability(device_numeric) ||
    resolved.device_target() != target ||
    artifacts.device_target() != target {
    raise InvalidFp8KernelManifest(DeviceTarget)
  }
  if resolved.model_identity() != plan.identity() ||
    artifacts.model_identity() != plan.identity() {
    raise InvalidFp8KernelManifest(ModelIdentity)
  }
  if resolved.catalog_version() != artifacts.catalog_version() {
    raise InvalidFp8KernelManifest(CatalogVersion)
  }
  if resolved.bindings().length() != plan.operations().length() ||
    plan.numeric_binding().operations().length() != plan.operations().length() {
    raise InvalidFp8KernelManifest(BindingOrder)
  }
  let expected : Array[ExpectedFp8Operation] = []
  for index, operation in plan.operations() {
    let numeric = plan.numeric_binding().operations()[index]
    let binding = resolved.bindings()[index]
    if numeric.operation() != operation.id() ||
      binding.operation_id() != operation.id() {
      raise InvalidFp8KernelManifest(BindingOrder)
    }
    let execution = numeric.execution()
    if mentions_fp8_execution(execution) {
      if !exact_fp8_w8a8_execution(operation, execution) {
        raise InvalidFp8KernelManifest(ExecutionContract)
      }
      match binding.implementation() {
        VendorBf16GemmImplicitHeuristicV1 =>
          raise InvalidFp8KernelManifest(VendorOrBf16Fallback)
        ContentAddressedAot(family) =>
          expected.push({
            operation: operation.id(),
            execution: execution.digest(),
            family,
            weights: fp8_weight_bindings(plan, operation),
          })
      }
    }
  }
  if expected.is_empty() {
    raise InvalidFp8KernelManifest(MissingFp8Operation)
  }
  if expected.length() > limits.max_entries {
    raise InvalidFp8KernelManifest(Limit)
  }
  expected
}

///|
fn artifact_module_index(
  artifacts : @artifact.KernelArtifactBundle,
  entry_point : @catalog.AotKernelEntryPoint,
) -> Int raise Fp8KernelManifestError {
  guard artifacts.entry_point(entry_point) is Some(entry) else {
    raise InvalidFp8KernelManifest(ArtifactBundle)
  }
  let module_index = entry.module_index()
  guard artifacts.modules().get(module_index) is Some(artifact_module) &&
    artifact_module.digest() == entry_point.family().module_digest() else {
    raise InvalidFp8KernelManifest(ArtifactBundle)
  }
  module_index
}

///|
fn validate_claim(
  claim : Fp8KernelManifestClaim,
  profile : @launch_contract.KernelProfileId,
  operation : ExpectedFp8Operation,
  entry_point : @catalog.AotKernelEntryPoint,
  artifacts : @artifact.KernelArtifactBundle,
) -> Fp8KernelManifestEntry raise Fp8KernelManifestError {
  if claim.profile != profile || claim.operation != operation.operation {
    raise InvalidFp8KernelManifest(ClaimOrder)
  }
  if claim.execution != operation.execution {
    raise InvalidFp8KernelManifest(ClaimExecution)
  }
  if claim.entry_point != entry_point ||
    entry_point.family() != operation.family {
    raise InvalidFp8KernelManifest(ClaimEntryPoint)
  }
  let module_index = artifact_module_index(artifacts, entry_point)
  {
    profile,
    operation: operation.operation,
    execution: operation.execution,
    entry_point,
    module_digest: entry_point.family().module_digest(),
    artifact_module_index: module_index,
    weights: operation.weights,
  }
}

///|
fn publish_admission(
  scope : Fp8KernelExecutionScope,
  plan : @plan.ModelPlan,
  device_numeric : @device.DeviceNumericCapability,
  target : @catalog.DeviceTarget,
  catalog_version : @catalog.CatalogVersion,
  entries : Array[Fp8KernelManifestEntry],
  artifacts : @artifact.KernelArtifactBundle,
  limits : Fp8KernelManifestLimits,
) -> AdmittedFp8KernelArtifacts raise Fp8KernelManifestError {
  let owned = ReadOnlyArray::from_array(entries)
  let canonical = encode_fp8_kernel_manifest(
    scope,
    plan,
    device_numeric,
    target,
    catalog_version,
    owned,
    limits.max_canonical_bytes,
  )
  let manifest = {
    version: Fp8KernelCapabilityManifestVersion::v1(),
    scope,
    model_identity: plan.identity(),
    numeric_schema: plan.numeric_binding().digest(),
    device_numeric: device_numeric.digest(),
    target,
    catalog_version,
    entries: owned,
    canonical,
    digest: Fp8KernelCapabilityManifestDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
  { manifest, artifacts }
}
