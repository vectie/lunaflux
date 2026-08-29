///|
fn find_manifest_entry(
  manifest : @numeric_capability_manifest.Fp8KernelCapabilityManifest,
  operation : @plan.OperationId,
  profile : @launch_contract.KernelProfileId?,
) -> @numeric_capability_manifest.Fp8KernelManifestEntry raise Fp8LaunchAbiError {
  let mut found : @numeric_capability_manifest.Fp8KernelManifestEntry? = None
  for entry in manifest.entries() {
    if entry.operation() == operation &&
      (profile is None || profile == Some(entry.profile())) {
      if found is Some(_) {
        raise InvalidFp8LaunchAbi(ManifestEntry)
      }
      found = Some(entry)
    }
  }
  guard found is Some(entry) else { raise InvalidFp8LaunchAbi(ManifestEntry) }
  entry
}

///|
fn validate_numeric_execution(
  plan : @plan.ModelPlan,
  entry : @numeric_capability_manifest.Fp8KernelManifestEntry,
) -> Unit raise Fp8LaunchAbiError {
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
    execution.activation_scale() !=
    @numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1() ||
    execution.tensor_input() != @numeric_contract.ComputeDType::f8_e4m3_finite() ||
    execution.accumulator() != @numeric_contract.AccumulatorDType::f32() ||
    execution.output() != @numeric_contract.ComputeDType::bfloat16() {
    raise InvalidFp8LaunchAbi(UnsupportedScaleMode)
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
}

///|
fn validate_stateless_shape(
  shape : Fp8ExactExecutionShape,
  profile : @launch_contract.KernelProfile,
) -> Unit raise Fp8LaunchAbiError {
  guard shape.data is Stateless(shape_profile, exact) else {
    raise InvalidFp8LaunchAbi(ManifestScope)
  }
  if shape_profile != profile.id() ||
    exact.batch_rows() != profile.batch_rows() ||
    exact.query_tokens_per_row() != profile.sequence_tokens() ||
    exact.context_tokens() != profile.sequence_tokens() ||
    exact.token_rows() != profile.token_rows() {
    raise InvalidFp8LaunchAbi(BindingMismatch)
  }
}

///|
fn validate_paged_shape(
  plan : @plan.ModelPlan,
  launches : @launch_contract.PagedKvLaunchContractSet,
  shape : Fp8ExactExecutionShape,
  profile : @launch_contract.PagedKernelProfile,
) -> Unit raise Fp8LaunchAbiError {
  guard shape.data is Paged(shape_profile, rows, query_tokens, page_entries) &&
    shape_profile == profile.id() else {
    raise InvalidFp8LaunchAbi(BindingMismatch)
  }
  if rows.length() > profile.max_query_rows() ||
    query_tokens > profile.max_query_tokens() ||
    page_entries > profile.max_page_table_entries() {
    raise InvalidFp8LaunchAbi(BindingMismatch)
  }
  let tokens_per_page = launches.device_kv_layout().tokens_per_page()
  let sequence_limit = plan.shape_constraints().max_sequence_tokens()
  for row in rows {
    if row.context_tokens > sequence_limit ||
      row.page_table_entries !=
      (row.context_tokens + tokens_per_page - 1) / tokens_per_page {
      raise InvalidFp8LaunchAbi(BindingMismatch)
    }
  }
}

///|
fn select_stateless_contract(
  plan : @plan.ModelPlan,
  manifest : @numeric_capability_manifest.Fp8KernelCapabilityManifest,
  launches : @launch_contract.LaunchContractSet,
  entry : @numeric_capability_manifest.Fp8KernelManifestEntry,
  shape : Fp8ExactExecutionShape,
) -> SelectedLaunchContract raise Fp8LaunchAbiError {
  if plan.kv_execution() != StatelessFullContext ||
    manifest.scope() != StatelessFullContextV1 ||
    launches.model_identity() != plan.identity() ||
    launches.device_target() != manifest.target() ||
    launches.catalog_version() != manifest.catalog_version() {
    raise InvalidFp8LaunchAbi(LaunchContract)
  }
  guard launches.contract(entry.profile(), entry.operation()) is Some(contract) else {
    raise InvalidFp8LaunchAbi(LaunchContract)
  }
  validate_stateless_shape(shape, contract.profile())
  if contract.entry_point() != entry.entry_point() ||
    contract.entry_point().family().module_digest() != entry.module_digest() {
    raise InvalidFp8LaunchAbi(KernelIdentity)
  }
  {
    entry_point: contract.entry_point(),
    workspace_bytes: contract.workspace_bytes(),
    workspace_alignment: contract.workspace_alignment(),
  }
}

///|
fn select_paged_contract(
  plan : @plan.ModelPlan,
  manifest : @numeric_capability_manifest.Fp8KernelCapabilityManifest,
  launches : @launch_contract.PagedKvLaunchContractSet,
  entry : @numeric_capability_manifest.Fp8KernelManifestEntry,
  shape : Fp8ExactExecutionShape,
) -> SelectedLaunchContract raise Fp8LaunchAbiError {
  if plan.kv_execution() != PagedKeyValue ||
    manifest.scope() != PagedFullGraphV1 ||
    launches.scope() != FullGraph ||
    launches.model_identity() != plan.identity() ||
    launches.device_target() != manifest.target() ||
    launches.catalog_version() != manifest.catalog_version() {
    raise InvalidFp8LaunchAbi(LaunchContract)
  }
  let mut found : @launch_contract.PagedKvAotLaunchContract? = None
  for contract in launches.contracts() {
    if contract.profile().id() == entry.profile() &&
      contract.operation_id() == entry.operation() {
      if found is Some(_) {
        raise InvalidFp8LaunchAbi(LaunchContract)
      }
      found = Some(contract)
    }
  }
  guard found is Some(contract) else {
    raise InvalidFp8LaunchAbi(LaunchContract)
  }
  validate_paged_shape(plan, launches, shape, contract.profile())
  if contract.entry_point() != entry.entry_point() ||
    contract.entry_point().family().module_digest() != entry.module_digest() {
    raise InvalidFp8LaunchAbi(KernelIdentity)
  }
  {
    entry_point: contract.entry_point(),
    workspace_bytes: contract.workspace_bytes(),
    workspace_alignment: contract.workspace_alignment(),
  }
}

///|
fn validate_paged_v4_shape(
  plan : @plan.ModelPlan,
  launches : @launch_contract.PagedV4LaunchContractSet,
  shape : Fp8ExactExecutionShape,
  profile : @launch_contract.PagedKernelProfile,
) -> Unit raise Fp8LaunchAbiError {
  guard shape.data is Paged(shape_profile, rows, query_tokens, page_entries) &&
    shape_profile == profile.id() else {
    raise InvalidFp8LaunchAbi(BindingMismatch)
  }
  if rows.length() > profile.max_query_rows() ||
    query_tokens > profile.max_query_tokens() ||
    page_entries > profile.max_page_table_entries() {
    raise InvalidFp8LaunchAbi(BindingMismatch)
  }
  let tokens_per_page = launches.device_kv_layout().tokens_per_page()
  let sequence_limit = plan.shape_constraints().max_sequence_tokens()
  for row in rows {
    if row.context_tokens > sequence_limit ||
      row.page_table_entries !=
      (row.context_tokens + tokens_per_page - 1) / tokens_per_page {
      raise InvalidFp8LaunchAbi(BindingMismatch)
    }
  }
}

///|
fn select_paged_v4_contract(
  plan : @plan.ModelPlan,
  manifest : @numeric_capability_manifest.Fp8KernelCapabilityManifest,
  launches : @launch_contract.PagedV4LaunchContractSet,
  entry : @numeric_capability_manifest.Fp8KernelManifestEntry,
  shape : Fp8ExactExecutionShape,
) -> SelectedLaunchContract raise Fp8LaunchAbiError {
  if plan.kv_execution() != PagedKeyValue ||
    manifest.scope() != PagedFullGraphV1 ||
    launches.catalog_version() != @catalog.CatalogVersion::v4() ||
    launches.model_identity() != plan.identity() ||
    launches.numeric_schema_digest() != plan.numeric_binding().digest() ||
    launches.device_target() != manifest.target() ||
    launches.catalog_version() != manifest.catalog_version() {
    raise InvalidFp8LaunchAbi(LaunchContract)
  }
  guard launches.contract(entry.profile(), entry.operation()) is Some(contract) else {
    raise InvalidFp8LaunchAbi(LaunchContract)
  }
  validate_paged_v4_shape(plan, launches, shape, contract.profile())
  if contract.operation_execution_digest() != entry.execution_digest() ||
    contract.entry_point() != entry.entry_point() ||
    contract.entry_point().family().module_digest() != entry.module_digest() {
    raise InvalidFp8LaunchAbi(KernelIdentity)
  }
  {
    entry_point: contract.entry_point(),
    workspace_bytes: contract.workspace_bytes(),
    workspace_alignment: contract.workspace_alignment(),
  }
}

///|
fn validate_workspace_regions(
  scale_tensors : ReadOnlyArray[Fp8ScaleTensorStorageRegion],
  scale : Fp8LaunchStorageRegion,
  workspace : Fp8LaunchStorageRegion?,
  expected_bytes : Int64,
  expected_alignment : Int64,
) -> Unit raise Fp8LaunchAbiError {
  if scale.byte_length != 4L || scale.alignment < 4L {
    raise InvalidFp8LaunchAbi(WorkspaceContract)
  }
  for index, tensor in scale_tensors {
    if tensor.region.byte_length != 4L || tensor.region.alignment < 4L {
      raise InvalidFp8LaunchAbi(WorkspaceContract)
    }
    validate_distinct_regions(tensor.region, scale)
    for previous in 0..<index {
      validate_distinct_regions(scale_tensors[previous].region, tensor.region)
    }
  }
  match (expected_bytes, workspace) {
    (0L, None) => ()
    (0L, Some(_)) | (_, None) => raise InvalidFp8LaunchAbi(WorkspaceContract)
    (_, Some(region)) => {
      if region.byte_length != expected_bytes ||
        region.alignment != expected_alignment {
        raise InvalidFp8LaunchAbi(WorkspaceContract)
      }
      validate_distinct_regions(scale, region)
      for tensor in scale_tensors {
        validate_distinct_regions(tensor.region, region)
      }
    }
  }
}

///|
fn bind_scale_tensor_regions(
  entry : @numeric_capability_manifest.Fp8KernelManifestEntry,
  regions : ArrayView[Fp8LaunchStorageRegion],
) -> ReadOnlyArray[Fp8ScaleTensorStorageRegion] raise Fp8LaunchAbiError {
  if regions.length() != entry.weights().length() {
    raise InvalidFp8LaunchAbi(BindingMismatch)
  }
  let bound : Array[Fp8ScaleTensorStorageRegion] = []
  for index, weight in entry.weights() {
    bound.push({
      tensor: weight.scale(),
      storage_digest: weight.scale_storage_digest(),
      region: regions[index],
    })
  }
  ReadOnlyArray::from_array(bound)
}

///|
/// Joins existing plan, manifest, and launch-contract authority into a pure
/// host FP8 launch ABI. It opens no module, device, allocation, or executor.
pub fn admit_dynamic_activation_scale_launch(
  plan : @plan.ModelPlan,
  kernels : @numeric_capability_manifest.AdmittedFp8KernelArtifacts,
  launches : Fp8LaunchContractSource,
  operation~ : @plan.OperationId,
  shape~ : Fp8ExactExecutionShape,
  weight_scale_regions~ : ArrayView[Fp8LaunchStorageRegion],
  activation_scale_workspace~ : Fp8LaunchStorageRegion,
  kernel_workspace~ : Fp8LaunchStorageRegion?,
) -> Fp8DynamicLaunchAbi raise Fp8LaunchAbiError {
  let manifest = kernels.manifest()
  if manifest.model_identity() != plan.identity() {
    raise InvalidFp8LaunchAbi(ModelIdentity)
  }
  if manifest.numeric_schema_digest() != plan.numeric_binding().digest() {
    raise InvalidFp8LaunchAbi(NumericSchema)
  }
  let profile = shape.profile()
  let entry = find_manifest_entry(manifest, operation, profile)
  validate_numeric_execution(plan, entry)
  let scale_tensor_regions = bind_scale_tensor_regions(
    entry, weight_scale_regions,
  )
  let selected = match launches {
    Stateless(contracts) =>
      select_stateless_contract(plan, manifest, contracts, entry, shape)
    Paged(contracts) =>
      select_paged_contract(plan, manifest, contracts, entry, shape)
    PagedV4(contracts) =>
      select_paged_v4_contract(plan, manifest, contracts, entry, shape)
  }
  validate_workspace_regions(
    scale_tensor_regions,
    activation_scale_workspace,
    kernel_workspace,
    selected.workspace_bytes,
    selected.workspace_alignment,
  )
  let digest = encode_abi(
    plan,
    manifest,
    entry,
    shape,
    scale_tensor_regions,
    activation_scale_workspace,
    kernel_workspace,
    selected.workspace_bytes,
    selected.workspace_alignment,
  )
  {
    version: Fp8LaunchAbiVersion::v1(),
    scope: shape.scope(),
    plan_identity: plan.identity(),
    numeric_schema: plan.numeric_binding().digest(),
    manifest_digest: manifest.digest(),
    target: manifest.target(),
    profile: entry.profile(),
    operation: entry.operation(),
    execution: entry.execution_digest(),
    module_digest: entry.module_digest(),
    entry_point: selected.entry_point,
    shape,
    scale_kind: FiniteE4M3DynamicF32V1,
    scale_granularity: @numeric_contract.ScaleGranularity::per_tensor(),
    weight_scales: entry.weights(),
    scale_tensor_regions,
    activation_scale_workspace,
    kernel_workspace,
    workspace_bytes: selected.workspace_bytes,
    workspace_alignment: selected.workspace_alignment,
    digest,
  }
}
