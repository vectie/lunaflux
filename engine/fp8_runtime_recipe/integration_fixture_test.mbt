///|
struct RuntimeRecipeFixture {
  plan : @plan.ModelPlan
  startup : @fp8_startup_admission.Fp8StartupAdmission
  contracts : @launch_contract.LaunchContractSet
}

///|
fn recipe_operations(
  projection_workspace? : Int64 = 0L,
) -> Array[@plan.PlanOperation] {
  [
    try! @plan.PlanOperation::new_stateless(
      id=@plan.OperationId::new(0),
      kind=Embedding,
      inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
      outputs=[@plan.ActivationRef::new(0)],
      shape=EmbeddingShape(vocabulary_size=64, width=16),
      required_capability=@plan.KernelCapabilityId::embedding_lookup(),
      workspace_bytes=0L,
    ),
    try! @plan.PlanOperation::new_stateless(
      id=@plan.OperationId::new(1),
      kind=OutputProjection,
      inputs=[
        Activation(@plan.ActivationRef::new(0)),
        Tensor(@plan.TensorRef::new(1)),
      ],
      outputs=[@plan.ActivationRef::new(1)],
      shape=ProjectionShape(input_width=16, output_width=16),
      required_capability=@plan.KernelCapabilityId::dense_projection(),
      workspace_bytes=projection_workspace,
    ),
    try! @plan.PlanOperation::new_stateless(
      id=@plan.OperationId::new(2),
      kind=OutputProjection,
      inputs=[
        Activation(@plan.ActivationRef::new(1)),
        Tensor(@plan.TensorRef::new(2)),
      ],
      outputs=[@plan.ActivationRef::new(2)],
      shape=ProjectionShape(input_width=16, output_width=16),
      required_capability=@plan.KernelCapabilityId::dense_projection(),
      workspace_bytes=projection_workspace,
    ),
  ]
}

///|
fn recipe_plan(
  content? : String = "1111111111111111111111111111111111111111111111111111111111111111",
  fp8_operations? : Array[@plan.OperationId] = [try! @plan.OperationId::new(1)],
  projection_workspace? : Int64 = 0L,
) -> @plan.ModelPlan {
  let operations = recipe_operations(projection_workspace~)
  try! @plan.ModelPlan::new_stateless(
    content_digest=@spec.ContentDigest::from_sha256(content),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows=1,
      max_sequence_tokens=4,
      vocabulary_size=64,
      hidden_size=16,
      intermediate_size=32,
      attention_head_count=4,
      kv_head_count=2,
      head_dimension=4,
    ),
    kv_cache_geometry=@plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=2,
      head_dimension=4,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    workspace_bounds=@plan.WorkspaceBounds::new(
      max_temporary_bytes=projection_workspace,
    ),
    operations~,
    final_output=@plan.ActivationRef::new(2),
    numeric_schema=@plan_test_fixture.synthetic_fp8_w8a8_numeric_schema(
      operations, fp8_operations,
    ),
  )
}

///|
fn recipe_target() -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=9,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn recipe_device() -> @device.DeviceNumericCapability {
  try! @device.admit_finite_fp8_e4m3_w8a8(
    @device.DeviceCapability::from_observed_facts(
      compute_major=8,
      compute_minor=9,
      total_memory_bytes=8589934592L,
      supports_bf16=true,
      supports_cublas_lt=true,
    ),
  )
}

///|
fn recipe_module_digest() -> @catalog.AotArtifactDigest {
  try! @catalog.AotArtifactDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(b"fp8-recipe-module")),
  )
}

///|
fn recipe_entry(index : Int) -> @catalog.AotKernelEntryPoint {
  let family_index = if index == 0 { 1 } else { 2 }
  @catalog.AotKernelEntryPoint::new(
    family=@catalog.AotKernelFamily::new(
      module_digest=recipe_module_digest(),
      family_id=try! @catalog.AotKernelFamilyId::from_stable_int(family_index),
    ),
    entry_point_id=try! @catalog.AotKernelEntryPointId::from_stable_int(
      index + 1,
    ),
  )
}

///|
fn recipe_resolved(plan : @plan.ModelPlan) -> @catalog.ResolvedKernelCatalog {
  let entries : Array[@catalog.CatalogEntry] = []
  // Both projections have identical catalog semantics and intentionally share
  // one resolved family entry; per-operation entry points remain distinct.
  for index in 0..<2 {
    let operation = plan.operations()[index]
    let family_index = if index == 0 { 1 } else { 2 }
    entries.push(
      try! @catalog.CatalogEntry::new(
        semantic_version=@catalog.KernelSemanticVersion::v1(),
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        compute_major=8,
        compute_minor=9,
        implementation=ContentAddressedAot(
          @catalog.AotKernelFamily::new(
            module_digest=recipe_module_digest(),
            family_id=try! @catalog.AotKernelFamilyId::from_stable_int(
              family_index,
            ),
          ),
        ),
        workspace_bytes=operation.workspace_bytes(),
        workspace_alignment=if operation.workspace_bytes() > 0L {
          4L
        } else {
          1L
        },
      ),
    )
  }
  try! @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v1(),
    entries~,
  ).resolve(plan, recipe_target())
}

///|
fn recipe_operands(
  index : Int,
  workspace_bytes~ : Int64,
) -> Array[@launch_contract.AotOperand] {
  let operands = if index == 0 {
    [
      try! @launch_contract.AotOperand::new(
        role=ExternalInput(TokenIds),
        byte_count=8L,
        alignment=4L,
      ),
      try! @launch_contract.AotOperand::new(
        role=WeightInput(@plan.TensorRef::new(0)),
        byte_count=2048L,
        alignment=16L,
      ),
      try! @launch_contract.AotOperand::new(
        role=ActivationOutput(@plan.ActivationRef::new(0)),
        byte_count=64L,
        alignment=16L,
      ),
    ]
  } else {
    [
      try! @launch_contract.AotOperand::new(
        role=ActivationInput(@plan.ActivationRef::new(index - 1)),
        byte_count=64L,
        alignment=16L,
      ),
      try! @launch_contract.AotOperand::new(
        role=WeightInput(@plan.TensorRef::new(index)),
        byte_count=256L,
        alignment=16L,
      ),
      try! @launch_contract.AotOperand::new(
        role=ActivationOutput(@plan.ActivationRef::new(index)),
        byte_count=64L,
        alignment=16L,
      ),
    ]
  }
  if workspace_bytes > 0L {
    operands.push(
      try! @launch_contract.AotOperand::new(
        role=Workspace,
        byte_count=workspace_bytes,
        alignment=4L,
      ),
    )
  }
  operands
}

///|
fn recipe_contracts(
  plan : @plan.ModelPlan,
  resolved : @catalog.ResolvedKernelCatalog,
) -> @launch_contract.LaunchContractSet {
  let inputs : Array[@launch_contract.AotLaunchContractInput] = []
  let profile = try! @launch_contract.KernelProfileId::from_stable_int(1)
  for index in 0..<3 {
    inputs.push(
      try! @launch_contract.AotLaunchContractInput::new(
        profile_id=profile,
        operation_id=@plan.OperationId::new(index),
        entry_point=recipe_entry(index),
        dimensions=@launch_contract.AotLaunchDimensions::new(
          grid_x=2,
          grid_y=1,
          grid_z=1,
          block_x=32,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        operands=recipe_operands(
          index,
          workspace_bytes=plan.operations()[index].workspace_bytes(),
        ),
      ),
    )
  }
  try! @launch_contract.admit(
    plan,
    resolved,
    [
      @launch_contract.KernelProfile::full_prefill(
        id=profile,
        batch_rows=1,
        sequence_tokens=2,
        token_rows=2L,
      ),
    ],
    inputs,
    @launch_contract.LaunchContractLimits::new(
      max_profiles=1,
      max_contracts=3,
      max_operands_per_contract=4,
    ),
  )
}

///|
fn runtime_recipe_fixture(
  plan? : @plan.ModelPlan = recipe_plan(),
) -> RuntimeRecipeFixture {
  let resolved = recipe_resolved(plan)
  let contracts = recipe_contracts(plan, resolved)
  let artifacts = try! @artifact.admit(
    contracts,
    [
      @artifact.KernelModuleInput::from_bytes(
        digest=recipe_module_digest(),
        module_bytes=b"fp8-recipe-module",
      ),
    ],
    [
      @artifact.KernelEntryPointInput::new(
        entry_point=recipe_entry(0),
        function_symbol="embedding_bf16",
      ),
      @artifact.KernelEntryPointInput::new(
        entry_point=recipe_entry(1),
        function_symbol="projection_fp8_1",
      ),
      @artifact.KernelEntryPointInput::new(
        entry_point=recipe_entry(2),
        function_symbol="projection_fp8_2",
      ),
    ],
    @artifact.KernelArtifactLimits::new(
      max_modules=1,
      max_entry_points=3,
      max_module_bytes=256L,
      max_total_module_bytes=256L,
      max_function_symbol_bytes=64,
    ),
  )
  let profile = try! @launch_contract.KernelProfileId::from_stable_int(1)
  let claims : Array[@numeric_capability_manifest.Fp8KernelManifestClaim] = []
  for index in 0..<plan.operations().length() {
    let execution = plan.numeric_binding().operations()[index].execution()
    if execution.activation_compute() ==
      @numeric_contract.ComputeDType::f8_e4m3_finite() {
      claims.push(
        @numeric_capability_manifest.Fp8KernelManifestClaim::new(
          profile~,
          operation=try! @plan.OperationId::new(index),
          execution=execution.digest(),
          entry_point=recipe_entry(index),
        ),
      )
    }
  }
  let kernels = @numeric_capability_manifest.admit_stateless_fp8_e4m3_w8a8(
    plan,
    recipe_device(),
    resolved,
    contracts,
    artifacts,
    claims,
    @numeric_capability_manifest.Fp8KernelManifestLimits::new(
      max_entries=4,
      max_canonical_bytes=8192,
    ),
  ) catch {
    InvalidFp8KernelManifest(ClaimCount) =>
      abort("fixture manifest claim count")
    InvalidFp8KernelManifest(ClaimOrder) =>
      abort("fixture manifest claim order")
    InvalidFp8KernelManifest(ClaimExecution) =>
      abort("fixture manifest claim execution")
    InvalidFp8KernelManifest(ClaimEntryPoint) =>
      abort("fixture manifest claim entry")
    InvalidFp8KernelManifest(ArtifactBundle) =>
      abort("fixture manifest artifact")
    _ => abort("fixture manifest other")
  }
  let startup = try! @fp8_startup_admission.admit(
    plan,
    recipe_device(),
    kernels,
    @fp8_startup_admission.Fp8StartupAdmissionLimits::new(
      max_canonical_bytes=8192,
    ),
  )
  { plan, startup, contracts }
}
