///|
struct Fp8AbiFixture {
  plan : @plan.ModelPlan
  launches : @launch_contract.LaunchContractSet
  kernels : @numeric_capability_manifest.AdmittedFp8KernelArtifacts
  weight_scale : @fp8_launch_abi.Fp8LaunchStorageRegion
  scale : @fp8_launch_abi.Fp8LaunchStorageRegion
  workspace : @fp8_launch_abi.Fp8LaunchStorageRegion
  shape : @fp8_launch_abi.Fp8ExactExecutionShape
}

///|
fn fixture_digest(value : String) -> String {
  match value {
    "content" =>
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    "storage" =>
      "1111111111111111111111111111111111111111111111111111111111111111"
    _ => "2222222222222222222222222222222222222222222222222222222222222222"
  }
}

///|
fn fixture_operations(
  projection_workspace? : Int64 = 64L,
) -> Array[@plan.PlanOperation] {
  [
    try! @plan.PlanOperation::new_stateless(
      id=@plan.OperationId::new(0),
      kind=Embedding,
      inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
      outputs=[@plan.ActivationRef::new(0)],
      shape=EmbeddingShape(vocabulary_size=128, width=16),
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
  ]
}

///|
fn fixture_plan(projection_workspace? : Int64 = 64L) -> @plan.ModelPlan {
  let operations = fixture_operations(projection_workspace~)
  let numeric = @plan_test_fixture.synthetic_fp8_w8a8_numeric_schema(
    operations,
    [try! @plan.OperationId::new(1)],
  )
  try! @plan.ModelPlan::new_stateless(
    content_digest=@spec.ContentDigest::from_sha256(fixture_digest("content")),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows=1,
      max_sequence_tokens=8,
      vocabulary_size=128,
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
    final_output=@plan.ActivationRef::new(1),
    numeric_schema=numeric,
  )
}

///|
fn fixture_module_bytes() -> FixedArray[Byte] {
  b"fp8-dynamic-scale-aot-fixture".to_fixedarray()
}

///|
fn fixture_module_digest() -> @catalog.AotArtifactDigest {
  try! @catalog.AotArtifactDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(fixture_module_bytes())),
  )
}

///|
fn fixture_family(id : Int) -> @catalog.AotKernelFamily {
  @catalog.AotKernelFamily::new(
    module_digest=fixture_module_digest(),
    family_id=try! @catalog.AotKernelFamilyId::from_stable_int(id),
  )
}

///|
fn fixture_entry(family : Int, id : Int) -> @catalog.AotKernelEntryPoint {
  @catalog.AotKernelEntryPoint::new(
    family=fixture_family(family),
    entry_point_id=try! @catalog.AotKernelEntryPointId::from_stable_int(id),
  )
}

///|
fn fixture_target() -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=9,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn fixture_catalog(plan : @plan.ModelPlan) -> @catalog.ResolvedKernelCatalog {
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in plan.operations() {
    entries.push(
      try! @catalog.CatalogEntry::new(
        semantic_version=@catalog.KernelSemanticVersion::v1(),
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        compute_major=8,
        compute_minor=9,
        implementation=ContentAddressedAot(fixture_family(index + 1)),
        workspace_bytes=operation.workspace_bytes(),
        workspace_alignment=if operation.workspace_bytes() == 4L ||
          operation.workspace_bytes() == 8L {
          4L
        } else if operation.workspace_bytes() > 0L {
          16L
        } else {
          1L
        },
      ),
    )
  }
  try! @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v1(),
    entries~,
  ).resolve(plan, fixture_target())
}

///|
fn fixture_profile(
  id? : Int = 1,
  sequence_tokens? : Int = 2,
) -> @launch_contract.KernelProfile {
  try! @launch_contract.KernelProfile::full_prefill(
    id=@launch_contract.KernelProfileId::from_stable_int(id),
    batch_rows=1,
    sequence_tokens~,
    token_rows=sequence_tokens.to_int64(),
  )
}

///|
fn fixture_operands(
  operation : @plan.PlanOperation,
  sequence_tokens? : Int = 2,
) -> Array[@launch_contract.AotOperand] {
  let activation_bytes = sequence_tokens.to_int64() * 32L
  if operation.id().as_int() == 0 {
    [
      try! @launch_contract.AotOperand::new(
        role=ExternalInput(TokenIds),
        byte_count=sequence_tokens.to_int64() * 4L,
        alignment=4L,
      ),
      try! @launch_contract.AotOperand::new(
        role=WeightInput(@plan.TensorRef::new(0)),
        byte_count=4096L,
        alignment=16L,
      ),
      try! @launch_contract.AotOperand::new(
        role=ActivationOutput(@plan.ActivationRef::new(0)),
        byte_count=activation_bytes,
        alignment=16L,
      ),
    ]
  } else {
    [
      try! @launch_contract.AotOperand::new(
        role=ActivationInput(@plan.ActivationRef::new(0)),
        byte_count=activation_bytes,
        alignment=16L,
      ),
      try! @launch_contract.AotOperand::new(
        role=WeightInput(@plan.TensorRef::new(1)),
        byte_count=256L,
        alignment=16L,
      ),
      try! @launch_contract.AotOperand::new(
        role=ActivationOutput(@plan.ActivationRef::new(1)),
        byte_count=activation_bytes,
        alignment=16L,
      ),
      try! @launch_contract.AotOperand::new(
        role=Workspace,
        byte_count=operation.workspace_bytes(),
        alignment=if operation.workspace_bytes() == 4L { 4L } else { 16L },
      ),
    ]
  }
}

///|
fn fixture_launches(
  plan : @plan.ModelPlan,
  catalog : @catalog.ResolvedKernelCatalog,
) -> @launch_contract.LaunchContractSet {
  let inputs : Array[@launch_contract.AotLaunchContractInput] = []
  for profile in 1..<=2 {
    let sequence_tokens = profile * 2
    for index, operation in plan.operations() {
      inputs.push(
        try! @launch_contract.AotLaunchContractInput::new(
          profile_id=@launch_contract.KernelProfileId::from_stable_int(profile),
          operation_id=@plan.OperationId::new(index),
          entry_point=fixture_entry(index + 1, 10 + index),
          dimensions=@launch_contract.AotLaunchDimensions::new(
            grid_x=2,
            grid_y=1,
            grid_z=1,
            block_x=32,
            block_y=1,
            block_z=1,
            shared_memory_bytes=0,
          ),
          operands=fixture_operands(operation, sequence_tokens~),
        ),
      )
    }
  }
  try! @launch_contract.admit(
    plan,
    catalog,
    [
      fixture_profile(id=1, sequence_tokens=2),
      fixture_profile(id=2, sequence_tokens=4),
    ],
    inputs,
    @launch_contract.LaunchContractLimits::new(
      max_profiles=2,
      max_contracts=4,
      max_operands_per_contract=4,
    ),
  )
}

///|
fn fixture_kernels(
  plan : @plan.ModelPlan,
  catalog : @catalog.ResolvedKernelCatalog,
  launches : @launch_contract.LaunchContractSet,
) -> @numeric_capability_manifest.AdmittedFp8KernelArtifacts {
  let artifacts = try! @artifact.admit(
    launches,
    [
      @artifact.KernelModuleInput::new(
        digest=fixture_module_digest(),
        module_bytes=fixture_module_bytes(),
      ),
    ],
    [
      @artifact.KernelEntryPointInput::new(
        entry_point=fixture_entry(1, 10),
        function_symbol="embedding_bf16",
      ),
      @artifact.KernelEntryPointInput::new(
        entry_point=fixture_entry(2, 11),
        function_symbol="projection_fp8_dynamic",
      ),
    ],
    @artifact.KernelArtifactLimits::new(
      max_modules=1,
      max_entry_points=2,
      max_module_bytes=1024L,
      max_total_module_bytes=1024L,
      max_function_symbol_bytes=64,
    ),
  )
  let observed = try! @device.DeviceCapability::from_observed_facts(
    compute_major=8,
    compute_minor=9,
    total_memory_bytes=16L * 1024L * 1024L * 1024L,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  let numeric = try! @device.admit_finite_fp8_e4m3_w8a8(observed)
  let operation = try! @plan.OperationId::new(1)
  let claim_one = @numeric_capability_manifest.Fp8KernelManifestClaim::new(
    profile=try! @launch_contract.KernelProfileId::from_stable_int(1),
    operation~,
    execution=plan.numeric_binding().operations()[1].execution().digest(),
    entry_point=fixture_entry(2, 11),
  )
  let claim_two = @numeric_capability_manifest.Fp8KernelManifestClaim::new(
    profile=try! @launch_contract.KernelProfileId::from_stable_int(2),
    operation~,
    execution=plan.numeric_binding().operations()[1].execution().digest(),
    entry_point=fixture_entry(2, 11),
  )
  try! @numeric_capability_manifest.admit_stateless_fp8_e4m3_w8a8(
    plan,
    numeric,
    catalog,
    launches,
    artifacts,
    [claim_one, claim_two],
    @numeric_capability_manifest.Fp8KernelManifestLimits::new(
      max_entries=4,
      max_canonical_bytes=4096,
    ),
  )
}

///|
fn fixture_region(
  offset : Int64,
  length : Int64,
  alignment : Int64,
  workspace? : Bool = false,
) -> @fp8_launch_abi.Fp8LaunchStorageRegion {
  try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
    storage=@fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(
      fixture_digest(if workspace { "workspace" } else { "storage" }),
    ),
    storage_bytes=256L,
    offset~,
    byte_length=length,
    alignment~,
  )
}

///|
fn fp8_abi_fixture() -> Fp8AbiFixture {
  let plan = fixture_plan()
  let catalog = fixture_catalog(plan)
  let launches = fixture_launches(plan, catalog)
  {
    plan,
    launches,
    kernels: fixture_kernels(plan, catalog, launches),
    weight_scale: try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
      storage=@fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(
        "6666666666666666666666666666666666666666666666666666666666666666",
      ),
      storage_bytes=4L,
      offset=0L,
      byte_length=4L,
      alignment=4L,
    ),
    scale: fixture_region(0L, 4L, 4L),
    workspace: fixture_region(0L, 64L, 16L, workspace=true),
    shape: try! @fp8_launch_abi.Fp8ExactExecutionShape::stateless_full_prefill(
      profile=@launch_contract.KernelProfileId::from_stable_int(1),
      @device_profile.ExactExecutionShape::full_prefill(
        batch_rows=1,
        sequence_tokens=2,
      ),
    ),
  }
}

///|
fn admit_fixture(
  fixture : Fp8AbiFixture,
  scale? : @fp8_launch_abi.Fp8LaunchStorageRegion,
  workspace? : @fp8_launch_abi.Fp8LaunchStorageRegion,
  shape? : @fp8_launch_abi.Fp8ExactExecutionShape,
  weight_scale? : @fp8_launch_abi.Fp8LaunchStorageRegion,
) -> @fp8_launch_abi.Fp8DynamicLaunchAbi raise @fp8_launch_abi.Fp8LaunchAbiError {
  @fp8_launch_abi.admit_dynamic_activation_scale_launch(
    fixture.plan,
    fixture.kernels,
    Stateless(fixture.launches),
    operation=try! @plan.OperationId::new(1),
    shape=shape.unwrap_or(fixture.shape),
    weight_scale_regions=[weight_scale.unwrap_or(fixture.weight_scale)],
    activation_scale_workspace=scale.unwrap_or(fixture.scale),
    kernel_workspace=Some(workspace.unwrap_or(fixture.workspace)),
  )
}
