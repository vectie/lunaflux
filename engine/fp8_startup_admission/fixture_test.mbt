///|
struct StartupFixture {
  plan : @plan.ModelPlan
  device_numeric : @device.DeviceNumericCapability
  kernels : @numeric_capability_manifest.AdmittedFp8KernelArtifacts
}

///|
fn startup_operations() -> Array[@plan.PlanOperation] {
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
      workspace_bytes=0L,
    ),
  ]
}

///|
fn startup_plan() -> @plan.ModelPlan {
  let operations = startup_operations()
  let fp8_operation = try! @plan.OperationId::new(1)
  try! @plan.ModelPlan::new_stateless(
    content_digest=@spec.ContentDigest::from_sha256(
      "1111111111111111111111111111111111111111111111111111111111111111",
    ),
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
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations~,
    final_output=@plan.ActivationRef::new(1),
    numeric_schema=@plan_test_fixture.synthetic_fp8_w8a8_numeric_schema(
      operations,
      [fp8_operation],
    ),
  )
}

///|
fn startup_target() -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=9,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn startup_device(
  major? : Int = 8,
  minor? : Int = 9,
) -> @device.DeviceNumericCapability {
  let observed = try! @device.DeviceCapability::from_observed_facts(
    compute_major=major,
    compute_minor=minor,
    total_memory_bytes=8L * 1024L * 1024L * 1024L,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  try! @device.admit_finite_fp8_e4m3_w8a8(observed)
}

///|
fn startup_digest() -> @catalog.AotArtifactDigest {
  let bytes = b"startup-fp8-module".to_fixedarray()
  let digest = try! @catalog.AotArtifactDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
  digest
}

///|
fn startup_family(id : Int) -> @catalog.AotKernelFamily {
  let family_id = try! @catalog.AotKernelFamilyId::from_stable_int(id)
  @catalog.AotKernelFamily::new(module_digest=startup_digest(), family_id~)
}

///|
fn startup_entry(operation : Int) -> @catalog.AotKernelEntryPoint {
  let entry_point_id = try! @catalog.AotKernelEntryPointId::from_stable_int(
    operation + 1,
  )
  @catalog.AotKernelEntryPoint::new(
    family=startup_family(operation + 1),
    entry_point_id~,
  )
}

///|
fn startup_resolved(plan : @plan.ModelPlan) -> @catalog.ResolvedKernelCatalog {
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
        implementation=ContentAddressedAot(startup_family(index + 1)),
        workspace_bytes=0L,
        workspace_alignment=1L,
      ),
    )
  }
  try! @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v1(),
    entries~,
  ).resolve(plan, startup_target())
}

///|
fn startup_operands(operation : Int) -> Array[@launch_contract.AotOperand] {
  if operation == 0 {
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
        role=ActivationInput(@plan.ActivationRef::new(0)),
        byte_count=64L,
        alignment=16L,
      ),
      try! @launch_contract.AotOperand::new(
        role=WeightInput(@plan.TensorRef::new(1)),
        byte_count=256L,
        alignment=16L,
      ),
      try! @launch_contract.AotOperand::new(
        role=ActivationOutput(@plan.ActivationRef::new(1)),
        byte_count=64L,
        alignment=16L,
      ),
    ]
  }
}

///|
fn startup_launches(
  plan : @plan.ModelPlan,
  resolved : @catalog.ResolvedKernelCatalog,
) -> @launch_contract.LaunchContractSet {
  let inputs : Array[@launch_contract.AotLaunchContractInput] = []
  let profile_id = try! @launch_contract.KernelProfileId::from_stable_int(1)
  let dimensions = try! @launch_contract.AotLaunchDimensions::new(
    grid_x=2,
    grid_y=1,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  for operation in 0..<2 {
    let operation_id = try! @plan.OperationId::new(operation)
    inputs.push(
      try! @launch_contract.AotLaunchContractInput::new(
        profile_id~,
        operation_id~,
        entry_point=startup_entry(operation),
        dimensions~,
        operands=startup_operands(operation),
      ),
    )
  }
  let profile = try! @launch_contract.KernelProfile::full_prefill(
    id=profile_id,
    batch_rows=1,
    sequence_tokens=2,
    token_rows=2L,
  )
  try! @launch_contract.admit(
    plan,
    resolved,
    [profile],
    inputs,
    @launch_contract.LaunchContractLimits::new(
      max_profiles=1,
      max_contracts=2,
      max_operands_per_contract=4,
    ),
  )
}

///|
fn startup_artifacts(
  launches : @launch_contract.LaunchContractSet,
) -> @artifact.KernelArtifactBundle {
  try! @artifact.admit(
    launches,
    [
      @artifact.KernelModuleInput::new(
        digest=startup_digest(),
        module_bytes=b"startup-fp8-module".to_fixedarray(),
      ),
    ],
    [
      @artifact.KernelEntryPointInput::new(
        entry_point=startup_entry(0),
        function_symbol="embedding_bf16",
      ),
      @artifact.KernelEntryPointInput::new(
        entry_point=startup_entry(1),
        function_symbol="projection_fp8",
      ),
    ],
    @artifact.KernelArtifactLimits::new(
      max_modules=1,
      max_entry_points=2,
      max_module_bytes=256L,
      max_total_module_bytes=256L,
      max_function_symbol_bytes=64,
    ),
  )
}

///|
fn startup_fixture() -> StartupFixture {
  let plan = startup_plan()
  let resolved = startup_resolved(plan)
  let launches = startup_launches(plan, resolved)
  let artifacts = startup_artifacts(launches)
  let device_numeric = startup_device()
  let profile = try! @launch_contract.KernelProfileId::from_stable_int(1)
  let operation = try! @plan.OperationId::new(1)
  let claim = @numeric_capability_manifest.Fp8KernelManifestClaim::new(
    profile~,
    operation~,
    execution=plan.numeric_binding().operations()[1].execution().digest(),
    entry_point=startup_entry(1),
  )
  let kernels = try! @numeric_capability_manifest.admit_stateless_fp8_e4m3_w8a8(
    plan,
    device_numeric,
    resolved,
    launches,
    artifacts,
    [claim],
    @numeric_capability_manifest.Fp8KernelManifestLimits::new(
      max_entries=4,
      max_canonical_bytes=4096,
    ),
  )
  { plan, device_numeric, kernels }
}
