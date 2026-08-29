///|
struct Fp8ManifestFixture {
  plan : @plan.ModelPlan
  target : @catalog.DeviceTarget
  resolved : @catalog.ResolvedKernelCatalog
  launches : @launch_contract.LaunchContractSet
  artifacts : @artifact.KernelArtifactBundle
  device_numeric : @device.DeviceNumericCapability
  claim : @numeric_capability_manifest.Fp8KernelManifestClaim
}

///|
fn fp8_fixture_content() -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
}

///|
fn fp8_fixture_operations() -> Array[@plan.PlanOperation] {
  let embedding = try! @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
    outputs=[@plan.ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=128, width=16),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes=0L,
  )
  let output = try! @plan.PlanOperation::new_stateless(
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
  )
  [embedding, output]
}

///|
fn fp8_fixture_plan(fp8? : Bool = true) -> @plan.ModelPlan {
  let operations = fp8_fixture_operations()
  let fp8_operation = try! @plan.OperationId::new(1)
  let numeric = if fp8 {
    @plan_test_fixture.synthetic_fp8_w8a8_numeric_schema(operations, [
      fp8_operation,
    ])
  } else {
    @plan_test_fixture.synthetic_bf16_numeric_schema(operations)
  }
  fp8_plan_with_numeric(operations, numeric)
}

///|
fn fp8_plan_with_numeric(
  operations : ArrayView[@plan.PlanOperation],
  numeric : @numeric_contract.ModelNumericSchema,
) -> @plan.ModelPlan {
  try! @plan.ModelPlan::new_stateless(
    content_digest=fp8_fixture_content(),
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
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations~,
    final_output=@plan.ActivationRef::new(1),
    numeric_schema=numeric,
  )
}

///|
fn fp8_weight_only_activation_plan() -> @plan.ModelPlan {
  let operations = fp8_fixture_operations()
  let fp8_operation = try! @plan.OperationId::new(1)
  let base = @plan_test_fixture.synthetic_fp8_w8a8_numeric_schema(operations, [
    fp8_operation,
  ])
  let bindings : Array[@numeric_contract.NumericOperationSchema] = []
  bindings.push(base.operations()[0])
  let ordinal = try! @numeric_contract.NumericOperationOrdinal::new(1)
  let execution = try! @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::bfloat16(),
    activation_compute=@numeric_contract.ComputeDType::bfloat16(),
    activation_scale=@numeric_contract.ActivationScalePolicy::absent_exact_v1(),
    tensor_input=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
  bindings.push(
    @numeric_contract.NumericOperationSchema::new(ordinal~, execution~),
  )
  let numeric = try! @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors=base.tensors(),
    operations=bindings,
  )
  fp8_plan_with_numeric(operations, numeric)
}

///|
fn fp8_module_bytes() -> FixedArray[Byte] {
  b"finite-e4m3-fixture-cubin".to_fixedarray()
}

///|
fn fp8_module_digest() -> @catalog.AotArtifactDigest {
  let bytes = fp8_module_bytes()
  let digest = try! @catalog.AotArtifactDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
  digest
}

///|
fn fp8_family(stable_id : Int) -> @catalog.AotKernelFamily {
  let family_id = try! @catalog.AotKernelFamilyId::from_stable_int(stable_id)
  @catalog.AotKernelFamily::new(module_digest=fp8_module_digest(), family_id~)
}

///|
fn fp8_entry(family : Int, entry : Int) -> @catalog.AotKernelEntryPoint {
  let entry_point_id = try! @catalog.AotKernelEntryPointId::from_stable_int(
    entry,
  )
  @catalog.AotKernelEntryPoint::new(family=fp8_family(family), entry_point_id~)
}

///|
fn fp8_target(major? : Int = 8, minor? : Int = 9) -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=major,
    compute_minor=minor,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn fp8_device_numeric(
  major? : Int = 8,
  minor? : Int = 9,
) -> @device.DeviceNumericCapability {
  let observed = try! @device.DeviceCapability::from_observed_facts(
    compute_major=major,
    compute_minor=minor,
    total_memory_bytes=16L * 1024L * 1024L * 1024L,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  try! @device.admit_finite_fp8_e4m3_w8a8(observed)
}

///|
fn fp8_catalog_entry(
  operation : @plan.PlanOperation,
  family : Int,
  vendor? : Bool = false,
) -> @catalog.CatalogEntry {
  try! @catalog.CatalogEntry::new(
    semantic_version=@catalog.KernelSemanticVersion::v1(),
    operation_kind=operation.kind(),
    operation_shape=operation.shape(),
    capability=operation.required_capability(),
    compute_major=8,
    compute_minor=9,
    implementation=if vendor {
      VendorBf16GemmImplicitHeuristicV1
    } else {
      ContentAddressedAot(fp8_family(family))
    },
    workspace_bytes=0L,
    workspace_alignment=1L,
  )
}

///|
fn fp8_resolved_catalog(
  plan : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
  vendor_output? : Bool = false,
) -> @catalog.ResolvedKernelCatalog {
  try! @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v1(), entries=[
    fp8_catalog_entry(plan.operations()[0], 1),
    fp8_catalog_entry(plan.operations()[1], 2, vendor=vendor_output),
  ]).resolve(plan, target)
}

///|
fn fp8_profile() -> @launch_contract.KernelProfile {
  try! @launch_contract.KernelProfile::full_prefill(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    batch_rows=1,
    sequence_tokens=2,
    token_rows=2L,
  )
}

///|
fn fp8_operands(operation : Int) -> Array[@launch_contract.AotOperand] {
  if operation == 0 {
    [
      try! @launch_contract.AotOperand::new(
        role=ExternalInput(TokenIds),
        byte_count=8L,
        alignment=4L,
      ),
      try! @launch_contract.AotOperand::new(
        role=WeightInput(@plan.TensorRef::new(0)),
        byte_count=4096L,
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
fn fp8_launch_input(operation : Int) -> @launch_contract.AotLaunchContractInput {
  let profile_id = try! @launch_contract.KernelProfileId::from_stable_int(1)
  let operation_id = try! @plan.OperationId::new(operation)
  let dimensions = try! @launch_contract.AotLaunchDimensions::new(
    grid_x=2,
    grid_y=1,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  try! @launch_contract.AotLaunchContractInput::new(
    profile_id~,
    operation_id~,
    entry_point=fp8_entry(operation + 1, 10 + operation),
    dimensions~,
    operands=fp8_operands(operation),
  )
}

///|
fn fp8_launches(
  plan : @plan.ModelPlan,
  resolved : @catalog.ResolvedKernelCatalog,
) -> @launch_contract.LaunchContractSet {
  try! @launch_contract.admit(
    plan,
    resolved,
    [fp8_profile()],
    [fp8_launch_input(0), fp8_launch_input(1)],
    @launch_contract.LaunchContractLimits::new(
      max_profiles=2,
      max_contracts=4,
      max_operands_per_contract=4,
    ),
  )
}

///|
fn fp8_artifacts(
  launches : @launch_contract.LaunchContractSet,
) -> @artifact.KernelArtifactBundle {
  let digest = fp8_module_digest()
  try! @artifact.admit(
    launches,
    [@artifact.KernelModuleInput::new(digest~, module_bytes=fp8_module_bytes())],
    [
      @artifact.KernelEntryPointInput::new(
        entry_point=fp8_entry(1, 10),
        function_symbol="embedding_bf16",
      ),
      @artifact.KernelEntryPointInput::new(
        entry_point=fp8_entry(2, 11),
        function_symbol="projection_fp8",
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
}

///|
fn fp8_manifest_fixture() -> Fp8ManifestFixture {
  let plan = fp8_fixture_plan()
  let target = fp8_target()
  let resolved = fp8_resolved_catalog(plan, target)
  let launches = fp8_launches(plan, resolved)
  let artifacts = fp8_artifacts(launches)
  let execution = plan.numeric_binding().operations()[1].execution().digest()
  let profile = try! @launch_contract.KernelProfileId::from_stable_int(1)
  let operation = try! @plan.OperationId::new(1)
  {
    plan,
    target,
    resolved,
    launches,
    artifacts,
    device_numeric: fp8_device_numeric(),
    claim: @numeric_capability_manifest.Fp8KernelManifestClaim::new(
      profile~,
      operation~,
      execution~,
      entry_point=fp8_entry(2, 11),
    ),
  }
}
