///|
fn numeric_v4_operations() -> Array[@plan.PlanOperation] {
  let layer = try! @plan.KvLayerId::new(0)
  [
    try! @plan.PlanOperation::new_with_runtime_inputs(
      id=@plan.OperationId::new(0),
      kind=Embedding,
      inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
      outputs=[@plan.ActivationRef::new(0)],
      shape=EmbeddingShape(vocabulary_size=128, width=16),
      required_capability=@plan.KernelCapabilityId::embedding_lookup(),
      workspace_bytes=0L,
      runtime_inputs=[StepCounts],
    ),
    try! @plan.PlanOperation::new_with_runtime_inputs(
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
      runtime_inputs=[StepCounts],
    ),
    try! @plan.PlanOperation::new_with_execution(
      id=@plan.OperationId::new(2),
      kind=RotaryEmbedding,
      inputs=[Activation(@plan.ActivationRef::new(1))],
      outputs=[@plan.ActivationRef::new(2)],
      shape=RotaryShape(
        query_heads=4,
        kv_heads=2,
        head_dimension=4,
        rope_theta=10000.0,
      ),
      required_capability=@plan.KernelCapabilityId::positioned_rotary_embedding(),
      workspace_bytes=0L,
      kv_layer=layer,
      runtime_inputs=[StepCounts, QueryPositions],
      persistent_state=NoPersistentState,
    ),
    try! @plan.PlanOperation::new_with_execution(
      id=@plan.OperationId::new(3),
      kind=CausalAttention,
      inputs=[Activation(@plan.ActivationRef::new(2))],
      outputs=[@plan.ActivationRef::new(3)],
      shape=AttentionShape(query_heads=4, kv_heads=2, head_dimension=4),
      required_capability=@plan.KernelCapabilityId::paged_causal_attention(),
      workspace_bytes=0L,
      kv_layer=layer,
      runtime_inputs=[
        StepCounts,
        QueryPositions,
        QueryRowOffsets,
        SequenceLengths,
        PageTableOffsets,
        PageTablePageIndices,
      ],
      persistent_state=KvCacheReadWrite(layer),
    ),
  ]
}

///|
fn numeric_v4_constraints() -> @plan.ShapeConstraints {
  try! @plan.ShapeConstraints::new(
    max_batch_rows=1,
    max_sequence_tokens=64,
    vocabulary_size=128,
    hidden_size=16,
    intermediate_size=32,
    attention_head_count=4,
    kv_head_count=2,
    head_dimension=4,
  )
}

///|
fn numeric_v4_plan(mode : Int) -> @plan.ModelPlan {
  let operations = numeric_v4_operations()
  let selected = [try! @plan.OperationId::new(1)]
  let numeric_schema = match mode {
    1 => @plan_test_fixture.synthetic_bf16_numeric_schema(operations)
    2 =>
      @plan_test_fixture.synthetic_fp8_w8a8_numeric_schema(operations, selected)
    3 =>
      @plan_test_fixture.synthetic_symmetric_i8_weight_only_numeric_schema_v1(
        operations, selected,
      )
    _ => abort("unknown numeric-v4 fixture mode")
  }
  try! @plan.ModelPlan::new_with_kv_execution(
    content_digest=test_content(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=numeric_v4_constraints(),
    kv_cache_geometry=catalog_test_kv_geometry(),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations~,
    final_output=@plan.ActivationRef::new(3),
    kv_execution=PagedKeyValue,
    numeric_schema~,
  )
}

///|
fn numeric_v4_digest(
  plan : @plan.ModelPlan,
  operation_index : Int,
) -> @numeric_contract.OperationExecutionDigest {
  let id = plan.operations()[operation_index].id()
  for binding in plan.numeric_binding().operations() {
    if binding.operation() == id {
      return binding.execution().digest()
    }
  }
  abort("numeric-v4 fixture lost an operation binding")
}

///|
fn numeric_v4_family(stable_id : Int) -> @catalog.AotKernelFamily {
  let digest = try! @catalog.AotArtifactDigest::from_sha256(
    "13579bdf02468ace13579bdf02468ace13579bdf02468ace13579bdf02468ace",
  )
  aot_family(digest, stable_id~)
}

///|
fn numeric_v4_entry(
  plan : @plan.ModelPlan,
  operation_index : Int,
  family_id : Int,
  compute_major? : Int = 8,
  entry_point_id? : Int = family_id,
) -> @catalog.CatalogEntry {
  let operation = plan.operations()[operation_index]
  let family = numeric_v4_family(family_id)
  let layer = try! @plan.KvLayerId::new(0)
  try! @catalog.CatalogEntry::new_paged_v4(
    operation_kind=operation.kind(),
    operation_shape=operation.shape(),
    capability=operation.required_capability(),
    execution_semantics=paged_v3_semantics(operation.kind(), layer),
    operation_execution_digest=numeric_v4_digest(plan, operation_index),
    aot_entry_point=@catalog.AotKernelEntryPoint::new(
      family~,
      entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(
        entry_point_id,
      ),
    ),
    compute_major~,
    compute_minor=0,
    implementation=ContentAddressedAot(family),
    workspace_bytes=0L,
    workspace_alignment=1L,
  )
}

///|
fn numeric_v4_layout(plan : @plan.ModelPlan) -> @device_layout.DeviceKvLayout {
  try! @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=8,
      max_arena_bytes=4096L,
    ),
  )
}

///|
fn numeric_v4_base_entries(
  bf16 : @plan.ModelPlan,
) -> Array[@catalog.CatalogEntry] {
  [
    numeric_v4_entry(bf16, 0, 1),
    numeric_v4_entry(bf16, 2, 2),
    numeric_v4_entry(bf16, 3, 3),
  ]
}

///|
test "catalog v4 selects BF16 FP8 and I8 only by exact numeric digest" {
  let bf16 = numeric_v4_plan(1)
  let fp8 = numeric_v4_plan(2)
  let i8 = numeric_v4_plan(3)
  let entries = numeric_v4_base_entries(bf16)
  // Deliberately place the non-BF16 variants first: catalog order is not
  // priority and cannot substitute a different numeric ABI.
  entries.push(numeric_v4_entry(i8, 1, 13, entry_point_id=103))
  entries.push(numeric_v4_entry(fp8, 1, 12, entry_point_id=102))
  entries.push(numeric_v4_entry(bf16, 1, 11, entry_point_id=101))
  let catalog = try! @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v4(),
    entries~,
  )
  let plans = [bf16, fp8, i8]
  let expected_families = [11, 12, 13]
  let expected_entry_points = [101, 102, 103]
  for index, plan in plans {
    let resolved = try! catalog.resolve_paged_v4(
      plan,
      numeric_v4_layout(plan),
      test_target(),
    )
    assert_eq(resolved.catalog_version(), @catalog.CatalogVersion::v4())
    let binding = resolved.bindings()[1]
    assert_eq(binding.semantic_version(), @catalog.KernelSemanticVersion::v4())
    assert_eq(
      binding.operation_execution_digest(),
      Some(numeric_v4_digest(plan, 1)),
    )
    guard binding.aot_entry_point() is Some(entry_point) else {
      fail("v4 binding lost its exact entry point")
    }
    assert_eq(
      entry_point.family().family_id().as_int(),
      expected_families[index],
    )
    assert_eq(
      entry_point.entry_point_id().as_int(),
      expected_entry_points[index],
    )
    assert_eq(
      entry_point.family().module_digest().sha256(),
      "13579bdf02468ace13579bdf02468ace13579bdf02468ace13579bdf02468ace",
    )
  }
}

///|
test "catalog v4 reports missing mismatch and duplicate exact digests" {
  let bf16 = numeric_v4_plan(1)
  let i8 = numeric_v4_plan(3)
  let base = numeric_v4_base_entries(bf16)
  let missing = try! @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v4(),
    entries=base,
  )
  try
    missing.resolve_paged_v4(bf16, numeric_v4_layout(bf16), test_target())
  catch {
    MissingKernel(id) => assert_eq(id.as_int(), 1)
    _ => fail("unexpected missing-v4 error")
  } noraise {
    _ => fail("v4 admitted a missing operation digest")
  }

  let mismatch_entries = numeric_v4_base_entries(bf16)
  mismatch_entries.push(numeric_v4_entry(i8, 1, 21))
  let mismatch = try! @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v4(),
    entries=mismatch_entries,
  )
  try
    mismatch.resolve_paged_v4(bf16, numeric_v4_layout(bf16), test_target())
  catch {
    MissingKernel(id) => assert_eq(id.as_int(), 1)
    _ => fail("unexpected numeric-mismatch error")
  } noraise {
    _ => fail("v4 substituted an I8 entry for BF16")
  }

  let duplicate_entries = numeric_v4_base_entries(bf16)
  duplicate_entries.push(numeric_v4_entry(bf16, 1, 31))
  duplicate_entries.push(numeric_v4_entry(bf16, 1, 32))
  let duplicate = try! @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v4(),
    entries=duplicate_entries,
  )
  try
    duplicate.resolve_paged_v4(bf16, numeric_v4_layout(bf16), test_target())
  catch {
    AmbiguousKernel(id) => assert_eq(id.as_int(), 1)
    _ => fail("unexpected duplicate-v4 error")
  } noraise {
    _ => fail("v4 assigned priority to duplicate exact entries")
  }
}

///|
test "catalog v4 family and entry identities cannot be substituted" {
  let bf16 = numeric_v4_plan(1)
  let i8 = numeric_v4_plan(3)
  let family_entry = numeric_v4_entry(bf16, 1, 41)
  let foreign_numeric = numeric_v4_entry(i8, 1, 41, entry_point_id=42)
  try
    @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v4(), entries=[
      family_entry, foreign_numeric,
    ])
  catch {
    InvalidCatalog(AotFamilySemanticMismatch) => ()
    _ => fail("unexpected v4 family-digest error")
  } noraise {
    _ => fail("one AOT family acquired two numeric ABIs")
  }

  let implementation_family = numeric_v4_family(51)
  let foreign_family = numeric_v4_family(52)
  let operation = bf16.operations()[1]
  let layer = try! @plan.KvLayerId::new(0)
  let semantics = try! paged_v3_semantics(operation.kind(), layer)
  let entry_point_id = try! @catalog.AotKernelEntryPointId::from_stable_int(1)
  let foreign_entry_point = @catalog.AotKernelEntryPoint::new(
    family=foreign_family,
    entry_point_id~,
  )
  try
    @catalog.CatalogEntry::new_paged_v4(
      operation_kind=operation.kind(),
      operation_shape=operation.shape(),
      capability=operation.required_capability(),
      execution_semantics=semantics,
      operation_execution_digest=numeric_v4_digest(bf16, 1),
      aot_entry_point=foreign_entry_point,
      compute_major=8,
      compute_minor=0,
      implementation=ContentAddressedAot(implementation_family),
      workspace_bytes=0L,
      workspace_alignment=1L,
    )
  catch {
    InvalidCatalog(AotEntryPointFamilyMismatch) => ()
    _ => fail("unexpected v4 entry-family error")
  } noraise {
    _ => fail("v4 admitted a foreign-family entry point")
  }

  let implementation_entry_point = @catalog.AotKernelEntryPoint::new(
    family=implementation_family,
    entry_point_id~,
  )
  try
    @catalog.CatalogEntry::new_paged_v4(
      operation_kind=operation.kind(),
      operation_shape=operation.shape(),
      capability=operation.required_capability(),
      execution_semantics=semantics,
      operation_execution_digest=numeric_v4_digest(bf16, 1),
      aot_entry_point=implementation_entry_point,
      compute_major=8,
      compute_minor=0,
      implementation=VendorBf16GemmImplicitHeuristicV1,
      workspace_bytes=0L,
      workspace_alignment=1L,
    )
  catch {
    InvalidCatalog(InvalidVendorImplementationKind) => ()
    _ => fail("unexpected v4 vendor error")
  } noraise {
    _ => fail("v4 admitted a vendor algorithm without content identity")
  }
}

///|
test "catalog v4 filters numeric identity before target compatibility" {
  let bf16 = numeric_v4_plan(1)
  let i8 = numeric_v4_plan(3)
  let entries = numeric_v4_base_entries(bf16)
  entries.push(numeric_v4_entry(i8, 1, 61))
  entries.push(numeric_v4_entry(bf16, 1, 62, compute_major=9))
  let catalog = try! @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v4(),
    entries~,
  )
  try
    catalog.resolve_paged_v4(bf16, numeric_v4_layout(bf16), test_target())
  catch {
    IncompatibleTarget(id) => assert_eq(id.as_int(), 1)
    _ => fail("numeric mismatch masked exact target incompatibility")
  } noraise {
    _ => fail("target-compatible I8 entry substituted for BF16")
  }
}

///|
test "opaque resolved values and bindings are minted by exact resolver" {
  let plan = numeric_v4_plan(3)
  let bf16 = numeric_v4_plan(1)
  let entries = numeric_v4_base_entries(bf16)
  entries.push(numeric_v4_entry(plan, 1, 74, entry_point_id=174))
  let catalog = try! @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v4(),
    entries~,
  )
  let resolved : @catalog.ResolvedKernelCatalog = try! catalog.resolve_paged_v4(
    plan,
    numeric_v4_layout(plan),
    test_target(),
  )
  let binding : @catalog.KernelBinding = resolved.bindings()[1]
  assert_eq(
    binding.operation_execution_digest(),
    Some(numeric_v4_digest(plan, 1)),
  )
  guard binding.aot_entry_point() is Some(entry_point) else {
    fail("resolver-minted binding lost its exact entry point")
  }
  assert_eq(entry_point.entry_point_id().as_int(), 174)
}
