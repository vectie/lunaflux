///|
fn numeric_legacy_operations() -> Array[@plan.PlanOperation] {
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
      workspace_bytes=0L,
    ),
  ]
}

///|
fn numeric_legacy_plan(mode : Int) -> @plan.ModelPlan {
  let operations = numeric_legacy_operations()
  let selected = [try! @plan.OperationId::new(1)]
  let numeric_schema = match mode {
    1 => @plan_test_fixture.synthetic_bf16_numeric_schema(operations)
    2 =>
      @plan_test_fixture.synthetic_fp8_w8a8_numeric_schema(operations, selected)
    3 =>
      @plan_test_fixture.synthetic_symmetric_i8_weight_only_numeric_schema_v1(
        operations, selected,
      )
    _ => abort("unknown numeric-legacy fixture mode")
  }
  try! @plan.ModelPlan::new_stateless(
    content_digest=test_content(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=numeric_v4_constraints(),
    kv_cache_geometry=catalog_test_kv_geometry(),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations~,
    final_output=@plan.ActivationRef::new(1),
    numeric_schema~,
  )
}

///|
fn numeric_legacy_v1_entry(
  operation : @plan.PlanOperation,
  family_id : Int,
  compute_major? : Int = 8,
) -> @catalog.CatalogEntry {
  try! @catalog.CatalogEntry::new(
    semantic_version=@catalog.KernelSemanticVersion::v1(),
    operation_kind=operation.kind(),
    operation_shape=operation.shape(),
    capability=operation.required_capability(),
    compute_major~,
    compute_minor=0,
    implementation=ContentAddressedAot(numeric_v4_family(family_id)),
    workspace_bytes=0L,
    workspace_alignment=1L,
  )
}

///|
fn numeric_legacy_hostile_executions() -> Array[
  @numeric_contract.OperationExecutionContract,
] {
  let exact = @numeric_contract.NumericConversion::exact_v1()
  let order = @numeric_contract.OperationOrderPolicy::strict_declared_order_v1()
  let no_scale = @numeric_contract.ActivationScalePolicy::absent_exact_v1()
  [
    // Activation input and compute must be equal for an exact constructible
    // contract, so this one hostile contract covers both I8 positions.
    try! @numeric_contract.OperationExecutionContract::new(
      activation_input=@numeric_contract.ComputeDType::i8(),
      activation_compute=@numeric_contract.ComputeDType::i8(),
      activation_scale=no_scale,
      tensor_input=@numeric_contract.ComputeDType::bfloat16(),
      accumulator=@numeric_contract.AccumulatorDType::f32(),
      output=@numeric_contract.ComputeDType::bfloat16(),
      conversion=exact,
      order~,
    ),
    try! @numeric_contract.OperationExecutionContract::new(
      activation_input=@numeric_contract.ComputeDType::bfloat16(),
      activation_compute=@numeric_contract.ComputeDType::bfloat16(),
      activation_scale=no_scale,
      tensor_input=@numeric_contract.ComputeDType::bfloat16(),
      accumulator=@numeric_contract.AccumulatorDType::i32(),
      output=@numeric_contract.ComputeDType::bfloat16(),
      conversion=exact,
      order~,
    ),
    try! @numeric_contract.OperationExecutionContract::new(
      activation_input=@numeric_contract.ComputeDType::bfloat16(),
      activation_compute=@numeric_contract.ComputeDType::bfloat16(),
      activation_scale=no_scale,
      tensor_input=@numeric_contract.ComputeDType::bfloat16(),
      accumulator=@numeric_contract.AccumulatorDType::f32(),
      output=@numeric_contract.ComputeDType::i8(),
      conversion=exact,
      order~,
    ),
  ]
}

///|
fn numeric_legacy_i8_producer_execution() -> @numeric_contract.OperationExecutionContract {
  try! @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::absent(),
    activation_compute=@numeric_contract.ComputeDType::absent(),
    activation_scale=@numeric_contract.ActivationScalePolicy::absent_exact_v1(),
    tensor_input=@numeric_contract.ComputeDType::bfloat16(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::i8(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
}

///|
fn numeric_legacy_i8_consumer_execution() -> @numeric_contract.OperationExecutionContract {
  try! @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::i8(),
    activation_compute=@numeric_contract.ComputeDType::i8(),
    activation_scale=@numeric_contract.ActivationScalePolicy::absent_exact_v1(),
    tensor_input=@numeric_contract.ComputeDType::absent(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
}

///|
fn numeric_schema_with_hostile_execution(
  operations : ArrayView[@plan.PlanOperation],
  hostile : @numeric_contract.OperationExecutionContract,
  hostile_index : Int,
) -> @numeric_contract.ModelNumericSchema {
  let base = @plan_test_fixture.synthetic_bf16_numeric_schema(operations)
  let operation_schemas : Array[@numeric_contract.NumericOperationSchema] = []
  for index, operation in base.operations() {
    let execution = if hostile_index == 0 && index == 0 {
      numeric_legacy_i8_producer_execution()
    } else if index == 1 {
      hostile
    } else if hostile_index == 2 && index == 2 {
      numeric_legacy_i8_consumer_execution()
    } else {
      operation.execution()
    }
    operation_schemas.push(
      @numeric_contract.NumericOperationSchema::new(
        ordinal=operation.ordinal(),
        execution~,
      ),
    )
  }
  try! @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors=base.tensors(),
    operations=operation_schemas,
  )
}

///|
fn numeric_legacy_plan_with_hostile_execution(
  hostile : @numeric_contract.OperationExecutionContract,
  hostile_index : Int,
) -> @plan.ModelPlan {
  let operations = numeric_legacy_operations()
  try! @plan.ModelPlan::new_stateless(
    content_digest=test_content(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=numeric_v4_constraints(),
    kv_cache_geometry=catalog_test_kv_geometry(),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations~,
    final_output=@plan.ActivationRef::new(1),
    numeric_schema=numeric_schema_with_hostile_execution(
      operations, hostile, hostile_index,
    ),
  )
}

///|
fn numeric_paged_plan_with_hostile_execution(
  hostile : @numeric_contract.OperationExecutionContract,
  hostile_index : Int,
) -> @plan.ModelPlan {
  let operations = numeric_v4_operations()
  try! @plan.ModelPlan::new_with_kv_execution(
    content_digest=test_content(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=numeric_v4_constraints(),
    kv_cache_geometry=catalog_test_kv_geometry(),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations~,
    final_output=@plan.ActivationRef::new(3),
    kv_execution=PagedKeyValue,
    numeric_schema=numeric_schema_with_hostile_execution(
      operations, hostile, hostile_index,
    ),
  )
}

///|
test "legacy v1 and v3 reject every constructible hostile integer dtype field" {
  for hostile_index, hostile in numeric_legacy_hostile_executions() {
    let expected_operation = if hostile_index == 0 { 0 } else { 1 }
    let stateless = numeric_legacy_plan_with_hostile_execution(
      hostile, hostile_index,
    )
    let v1 = try! @catalog.KernelCatalog::new(
      version=@catalog.CatalogVersion::v1(),
      entries=[
        numeric_legacy_v1_entry(stateless.operations()[0], 69, compute_major=9),
        numeric_legacy_v1_entry(stateless.operations()[0], 68),
        numeric_legacy_v1_entry(stateless.operations()[0], 67),
      ],
    )
    try v1.resolve(stateless, test_target()) catch {
      UnsupportedNumericContract(id) =>
        assert_eq(id.as_int(), expected_operation)
      _ => fail("unexpected hostile legacy-v1 numeric error")
    } noraise {
      _ => fail("legacy v1 reached target or entry selection")
    }

    let paged = numeric_paged_plan_with_hostile_execution(
      hostile, hostile_index,
    )
    let operation = paged.operations()[0]
    let v3_entry = try! @catalog.CatalogEntry::new_paged_v3(
      operation_kind=operation.kind(),
      operation_shape=operation.shape(),
      capability=operation.required_capability(),
      execution_semantics=paged_v3_semantics(
        operation.kind(),
        @plan.KvLayerId::new(0),
      ),
      compute_major=9,
      compute_minor=0,
      implementation=ContentAddressedAot(numeric_v4_family(70)),
      workspace_bytes=0L,
      workspace_alignment=1L,
    )
    let v3_duplicate = try! @catalog.CatalogEntry::new_paged_v3(
      operation_kind=operation.kind(),
      operation_shape=operation.shape(),
      capability=operation.required_capability(),
      execution_semantics=paged_v3_semantics(
        operation.kind(),
        @plan.KvLayerId::new(0),
      ),
      compute_major=8,
      compute_minor=0,
      implementation=ContentAddressedAot(numeric_v4_family(71)),
      workspace_bytes=0L,
      workspace_alignment=1L,
    )
    let v3_duplicate_two = try! @catalog.CatalogEntry::new_paged_v3(
      operation_kind=operation.kind(),
      operation_shape=operation.shape(),
      capability=operation.required_capability(),
      execution_semantics=paged_v3_semantics(
        operation.kind(),
        @plan.KvLayerId::new(0),
      ),
      compute_major=8,
      compute_minor=0,
      implementation=ContentAddressedAot(numeric_v4_family(73)),
      workspace_bytes=0L,
      workspace_alignment=1L,
    )
    let v3 = try! @catalog.KernelCatalog::new(
      version=@catalog.CatalogVersion::v3(),
      entries=[v3_entry, v3_duplicate, v3_duplicate_two],
    )
    try v3.resolve_paged(paged, numeric_v4_layout(paged), test_target()) catch {
      UnsupportedNumericContract(id) =>
        assert_eq(id.as_int(), expected_operation)
      _ => fail("unexpected hostile legacy-v3 numeric error")
    } noraise {
      _ => fail("legacy v3 reached target or entry selection")
    }
  }
}

///|
test "legacy v1 and v3 reject I8 before architecture selection" {
  let stateless = numeric_legacy_plan(3)
  let v1 = try! @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v1(),
    entries=[
      numeric_legacy_v1_entry(stateless.operations()[0], 71, compute_major=9),
    ],
  )
  try v1.resolve(stateless, test_target()) catch {
    UnsupportedNumericContract(id) => assert_eq(id.as_int(), 1)
    _ => fail("unexpected legacy-v1 I8 error")
  } noraise {
    _ => fail("legacy v1 reached target selection for I8")
  }

  let paged = numeric_v4_plan(3)
  let operation = paged.operations()[0]
  let v3_entry = try! @catalog.CatalogEntry::new_paged_v3(
    operation_kind=operation.kind(),
    operation_shape=operation.shape(),
    capability=operation.required_capability(),
    execution_semantics=paged_v3_semantics(
      operation.kind(),
      @plan.KvLayerId::new(0),
    ),
    compute_major=9,
    compute_minor=0,
    implementation=ContentAddressedAot(numeric_v4_family(72)),
    workspace_bytes=0L,
    workspace_alignment=1L,
  )
  let v3 = try! @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v3(),
    entries=[v3_entry],
  )
  try v3.resolve_paged(paged, numeric_v4_layout(paged), test_target()) catch {
    UnsupportedNumericContract(id) => assert_eq(id.as_int(), 1)
    _ => fail("unexpected legacy-v3 I8 error")
  } noraise {
    _ => fail("legacy v3 reached target selection for I8")
  }
}

///|
test "legacy BF16 and FP8 resolution remain unbound and unchanged" {
  for mode in 1..<=2 {
    let plan = numeric_legacy_plan(mode)
    let catalog = try! @catalog.KernelCatalog::new(
      version=@catalog.CatalogVersion::v1(),
      entries=[
        numeric_legacy_v1_entry(plan.operations()[0], 80 + mode * 2),
        numeric_legacy_v1_entry(plan.operations()[1], 81 + mode * 2),
      ],
    )
    let resolved = try! catalog.resolve(plan, test_target())
    assert_eq(resolved.catalog_version(), @catalog.CatalogVersion::v1())
    assert_eq(resolved.bindings()[1].operation_execution_digest(), None)
    assert_eq(resolved.bindings()[1].aot_entry_point(), None)
  }
}

///|
test "catalog versions reject legacy and exact entries in both directions" {
  let plan = numeric_v4_plan(1)
  let operation = plan.operations()[0]
  let legacy = try! @catalog.CatalogEntry::new_paged_v3(
    operation_kind=operation.kind(),
    operation_shape=operation.shape(),
    capability=operation.required_capability(),
    execution_semantics=paged_v3_semantics(
      operation.kind(),
      @plan.KvLayerId::new(0),
    ),
    compute_major=8,
    compute_minor=0,
    implementation=ContentAddressedAot(numeric_v4_family(91)),
    workspace_bytes=0L,
    workspace_alignment=1L,
  )
  let exact = numeric_v4_entry(plan, 0, 92)
  try
    @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v4(), entries=[
      legacy,
    ])
  catch {
    InvalidCatalog(CatalogSemanticVersionMismatch) => ()
    _ => fail("unexpected legacy-in-v4 error")
  } noraise {
    _ => fail("catalog v4 admitted a legacy entry")
  }
  try
    @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v3(), entries=[
      exact,
    ])
  catch {
    InvalidCatalog(CatalogSemanticVersionMismatch) => ()
    _ => fail("unexpected v4-in-legacy error")
  } noraise {
    _ => fail("catalog v3 admitted an exact v4 entry")
  }
}
