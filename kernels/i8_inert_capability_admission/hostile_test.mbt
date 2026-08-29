///|
fn hostile_i8_schema(
  base : @plan.ModelPlan,
  selected_operation : Int,
  partial? : Bool = false,
) -> @numeric_contract.ModelNumericSchema {
  let tensor_count = base.numeric_binding().tensors().length()
  let i8_tensors : Array[Bool] = []
  for _ in 0..<tensor_count {
    i8_tensors.push(false)
  }
  let operation = base.operations()[selected_operation]
  let mut selected_tensor_count = 0
  for input in operation.inputs() {
    match input {
      Tensor(reference) => {
        if !partial || selected_tensor_count == 0 {
          i8_tensors[reference.as_int()] = true
        }
        selected_tensor_count += 1
      }
      ModelInput(_) | Activation(_) => ()
    }
  }
  let tensors : Array[@numeric_contract.NumericTensorSchema] = []
  let mut scale_count = 0
  for index, tensor in base.numeric_binding().tensors() {
    if i8_tensors[index] {
      tensors.push(
        try! @numeric_contract.NumericTensorSchema::new(
          ordinal=@numeric_contract.NumericTensorOrdinal::new(index),
          role=@numeric_contract.NumericTensorRole::parameter(),
          shape=tensor.shape(),
          storage=@numeric_contract.TensorStorageContract::symmetric_i8_weight_only_v1(),
          scale=@numeric_contract.ScaleTensorOrdinal::new(
            @numeric_contract.NumericTensorOrdinal::new(
              tensor_count + scale_count,
            ),
          ),
        ),
      )
      scale_count += 1
    } else {
      tensors.push(
        try! @numeric_contract.NumericTensorSchema::new(
          ordinal=@numeric_contract.NumericTensorOrdinal::new(index),
          role=@numeric_contract.NumericTensorRole::parameter(),
          shape=tensor.shape(),
          storage=tensor.storage(),
        ),
      )
    }
  }
  let mut next_scale = 0
  for index, tensor in base.numeric_binding().tensors() {
    if i8_tensors[index] {
      guard tensor.shape().dimension(0) is Some(output_channels) else {
        abort("hostile I8 tensor lost its output axis")
      }
      tensors.push(
        try! @numeric_contract.NumericTensorSchema::new(
          ordinal=@numeric_contract.NumericTensorOrdinal::new(
            tensor_count + next_scale,
          ),
          role=@numeric_contract.NumericTensorRole::scale_metadata(),
          shape=@numeric_contract.NumericShape::new([output_channels]),
          storage=@numeric_contract.TensorStorageContract::symmetric_i8_weight_only_scale_v1(),
        ),
      )
      next_scale += 1
    }
  }
  let operations : Array[@numeric_contract.NumericOperationSchema] = []
  for index, binding in base.numeric_binding().operations() {
    let ordinal = try! @numeric_contract.NumericOperationOrdinal::new(index)
    operations.push(
      @numeric_contract.NumericOperationSchema::new(
        ordinal~,
        execution=if index == selected_operation {
          @numeric_contract.OperationExecutionContract::symmetric_i8_weight_only_v1()
        } else {
          binding.execution()
        },
      ),
    )
  }
  try! @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors~,
    operations~,
  )
}

///|
fn hostile_i8_plan(
  base : @plan.ModelPlan,
  schema : @numeric_contract.ModelNumericSchema,
) -> @plan.ModelPlan raise @plan.PlanValidationError {
  @plan.ModelPlan::new_with_kv_execution(
    content_digest=base.identity().content(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=base.shape_constraints(),
    kv_cache_geometry=base.kv_cache_geometry(),
    workspace_bounds=base.workspace_bounds(),
    operations=base.operations(),
    final_output=base.final_output(),
    kv_execution=PagedKeyValue,
    numeric_schema=schema,
  )
}

///|
test "unsupported I8 operation cannot become an admitted ModelPlan" {
  let bf16 = i8_admission_plan(3)
  let schema = hostile_i8_schema(bf16, 0)
  try hostile_i8_plan(bf16, schema) catch {
    InvalidGraph(NumericOperationBinding) => ()
    _ => fail("unexpected unsupported-I8 plan error")
  } noraise {
    _ => fail("embedding acquired I8 plan authority")
  }
}

///|
test "mixed I8 and BF16 operands cannot become a ModelPlan" {
  let bf16 = i8_admission_plan(3)
  let mixed = hostile_i8_schema(bf16, 1, partial=true)
  try hostile_i8_plan(bf16, mixed) catch {
    InvalidGraph(NumericTensorType) => ()
    _ => fail("unexpected mixed numeric-plan error")
  } noraise {
    _ => fail("mixed I8/BF16 operation reached catalog admission")
  }
}

///|
test "missing shared and wrongly shaped scales fail before admission" {
  let weight_storage = @numeric_contract.TensorStorageContract::symmetric_i8_weight_only_v1()
  let scale_storage = @numeric_contract.TensorStorageContract::symmetric_i8_weight_only_scale_v1()
  try
    @numeric_contract.NumericTensorSchema::new(
      ordinal=@numeric_contract.NumericTensorOrdinal::new(0),
      role=@numeric_contract.NumericTensorRole::parameter(),
      shape=@numeric_contract.NumericShape::new([4L, 4L]),
      storage=weight_storage,
    )
  catch {
    InvalidNumericContract(StorageCombination) => ()
    _ => fail("unexpected missing-scale error")
  } noraise {
    _ => fail("I8 weight omitted its scale")
  }
  let shared_ordinal = try! @numeric_contract.NumericTensorOrdinal::new(2)
  let shared = @numeric_contract.ScaleTensorOrdinal::new(shared_ordinal)
  let weight0 = try! @numeric_contract.NumericTensorSchema::new(
    ordinal=@numeric_contract.NumericTensorOrdinal::new(0),
    role=@numeric_contract.NumericTensorRole::parameter(),
    shape=@numeric_contract.NumericShape::new([4L, 4L]),
    storage=weight_storage,
    scale=shared,
  )
  let weight1 = try! @numeric_contract.NumericTensorSchema::new(
    ordinal=@numeric_contract.NumericTensorOrdinal::new(1),
    role=@numeric_contract.NumericTensorRole::parameter(),
    shape=@numeric_contract.NumericShape::new([4L, 4L]),
    storage=weight_storage,
    scale=shared,
  )
  let bad_scale = try! @numeric_contract.NumericTensorSchema::new(
    ordinal=@numeric_contract.NumericTensorOrdinal::new(2),
    role=@numeric_contract.NumericTensorRole::scale_metadata(),
    shape=@numeric_contract.NumericShape::new([3L]),
    storage=scale_storage,
  )
  try
    @numeric_contract.ModelNumericSchema::new(
      limits=@numeric_contract.ModelNumericSchemaLimits::production(),
      tensors=[weight0, weight1, bad_scale],
      operations=[],
    )
  catch {
    InvalidNumericContract(StorageCombination) => ()
    _ => fail("unexpected duplicate-or-shape scale error")
  } noraise {
    _ => fail("shared wrongly shaped scale reached admission")
  }
}

///|
test "duplicate exact catalog evidence has no priority" {
  let plan = i8_admission_plan(1)
  let entries : Array[@catalog.CatalogEntry] = []
  for index in 0..<plan.operations().length() {
    entries.push(
      i8_admission_entry(
        plan, index, "13579bdf02468ace13579bdf02468ace13579bdf02468ace13579bdf02468ace",
      ),
    )
  }
  entries.push(
    i8_admission_entry(
      plan, 1, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ),
  )
  let catalog = try! @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v4(),
    entries~,
  )
  try
    catalog.resolve_paged_v4(
      plan,
      i8_admission_layout(plan),
      i8_admission_target(),
    )
  catch {
    AmbiguousKernel(id) => assert_eq(id.as_int(), 1)
    _ => fail("unexpected duplicate-catalog error")
  } noraise {
    _ => fail("duplicate exact catalog evidence acquired order priority")
  }
}

///|
test "vendor and foreign-family entries are structurally rejected upstream" {
  let plan = i8_admission_plan(1)
  let operation = plan.operations()[1]
  let family = i8_admission_family(
    "13579bdf02468ace13579bdf02468ace13579bdf02468ace13579bdf02468ace", 99,
  )
  let foreign = i8_admission_family(
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 100,
  )
  let foreign_entry_id = try! @catalog.AotKernelEntryPointId::from_stable_int(1)
  let entry_point = @catalog.AotKernelEntryPoint::new(
    family=foreign,
    entry_point_id=foreign_entry_id,
  )
  let semantics = i8_admission_semantics(plan, operation)
  try
    @catalog.CatalogEntry::new_paged_v4(
      operation_kind=operation.kind(),
      operation_shape=operation.shape(),
      capability=operation.required_capability(),
      execution_semantics=semantics,
      operation_execution_digest=i8_admission_operation_digest(plan, 1),
      aot_entry_point=entry_point,
      compute_major=8,
      compute_minor=9,
      implementation=ContentAddressedAot(family),
      workspace_bytes=32L,
      workspace_alignment=16L,
    )
  catch {
    InvalidCatalog(AotEntryPointFamilyMismatch) => ()
    _ => fail("unexpected foreign-family error")
  } noraise {
    _ => fail("foreign entry-point family entered catalog v4")
  }
  let valid_entry_id = try! @catalog.AotKernelEntryPointId::from_stable_int(2)
  let valid_entry_point = @catalog.AotKernelEntryPoint::new(
    family~,
    entry_point_id=valid_entry_id,
  )
  try
    @catalog.CatalogEntry::new_paged_v4(
      operation_kind=operation.kind(),
      operation_shape=operation.shape(),
      capability=operation.required_capability(),
      execution_semantics=semantics,
      operation_execution_digest=i8_admission_operation_digest(plan, 1),
      aot_entry_point=valid_entry_point,
      compute_major=8,
      compute_minor=9,
      implementation=VendorBf16GemmImplicitHeuristicV1,
      workspace_bytes=32L,
      workspace_alignment=16L,
    )
  catch {
    InvalidCatalog(InvalidVendorImplementationKind) => ()
    _ => fail("unexpected vendor-entry error")
  } noraise {
    _ => fail("vendor fallback entered catalog v4")
  }
}
