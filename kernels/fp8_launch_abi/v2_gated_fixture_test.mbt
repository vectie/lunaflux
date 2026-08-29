///|
fn v2_gated_operations(workspace : Int64) -> Array[@plan.PlanOperation] {
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
      kind=GatedMlp,
      inputs=[
        Activation(@plan.ActivationRef::new(0)),
        Tensor(@plan.TensorRef::new(1)),
        Tensor(@plan.TensorRef::new(2)),
        Tensor(@plan.TensorRef::new(3)),
      ],
      outputs=[@plan.ActivationRef::new(1)],
      shape=GatedMlpShape(hidden_size=16, intermediate_size=32),
      required_capability=@plan.KernelCapabilityId::gated_mlp(),
      workspace_bytes=workspace,
    ),
  ]
}

///|
fn v2_compound_execution() -> @numeric_contract.OperationExecutionContract {
  try! @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::bfloat16(),
    activation_compute=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    activation_scale=@numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_bound_expf_f32_v2(),
    tensor_input=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::quantized_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
}

///|
fn v2_gated_schema(
  operations : ArrayView[@plan.PlanOperation],
  compound : Bool,
) -> @numeric_contract.ModelNumericSchema {
  let source = @plan_test_fixture.synthetic_fp8_w8a8_numeric_schema(
    operations,
    [try! @plan.OperationId::new(1)],
  )
  if !compound {
    return source
  }
  let tensors : Array[@numeric_contract.NumericTensorSchema] = []
  for tensor in source.tensors() {
    tensors.push(tensor)
  }
  let numeric_operations : Array[@numeric_contract.NumericOperationSchema] = []
  for index, operation in source.operations() {
    numeric_operations.push(
      if index == 1 {
        @numeric_contract.NumericOperationSchema::new(
          ordinal=try! @numeric_contract.NumericOperationOrdinal::new(index),
          execution=v2_compound_execution(),
        )
      } else {
        operation
      },
    )
  }
  try! @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors~,
    operations=numeric_operations,
  )
}

///|
fn v2_gated_plan(
  compound? : Bool = true,
  workspace? : Int64 = 8L,
) -> @plan.ModelPlan {
  let operations = v2_gated_operations(workspace)
  try! @plan.ModelPlan::new_stateless(
    content_digest=@spec.ContentDigest::from_sha256(
      if compound {
        "abababababababababababababababababababababababababababababababab"
      } else {
        "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd"
      },
    ),
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
      max_temporary_bytes=if workspace > 0L { workspace } else { 1L },
    ),
    operations~,
    final_output=@plan.ActivationRef::new(1),
    numeric_schema=v2_gated_schema(operations, compound),
  )
}

///|
fn v2_gated_operands(
  operation : Int,
  sequence_tokens : Int,
  workspace : Int64,
) -> Array[@launch_contract.AotOperand] {
  let activation_bytes = sequence_tokens.to_int64() * 32L
  if operation == 0 {
    fixture_operands(v2_gated_operations(workspace)[0], sequence_tokens~)
  } else {
    [
      try! @launch_contract.AotOperand::new(
        role=ActivationInput(@plan.ActivationRef::new(0)),
        byte_count=activation_bytes,
        alignment=16L,
      ),
      try! @launch_contract.AotOperand::new(
        role=WeightInput(@plan.TensorRef::new(1)),
        byte_count=512L,
        alignment=16L,
      ),
      try! @launch_contract.AotOperand::new(
        role=WeightInput(@plan.TensorRef::new(2)),
        byte_count=512L,
        alignment=16L,
      ),
      try! @launch_contract.AotOperand::new(
        role=WeightInput(@plan.TensorRef::new(3)),
        byte_count=512L,
        alignment=16L,
      ),
      try! @launch_contract.AotOperand::new(
        role=ActivationOutput(@plan.ActivationRef::new(1)),
        byte_count=activation_bytes,
        alignment=16L,
      ),
      try! @launch_contract.AotOperand::new(
        role=Workspace,
        byte_count=workspace,
        alignment=if workspace == 8L { 4L } else { 16L },
      ),
    ]
  }
}

///|
fn v2_gated_launches(
  plan : @plan.ModelPlan,
  catalog : @catalog.ResolvedKernelCatalog,
) -> @launch_contract.LaunchContractSet {
  let workspace = plan.operations()[1].workspace_bytes()
  let inputs : Array[@launch_contract.AotLaunchContractInput] = []
  for profile in 1..<=2 {
    let tokens = profile * 2
    for operation in 0..<2 {
      inputs.push(
        try! @launch_contract.AotLaunchContractInput::new(
          profile_id=@launch_contract.KernelProfileId::from_stable_int(profile),
          operation_id=@plan.OperationId::new(operation),
          entry_point=fixture_entry(operation + 1, 10 + operation),
          dimensions=@launch_contract.AotLaunchDimensions::new(
            grid_x=2,
            grid_y=1,
            grid_z=1,
            block_x=32,
            block_y=1,
            block_z=1,
            shared_memory_bytes=0,
          ),
          operands=v2_gated_operands(operation, tokens, workspace),
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
      max_operands_per_contract=8,
    ),
  )
}

///|
fn v2_gated_fixture(
  compound? : Bool = true,
  workspace? : Int64 = 8L,
) -> Fp8AbiFixture {
  let plan = v2_gated_plan(compound~, workspace~)
  let catalog = fixture_catalog(plan)
  let launches = v2_gated_launches(plan, catalog)
  {
    plan,
    launches,
    kernels: fixture_kernels(plan, catalog, launches),
    weight_scale: fixture_region(0L, 4L, 4L),
    scale: fixture_region(4L, 4L, 4L),
    workspace: fixture_region(16L, workspace, 4L, workspace=true),
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
fn v2_gated_weight_scales() -> Array[@fp8_launch_abi.Fp8LaunchStorageRegion] {
  let storage = try! @fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(
    "7878787878787878787878787878787878787878787878787878787878787878",
  )
  let result : Array[@fp8_launch_abi.Fp8LaunchStorageRegion] = []
  for index in 0..<3 {
    result.push(
      try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
        storage~,
        storage_bytes=12L,
        offset=index.to_int64() * 4L,
        byte_length=4L,
        alignment=4L,
      ),
    )
  }
  result
}

///|
