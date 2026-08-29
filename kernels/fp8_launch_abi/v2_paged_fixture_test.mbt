///|
struct Fp8V2PagedFixture {
  plan : @plan.ModelPlan
  launches : @launch_contract.PagedV4LaunchContractSet
  kernels : @numeric_capability_manifest.AdmittedFp8KernelArtifacts
  weight_scales : Array[@fp8_launch_abi.Fp8LaunchStorageRegion]
  workspace : @fp8_launch_abi.Fp8LaunchStorageRegion
  shape : @fp8_launch_abi.Fp8ExactExecutionShape
}

///|
fn v2_paged_operations(gated : Bool) -> Array[@plan.PlanOperation] {
  let layer = try! @plan.KvLayerId::new(0)
  let selected = if gated {
    try! @plan.PlanOperation::new_with_runtime_inputs(
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
      workspace_bytes=8L,
      runtime_inputs=[StepCounts],
    )
  } else {
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
      workspace_bytes=4L,
      runtime_inputs=[StepCounts],
    )
  }
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
    selected,
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
fn v2_paged_plan(gated : Bool) -> @plan.ModelPlan {
  let operations = v2_paged_operations(gated)
  let numeric = if gated {
    v2_gated_schema(operations, true)
  } else {
    @plan_test_fixture.synthetic_fp8_w8a8_numeric_schema(operations, [
      try! @plan.OperationId::new(1),
    ])
  }
  try! @plan.ModelPlan::new_with_kv_execution(
    content_digest=@spec.ContentDigest::from_sha256(
      if gated {
        "3434343434343434343434343434343434343434343434343434343434343434"
      } else {
        "5656565656565656565656565656565656565656565656565656565656565656"
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
      max_temporary_bytes=if gated { 8L } else { 4L },
    ),
    operations~,
    final_output=@plan.ActivationRef::new(3),
    kv_execution=PagedKeyValue,
    numeric_schema=numeric,
  )
}

///|
fn v2_paged_layout(plan : @plan.ModelPlan) -> @device_layout.DeviceKvLayout {
  try! @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=4,
      max_arena_bytes=65536L,
    ),
  )
}

///|
fn v2_paged_profile() -> @launch_contract.PagedKernelProfile {
  try! @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=1,
    max_query_tokens=2,
    max_page_table_entries=2,
  )
}

///|
fn v2_paged_semantics(
  operation : @plan.PlanOperation,
) -> @catalog.KernelExecutionSemantics {
  let layout = if operation.kind() == CausalAttention {
    Some(
      try! @catalog.PagedKvLayoutContract::new(
        layout=LayerMajorSplitKeyValueV1,
        stable_version=1,
        tokens_per_page=4,
      ),
    )
  } else {
    None
  }
  try! @catalog.KernelExecutionSemantics::paged_v3(
    kv_layer=operation.kv_layer(),
    runtime_inputs=operation.runtime_inputs(),
    persistent_state=operation.persistent_state(),
    kv_layout=layout,
  )
}

///|
fn v2_paged_catalog(plan : @plan.ModelPlan) -> @catalog.ResolvedKernelCatalog {
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in plan.operations() {
    entries.push(
      try! @catalog.CatalogEntry::new_paged_v4(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=v2_paged_semantics(operation),
        operation_execution_digest=plan.numeric_binding().operations()[index]
          .execution()
          .digest(),
        aot_entry_point=fixture_entry(index + 1, 10 + index),
        compute_major=8,
        compute_minor=9,
        implementation=ContentAddressedAot(fixture_family(index + 1)),
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
    version=@catalog.CatalogVersion::v4(),
    entries~,
  ).resolve_paged_v4(plan, v2_paged_layout(plan), fixture_target())
}

///|
fn v2_paged_tensor_bytes(
  plan : @plan.ModelPlan,
  tensor : @plan.TensorRef,
) -> Int64 {
  guard plan.tensor(tensor) is Some(schema) else {
    abort("v2 paged fixture lost tensor")
  }
  try! @numeric_contract.storage_byte_length(
    schema.storage().dtype(),
    schema.shape(),
  )
}

///|
fn v2_paged_operand(
  role : @launch_contract.AotOperandRole,
  bytes : Int64,
  alignment : Int64,
) -> @launch_contract.AotOperand {
  try! @launch_contract.AotOperand::new(role~, byte_count=bytes, alignment~)
}

///|
fn v2_paged_operands(
  plan : @plan.ModelPlan,
  operation : Int,
) -> Array[@launch_contract.AotOperand] {
  if operation == 0 {
    return [
      v2_paged_operand(RuntimeInput(StepCounts), 20L, 4L),
      v2_paged_operand(ExternalInput(TokenIds), 8L, 4L),
      v2_paged_operand(
        WeightInput(try! @plan.TensorRef::new(0)),
        v2_paged_tensor_bytes(plan, try! @plan.TensorRef::new(0)),
        16L,
      ),
      v2_paged_operand(
        ActivationOutput(try! @plan.ActivationRef::new(0)),
        64L,
        16L,
      ),
    ]
  }
  if operation == 2 {
    return [
      v2_paged_operand(RuntimeInput(StepCounts), 20L, 4L),
      v2_paged_operand(RuntimeInput(QueryPositions), 8L, 4L),
      v2_paged_operand(
        ActivationInput(try! @plan.ActivationRef::new(1)),
        128L,
        16L,
      ),
      v2_paged_operand(
        ActivationOutput(try! @plan.ActivationRef::new(2)),
        128L,
        16L,
      ),
    ]
  }
  if operation == 3 {
    let layer = try! @plan.KvLayerId::new(0)
    let layout = v2_paged_layout(plan)
    return [
      v2_paged_operand(RuntimeInput(StepCounts), 20L, 4L),
      v2_paged_operand(RuntimeInput(QueryPositions), 8L, 4L),
      v2_paged_operand(RuntimeInput(QueryRowOffsets), 8L, 4L),
      v2_paged_operand(RuntimeInput(SequenceLengths), 4L, 4L),
      v2_paged_operand(RuntimeInput(PageTableOffsets), 8L, 4L),
      v2_paged_operand(RuntimeInput(PageTablePageIndices), 8L, 4L),
      v2_paged_operand(
        ActivationInput(try! @plan.ActivationRef::new(2)),
        128L,
        16L,
      ),
      v2_paged_operand(
        ActivationOutput(try! @plan.ActivationRef::new(3)),
        64L,
        16L,
      ),
      v2_paged_operand(
        PersistentKvComponent(layer, Key),
        layout.component_stride_bytes(),
        layout.segment_alignment_bytes(),
      ),
      v2_paged_operand(
        PersistentKvComponent(layer, Value),
        layout.component_stride_bytes(),
        layout.segment_alignment_bytes(),
      ),
    ]
  }
  let result = [
    v2_paged_operand(RuntimeInput(StepCounts), 20L, 4L),
    v2_paged_operand(
      ActivationInput(try! @plan.ActivationRef::new(0)),
      64L,
      16L,
    ),
  ]
  for input in plan.operations()[1].inputs() {
    guard input is Tensor(weight) else { continue }
    guard plan.tensor(weight) is Some(tensor) && tensor.scale() is Some(scale) else {
      abort("v2 paged FP8 weight lost scale")
    }
    result.push(
      v2_paged_operand(
        WeightInput(weight),
        v2_paged_tensor_bytes(plan, weight),
        16L,
      ),
    )
    result.push(
      v2_paged_operand(
        WeightScaleInput(weight, scale),
        v2_paged_tensor_bytes(plan, scale.tensor()),
        16L,
      ),
    )
  }
  result.push(
    v2_paged_operand(
      ActivationOutput(try! @plan.ActivationRef::new(1)),
      64L,
      16L,
    ),
  )
  result.push(
    v2_paged_operand(Workspace, plan.operations()[1].workspace_bytes(), 4L),
  )
  result
}

///|
fn v2_paged_launches(
  plan : @plan.ModelPlan,
  catalog : @catalog.ResolvedKernelCatalog,
  grid_x? : Int = 2,
  block_x? : Int = 32,
  shared_memory_bytes? : Int = 0,
  selected_operands? : Array[@launch_contract.AotOperand] = [],
) -> @launch_contract.PagedV4LaunchContractSet raise @launch_contract.LaunchContractError {
  let inputs : Array[@launch_contract.PagedV4AotLaunchContractInput] = []
  for operation in 0..<plan.operations().length() {
    inputs.push(
      try! @launch_contract.PagedV4AotLaunchContractInput::new(
        profile_id=v2_paged_profile().id(),
        operation_id=@plan.OperationId::new(operation),
        entry_point=fixture_entry(operation + 1, 10 + operation),
        dimensions=@launch_contract.AotLaunchDimensions::new(
          grid_x~,
          grid_y=1,
          grid_z=1,
          block_x~,
          block_y=1,
          block_z=1,
          shared_memory_bytes~,
        ),
        operands=if operation == 1 && !selected_operands.is_empty() {
          selected_operands
        } else {
          v2_paged_operands(plan, operation)
        },
      ),
    )
  }
  @launch_contract.admit_paged_graph_v4(
    plan,
    catalog,
    v2_paged_layout(plan),
    [v2_paged_profile()],
    inputs,
    @launch_contract.LaunchContractLimits::new(
      max_profiles=2,
      max_contracts=8,
      max_operands_per_contract=16,
    ),
  )
}

///|
