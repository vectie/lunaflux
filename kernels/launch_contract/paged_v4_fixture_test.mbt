///|
fn launch_v4_content(
  digest? : String = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
) -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(digest)
}

///|
fn launch_v4_operations() -> Array[@plan.PlanOperation] {
  let layer = try! @plan.KvLayerId::new(0)
  [
    try! @plan.PlanOperation::new_with_runtime_inputs(
      id=@plan.OperationId::new(0),
      kind=Embedding,
      inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
      outputs=[@plan.ActivationRef::new(0)],
      shape=EmbeddingShape(vocabulary_size=8, width=1),
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
      shape=ProjectionShape(input_width=1, output_width=1),
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
        query_heads=1,
        kv_heads=1,
        head_dimension=1,
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
      shape=AttentionShape(query_heads=1, kv_heads=1, head_dimension=1),
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
fn launch_v4_plan(
  mode : Int,
  content_digest? : String = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
) -> @plan.ModelPlan {
  let operations = launch_v4_operations()
  let selected = [try! @plan.OperationId::new(1)]
  let numeric_schema = match mode {
    1 => @plan_test_fixture.synthetic_bf16_numeric_schema(operations)
    2 =>
      @plan_test_fixture.synthetic_fp8_w8a8_numeric_schema(operations, selected)
    3 =>
      @plan_test_fixture.synthetic_symmetric_i8_weight_only_numeric_schema_v1(
        operations, selected,
      )
    _ => abort("unknown launch-v4 numeric mode")
  }
  try! @plan.ModelPlan::new_with_kv_execution(
    content_digest=launch_v4_content(digest=content_digest),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows=1,
      max_sequence_tokens=8,
      vocabulary_size=8,
      hidden_size=1,
      intermediate_size=2,
      attention_head_count=1,
      kv_head_count=1,
      head_dimension=1,
    ),
    kv_cache_geometry=@plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=1,
      head_dimension=1,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations~,
    final_output=@plan.ActivationRef::new(3),
    kv_execution=PagedKeyValue,
    numeric_schema~,
  )
}

///|
fn launch_v4_layout(plan : @plan.ModelPlan) -> @device_layout.DeviceKvLayout {
  try! @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=4,
      max_arena_bytes=4096L,
    ),
  )
}

///|
fn launch_v4_profile() -> @launch_contract.PagedKernelProfile {
  try! @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=1,
    max_query_tokens=4,
    max_page_table_entries=2,
  )
}

///|
fn launch_v4_limits() -> @launch_contract.LaunchContractLimits {
  try! @launch_contract.LaunchContractLimits::new(
    max_profiles=4,
    max_contracts=32,
    max_operands_per_contract=20,
  )
}

///|
fn launch_v4_semantics(
  operation : @plan.PlanOperation,
) -> @catalog.KernelExecutionSemantics {
  let layer = try! @plan.KvLayerId::new(0)
  let kv_layout = if operation.kind() == CausalAttention {
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
    kv_layer=if operation.kind() is (RotaryEmbedding | CausalAttention) {
      Some(layer)
    } else {
      None
    },
    runtime_inputs=operation.runtime_inputs(),
    persistent_state=operation.persistent_state(),
    kv_layout~,
  )
}

///|
fn launch_v4_family(index : Int) -> @catalog.AotKernelFamily {
  @catalog.AotKernelFamily::new(
    module_digest=try! @catalog.AotArtifactDigest::from_sha256(
      "13579bdf02468ace13579bdf02468ace13579bdf02468ace13579bdf02468ace",
    ),
    family_id=try! @catalog.AotKernelFamilyId::from_stable_int(index + 1),
  )
}

///|
fn launch_v4_entry_point(index : Int) -> @catalog.AotKernelEntryPoint {
  @catalog.AotKernelEntryPoint::new(
    family=launch_v4_family(index),
    entry_point_id=try! @catalog.AotKernelEntryPointId::from_stable_int(
      index + 101,
    ),
  )
}

///|
fn launch_v4_catalog(plan : @plan.ModelPlan) -> @catalog.ResolvedKernelCatalog {
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in plan.operations() {
    entries.push(
      try! @catalog.CatalogEntry::new_paged_v4(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=launch_v4_semantics(operation),
        operation_execution_digest=plan.numeric_binding().operations()[index]
          .execution()
          .digest(),
        aot_entry_point=launch_v4_entry_point(index),
        compute_major=8,
        compute_minor=9,
        implementation=ContentAddressedAot(launch_v4_family(index)),
        workspace_bytes=0L,
        workspace_alignment=1L,
      ),
    )
  }
  let catalog = try! @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v4(),
    entries~,
  )
  try! catalog.resolve_paged_v4(
    plan,
    launch_v4_layout(plan),
    @catalog.DeviceTarget::new(
      compute_major=8,
      compute_minor=9,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
  )
}

///|
fn launch_v4_operand(
  role : @launch_contract.AotOperandRole,
  byte_count : Int64,
  alignment : Int64,
) -> @launch_contract.AotOperand {
  try! @launch_contract.AotOperand::new(role~, byte_count~, alignment~)
}

///|
fn launch_v4_tensor_bytes(
  plan : @plan.ModelPlan,
  reference : @plan.TensorRef,
) -> Int64 {
  guard plan.tensor(reference) is Some(tensor) else {
    abort("launch-v4 fixture lost tensor")
  }
  try! @numeric_contract.storage_byte_length(
    tensor.storage().dtype(),
    tensor.shape(),
  )
}

///|
fn launch_v4_operands(
  plan : @plan.ModelPlan,
  operation : Int,
) -> Array[@launch_contract.AotOperand] {
  let layer = try! @plan.KvLayerId::new(0)
  match operation {
    0 =>
      [
        launch_v4_operand(RuntimeInput(StepCounts), 20L, 4L),
        launch_v4_operand(ExternalInput(TokenIds), 16L, 4L),
        launch_v4_operand(
          WeightInput(try! @plan.TensorRef::new(0)),
          launch_v4_tensor_bytes(plan, try! @plan.TensorRef::new(0)),
          16L,
        ),
        launch_v4_operand(
          ActivationOutput(try! @plan.ActivationRef::new(0)),
          8L,
          16L,
        ),
      ]
    1 => {
      let weight = try! @plan.TensorRef::new(1)
      let result = [
        launch_v4_operand(RuntimeInput(StepCounts), 20L, 4L),
        launch_v4_operand(
          ActivationInput(try! @plan.ActivationRef::new(0)),
          8L,
          16L,
        ),
        launch_v4_operand(
          WeightInput(weight),
          launch_v4_tensor_bytes(plan, weight),
          16L,
        ),
      ]
      guard plan.tensor(weight) is Some(tensor) else {
        abort("launch-v4 fixture lost weight")
      }
      match tensor.scale() {
        Some(scale) =>
          result.push(
            launch_v4_operand(
              WeightScaleInput(weight, scale),
              launch_v4_tensor_bytes(plan, scale.tensor()),
              16L,
            ),
          )
        None => ()
      }
      result.push(
        launch_v4_operand(
          ActivationOutput(try! @plan.ActivationRef::new(1)),
          8L,
          16L,
        ),
      )
      result
    }
    2 =>
      [
        launch_v4_operand(RuntimeInput(StepCounts), 20L, 4L),
        launch_v4_operand(RuntimeInput(QueryPositions), 16L, 4L),
        launch_v4_operand(
          ActivationInput(try! @plan.ActivationRef::new(1)),
          24L,
          16L,
        ),
        launch_v4_operand(
          ActivationOutput(try! @plan.ActivationRef::new(2)),
          24L,
          16L,
        ),
      ]
    3 => {
      let layout = launch_v4_layout(plan)
      [
        launch_v4_operand(RuntimeInput(StepCounts), 20L, 4L),
        launch_v4_operand(RuntimeInput(QueryPositions), 16L, 4L),
        launch_v4_operand(RuntimeInput(QueryRowOffsets), 8L, 4L),
        launch_v4_operand(RuntimeInput(SequenceLengths), 4L, 4L),
        launch_v4_operand(RuntimeInput(PageTableOffsets), 8L, 4L),
        launch_v4_operand(RuntimeInput(PageTablePageIndices), 8L, 4L),
        launch_v4_operand(
          ActivationInput(try! @plan.ActivationRef::new(2)),
          24L,
          16L,
        ),
        launch_v4_operand(
          ActivationOutput(try! @plan.ActivationRef::new(3)),
          8L,
          16L,
        ),
        launch_v4_operand(
          PersistentKvComponent(layer, Key),
          layout.component_stride_bytes(),
          layout.segment_alignment_bytes(),
        ),
        launch_v4_operand(
          PersistentKvComponent(layer, Value),
          layout.component_stride_bytes(),
          layout.segment_alignment_bytes(),
        ),
      ]
    }
    _ => abort("unknown launch-v4 operation")
  }
}

///|
fn launch_v4_inputs(
  plan : @plan.ModelPlan,
  override_operation? : Int = -1,
  override_operands? : Array[@launch_contract.AotOperand] = [],
  omit_operation? : Int = -1,
) -> Array[@launch_contract.PagedV4AotLaunchContractInput] {
  let profile_id = try! @launch_contract.KernelProfileId::from_stable_int(1)
  let result : Array[@launch_contract.PagedV4AotLaunchContractInput] = []
  for operation in 0..<4 {
    if operation != omit_operation {
      result.push(
        try! @launch_contract.PagedV4AotLaunchContractInput::new(
          profile_id~,
          operation_id=try! @plan.OperationId::new(operation),
          entry_point=launch_v4_entry_point(operation),
          dimensions=dimensions(4),
          operands=if operation == override_operation {
            override_operands
          } else {
            launch_v4_operands(plan, operation)
          },
        ),
      )
    }
  }
  result
}
