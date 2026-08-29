///|
struct RuntimeRecipeV2Fixture {
  plan : @plan.ModelPlan
  legacy_startup : @fp8_startup_admission.Fp8StartupAdmission
  startup : @fp8_startup_admission.Fp8StartupAdmissionV2
  contracts : @launch_contract.PagedV4LaunchContractSet
}

///|
fn recipe_v2_entry(index : Int) -> @catalog.AotKernelEntryPoint {
  @catalog.AotKernelEntryPoint::new(
    family=@catalog.AotKernelFamily::new(
      module_digest=recipe_module_digest(),
      family_id=try! @catalog.AotKernelFamilyId::from_stable_int(index + 1),
    ),
    entry_point_id=try! @catalog.AotKernelEntryPointId::from_stable_int(
      index + 1,
    ),
  )
}

///|
fn recipe_v2_operations() -> Array[@plan.PlanOperation] {
  let layer = try! @plan.KvLayerId::new(0)
  [
    try! @plan.PlanOperation::new_with_runtime_inputs(
      id=@plan.OperationId::new(0),
      kind=Embedding,
      inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
      outputs=[@plan.ActivationRef::new(0)],
      shape=EmbeddingShape(vocabulary_size=64, width=16),
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
      workspace_bytes=4L,
      runtime_inputs=[StepCounts],
    ),
    try! @plan.PlanOperation::new_with_runtime_inputs(
      id=@plan.OperationId::new(2),
      kind=GatedMlp,
      inputs=[
        Activation(@plan.ActivationRef::new(1)),
        Tensor(@plan.TensorRef::new(2)),
        Tensor(@plan.TensorRef::new(3)),
        Tensor(@plan.TensorRef::new(4)),
      ],
      outputs=[@plan.ActivationRef::new(2)],
      shape=GatedMlpShape(hidden_size=16, intermediate_size=32),
      required_capability=@plan.KernelCapabilityId::gated_mlp(),
      workspace_bytes=8L,
      runtime_inputs=[StepCounts],
    ),
    try! @plan.PlanOperation::new_with_runtime_inputs(
      id=@plan.OperationId::new(3),
      kind=QkvProjection,
      inputs=[
        Activation(@plan.ActivationRef::new(2)),
        Tensor(@plan.TensorRef::new(5)),
        Tensor(@plan.TensorRef::new(6)),
        Tensor(@plan.TensorRef::new(7)),
      ],
      outputs=[@plan.ActivationRef::new(3)],
      shape=ProjectionShape(input_width=16, output_width=32),
      required_capability=@plan.KernelCapabilityId::qkv_projection(),
      workspace_bytes=4L,
      runtime_inputs=[StepCounts],
    ),
    try! @plan.PlanOperation::new_with_execution(
      id=@plan.OperationId::new(4),
      kind=RotaryEmbedding,
      inputs=[Activation(@plan.ActivationRef::new(3))],
      outputs=[@plan.ActivationRef::new(4)],
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
      id=@plan.OperationId::new(5),
      kind=CausalAttention,
      inputs=[Activation(@plan.ActivationRef::new(4))],
      outputs=[@plan.ActivationRef::new(5)],
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
fn recipe_v2_compound_execution() -> @numeric_contract.OperationExecutionContract {
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
fn recipe_v2_schema(
  operations : ArrayView[@plan.PlanOperation],
) -> @numeric_contract.ModelNumericSchema {
  let source = @plan_test_fixture.synthetic_fp8_w8a8_numeric_schema(
    operations,
    [
      try! @plan.OperationId::new(1),
      try! @plan.OperationId::new(2),
      try! @plan.OperationId::new(3),
    ],
  )
  let tensors : Array[@numeric_contract.NumericTensorSchema] = []
  for tensor in source.tensors() {
    tensors.push(tensor)
  }
  let numeric_operations : Array[@numeric_contract.NumericOperationSchema] = []
  for index, operation in source.operations() {
    numeric_operations.push(
      if index == 2 {
        @numeric_contract.NumericOperationSchema::new(
          ordinal=try! @numeric_contract.NumericOperationOrdinal::new(index),
          execution=recipe_v2_compound_execution(),
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
fn recipe_v2_plan(
  content? : String = "1111111111111111111111111111111111111111111111111111111111111111",
) -> @plan.ModelPlan raise {
  let operations = recipe_v2_operations()
  @plan.ModelPlan::new_with_kv_execution(
    content_digest=@spec.ContentDigest::from_sha256(content),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows=1,
      max_sequence_tokens=8,
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
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=8L),
    operations~,
    final_output=@plan.ActivationRef::new(5),
    kv_execution=PagedKeyValue,
    numeric_schema=recipe_v2_schema(operations),
  )
}

///|
fn recipe_v2_layout(plan : @plan.ModelPlan) -> @device_layout.DeviceKvLayout {
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
fn recipe_v2_semantics(
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
fn recipe_v2_resolved(
  plan : @plan.ModelPlan,
) -> @catalog.ResolvedKernelCatalog raise @catalog.CatalogError {
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in plan.operations() {
    entries.push(
      @catalog.CatalogEntry::new_paged_v4(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=recipe_v2_semantics(operation),
        operation_execution_digest=plan.numeric_binding().operations()[index]
          .execution()
          .digest(),
        aot_entry_point=recipe_v2_entry(index),
        compute_major=8,
        compute_minor=9,
        implementation=ContentAddressedAot(recipe_v2_entry(index).family()),
        workspace_bytes=operation.workspace_bytes(),
        workspace_alignment=if operation.workspace_bytes() > 0L {
          4L
        } else {
          1L
        },
      ),
    )
  }
  @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v4(), entries~).resolve_paged_v4(
    plan,
    recipe_v2_layout(plan),
    recipe_target(),
  )
}

///|
fn recipe_v2_tensor_bytes(
  plan : @plan.ModelPlan,
  reference : @plan.TensorRef,
) -> Int64 {
  let tensor = plan.tensor(reference).unwrap()
  try! @numeric_contract.storage_byte_length(
    tensor.storage().dtype(),
    tensor.shape(),
  )
}

///|
fn recipe_v2_operand(
  role : @launch_contract.AotOperandRole,
  bytes : Int64,
  alignment : Int64,
) -> @launch_contract.AotOperand {
  try! @launch_contract.AotOperand::new(role~, byte_count=bytes, alignment~)
}

///|
fn recipe_v2_operands(
  plan : @plan.ModelPlan,
  operation : Int,
) -> Array[@launch_contract.AotOperand] {
  let result : Array[@launch_contract.AotOperand] = [
    recipe_v2_operand(RuntimeInput(StepCounts), 20L, 4L),
  ]
  match operation {
    0 => {
      result.push(recipe_v2_operand(ExternalInput(TokenIds), 8L, 4L))
      result.push(
        recipe_v2_operand(
          WeightInput(try! @plan.TensorRef::new(0)),
          recipe_v2_tensor_bytes(plan, try! @plan.TensorRef::new(0)),
          16L,
        ),
      )
      result.push(
        recipe_v2_operand(
          ActivationOutput(try! @plan.ActivationRef::new(0)),
          64L,
          16L,
        ),
      )
    }
    1 | 2 | 3 => {
      result.push(
        recipe_v2_operand(
          ActivationInput(try! @plan.ActivationRef::new(operation - 1)),
          64L,
          16L,
        ),
      )
      let first = match operation {
        1 => 1
        2 => 2
        _ => 5
      }
      let count = if operation is (2 | 3) { 3 } else { 1 }
      for ordinal in 0..<count {
        let reference = try! @plan.TensorRef::new(first + ordinal)
        let tensor = plan.tensor(reference).unwrap()
        let scale = tensor.scale().unwrap()
        result.push(
          recipe_v2_operand(
            WeightInput(reference),
            recipe_v2_tensor_bytes(plan, reference),
            16L,
          ),
        )
        result.push(
          recipe_v2_operand(
            WeightScaleInput(reference, scale),
            recipe_v2_tensor_bytes(plan, scale.tensor()),
            16L,
          ),
        )
      }
      result.push(
        recipe_v2_operand(
          ActivationOutput(try! @plan.ActivationRef::new(operation)),
          if operation == 3 {
            128L
          } else {
            64L
          },
          16L,
        ),
      )
      result.push(
        recipe_v2_operand(
          Workspace,
          plan.operations()[operation].workspace_bytes(),
          4L,
        ),
      )
    }
    4 => {
      result.push(recipe_v2_operand(RuntimeInput(QueryPositions), 8L, 4L))
      result.push(
        recipe_v2_operand(
          ActivationInput(try! @plan.ActivationRef::new(3)),
          128L,
          16L,
        ),
      )
      result.push(
        recipe_v2_operand(
          ActivationOutput(try! @plan.ActivationRef::new(4)),
          128L,
          16L,
        ),
      )
    }
    5 => {
      let layer = try! @plan.KvLayerId::new(0)
      let layout = recipe_v2_layout(plan)
      result.push(recipe_v2_operand(RuntimeInput(QueryPositions), 8L, 4L))
      result.push(recipe_v2_operand(RuntimeInput(QueryRowOffsets), 8L, 4L))
      result.push(recipe_v2_operand(RuntimeInput(SequenceLengths), 4L, 4L))
      result.push(recipe_v2_operand(RuntimeInput(PageTableOffsets), 8L, 4L))
      result.push(recipe_v2_operand(RuntimeInput(PageTablePageIndices), 4L, 4L))
      result.push(
        recipe_v2_operand(
          ActivationInput(try! @plan.ActivationRef::new(4)),
          128L,
          16L,
        ),
      )
      result.push(
        recipe_v2_operand(
          ActivationOutput(try! @plan.ActivationRef::new(5)),
          64L,
          16L,
        ),
      )
      result.push(
        recipe_v2_operand(
          PersistentKvComponent(layer, Key),
          layout.component_stride_bytes(),
          layout.segment_alignment_bytes(),
        ),
      )
      result.push(
        recipe_v2_operand(
          PersistentKvComponent(layer, Value),
          layout.component_stride_bytes(),
          layout.segment_alignment_bytes(),
        ),
      )
    }
    _ => abort("unknown v2 recipe operation")
  }
  result
}

///|
fn recipe_v2_contracts(
  plan : @plan.ModelPlan,
  resolved : @catalog.ResolvedKernelCatalog,
) -> @launch_contract.PagedV4LaunchContractSet {
  let profile = try! @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=1,
    max_query_tokens=2,
    max_page_table_entries=1,
  )
  let inputs : Array[@launch_contract.PagedV4AotLaunchContractInput] = []
  for operation in 0..<6 {
    inputs.push(
      try! @launch_contract.PagedV4AotLaunchContractInput::new(
        profile_id=profile.id(),
        operation_id=@plan.OperationId::new(operation),
        entry_point=recipe_v2_entry(operation),
        dimensions=@launch_contract.AotLaunchDimensions::new(
          grid_x=2,
          grid_y=1,
          grid_z=1,
          block_x=32,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        operands=recipe_v2_operands(plan, operation),
      ),
    )
  }
  try! @launch_contract.admit_paged_graph_v4(
    plan,
    resolved,
    recipe_v2_layout(plan),
    [profile],
    inputs,
    @launch_contract.LaunchContractLimits::new(
      max_profiles=1,
      max_contracts=6,
      max_operands_per_contract=12,
    ),
  )
}
