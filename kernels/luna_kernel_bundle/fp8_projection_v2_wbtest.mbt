///|
fn fp8_v2_bundle_operations() -> Array[@plan.PlanOperation] {
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
      shape=GatedMlpShape(hidden_size=1, intermediate_size=2),
      required_capability=@plan.KernelCapabilityId::gated_mlp(),
      workspace_bytes=8L,
      runtime_inputs=[StepCounts],
    ),
    try! @plan.PlanOperation::new_with_execution(
      id=@plan.OperationId::new(3),
      kind=RotaryEmbedding,
      inputs=[Activation(@plan.ActivationRef::new(2))],
      outputs=[@plan.ActivationRef::new(3)],
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
      id=@plan.OperationId::new(4),
      kind=CausalAttention,
      inputs=[Activation(@plan.ActivationRef::new(3))],
      outputs=[@plan.ActivationRef::new(4)],
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
fn fp8_v2_bundle_schema(
  operations : ArrayView[@plan.PlanOperation],
) -> @numeric_contract.ModelNumericSchema {
  let source = @plan_test_fixture.synthetic_fp8_w8a8_numeric_schema(
    operations,
    [try! @plan.OperationId::new(1), try! @plan.OperationId::new(2)],
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
          execution=try! @numeric_contract.OperationExecutionContract::new(
            activation_input=@numeric_contract.ComputeDType::bfloat16(),
            activation_compute=@numeric_contract.ComputeDType::f8_e4m3_finite(),
            activation_scale=@numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_bound_expf_f32_v2(),
            tensor_input=@numeric_contract.ComputeDType::f8_e4m3_finite(),
            accumulator=@numeric_contract.AccumulatorDType::f32(),
            output=@numeric_contract.ComputeDType::bfloat16(),
            conversion=@numeric_contract.NumericConversion::quantized_v1(),
            order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
          ),
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
fn fp8_v2_bundle_plan(content? : String = "6".repeat(64)) -> @plan.ModelPlan {
  let operations = fp8_v2_bundle_operations()
  try! @plan.ModelPlan::new_with_kv_execution(
    content_digest=@spec.ContentDigest::from_sha256(content),
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
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=8L),
    operations~,
    final_output=@plan.ActivationRef::new(4),
    kv_execution=PagedKeyValue,
    numeric_schema=fp8_v2_bundle_schema(operations),
  )
}

///|
fn fp8_v2_bundle_layout(
  plan : @plan.ModelPlan,
) -> @device_layout.DeviceKvLayout {
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
fn fp8_v2_bundle_profile() -> @launch_contract.PagedKernelProfile {
  try! @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=1,
    max_query_tokens=4,
    max_page_table_entries=2,
  )
}

///|
fn fp8_v2_bundle_bytes() -> Bytes {
  b"real-fp8-v2-gated-offline-cubin"
}

///|
fn fp8_v2_bundle_family(index : Int) -> @catalog.AotKernelFamily {
  @catalog.AotKernelFamily::new(
    module_digest=try! @catalog.AotArtifactDigest::from_sha256(
      kernel_bundle_sha256(fp8_v2_bundle_bytes()),
    ),
    family_id=try! @catalog.AotKernelFamilyId::from_stable_int(index + 31),
  )
}

///|
fn fp8_v2_bundle_entry(index : Int) -> @catalog.AotKernelEntryPoint {
  @catalog.AotKernelEntryPoint::new(
    family=fp8_v2_bundle_family(index),
    entry_point_id=try! @catalog.AotKernelEntryPointId::from_stable_int(
      if index == 1 || index == 2 {
        2
      } else {
        index + 11
      },
    ),
  )
}

///|
fn fp8_v2_bundle_resolved(
  plan : @plan.ModelPlan,
) -> @catalog.ResolvedKernelCatalog {
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in plan.operations() {
    entries.push(
      try! @catalog.CatalogEntry::new_paged_v4(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=fp8_bundle_semantics(operation),
        operation_execution_digest=plan.numeric_binding().operations()[index]
          .execution()
          .digest(),
        aot_entry_point=fp8_v2_bundle_entry(index),
        compute_major=8,
        compute_minor=9,
        implementation=ContentAddressedAot(fp8_v2_bundle_family(index)),
        workspace_bytes=operation.workspace_bytes(),
        workspace_alignment=if operation.workspace_bytes() == 0L {
          1L
        } else {
          4L
        },
      ),
    )
  }
  try! @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v4(),
    entries~,
  ).resolve_paged_v4(plan, fp8_v2_bundle_layout(plan), fp8_bundle_target())
}

///|
fn fp8_v2_bundle_operands(
  plan : @plan.ModelPlan,
  index : Int,
) -> Array[@launch_contract.AotOperand] {
  if index == 1 || index == 2 {
    let admitted = try! @luna_cuda_fp8_projection_aot.derive_fp8_projection_operands_v2(
      plan,
      operation=try! @plan.OperationId::new(index),
      fp8_v2_bundle_profile(),
    )
    let result : Array[@launch_contract.AotOperand] = []
    for operand in admitted {
      result.push(operand)
    }
    return result
  }
  let layer = try! @plan.KvLayerId::new(0)
  match index {
    0 => fp8_bundle_operands(plan, 0)
    3 =>
      [
        fp8_bundle_operand(RuntimeInput(StepCounts), 20L, 4L),
        fp8_bundle_operand(RuntimeInput(QueryPositions), 16L, 4L),
        fp8_bundle_operand(
          ActivationInput(try! @plan.ActivationRef::new(2)),
          24L,
          16L,
        ),
        fp8_bundle_operand(
          ActivationOutput(try! @plan.ActivationRef::new(3)),
          24L,
          16L,
        ),
      ]
    _ => {
      let layout = fp8_v2_bundle_layout(plan)
      [
        fp8_bundle_operand(RuntimeInput(StepCounts), 20L, 4L),
        fp8_bundle_operand(RuntimeInput(QueryPositions), 16L, 4L),
        fp8_bundle_operand(RuntimeInput(QueryRowOffsets), 8L, 4L),
        fp8_bundle_operand(RuntimeInput(SequenceLengths), 4L, 4L),
        fp8_bundle_operand(RuntimeInput(PageTableOffsets), 8L, 4L),
        fp8_bundle_operand(RuntimeInput(PageTablePageIndices), 8L, 4L),
        fp8_bundle_operand(
          ActivationInput(try! @plan.ActivationRef::new(3)),
          24L,
          16L,
        ),
        fp8_bundle_operand(
          ActivationOutput(try! @plan.ActivationRef::new(4)),
          8L,
          16L,
        ),
        fp8_bundle_operand(
          PersistentKvComponent(layer, Key),
          layout.component_stride_bytes(),
          layout.segment_alignment_bytes(),
        ),
        fp8_bundle_operand(
          PersistentKvComponent(layer, Value),
          layout.component_stride_bytes(),
          layout.segment_alignment_bytes(),
        ),
      ]
    }
  }
}

///|
fn fp8_v2_bundle_contracts(
  plan : @plan.ModelPlan,
) -> @launch_contract.PagedV4LaunchContractSet raise {
  let inputs : Array[@launch_contract.PagedV4AotLaunchContractInput] = []
  for index in 0..<5 {
    inputs.push(
      try! @launch_contract.PagedV4AotLaunchContractInput::new(
        profile_id=fp8_v2_bundle_profile().id(),
        operation_id=@plan.OperationId::new(index),
        entry_point=fp8_v2_bundle_entry(index),
        dimensions=@launch_contract.AotLaunchDimensions::new(
          grid_x=1,
          grid_y=1,
          grid_z=1,
          block_x=256,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        operands=fp8_v2_bundle_operands(plan, index),
      ),
    )
  }
  @launch_contract.admit_paged_graph_v4(
    plan,
    fp8_v2_bundle_resolved(plan),
    fp8_v2_bundle_layout(plan),
    [fp8_v2_bundle_profile()],
    inputs,
    @launch_contract.LaunchContractLimits::new(
      max_profiles=1,
      max_contracts=5,
      max_operands_per_contract=20,
    ),
  )
}
