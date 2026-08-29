///|
fn projection_fixture_operation(
  id : Int,
  kind : @plan.OperationKind,
  inputs : ArrayView[@plan.ValueRef],
  shape : @plan.OperationShape,
  capability : @plan.KernelCapabilityId,
) -> @plan.PlanOperation raise {
  @plan.PlanOperation::new_with_runtime_inputs(
    id=@plan.OperationId::new(id),
    kind~,
    inputs~,
    outputs=[@plan.ActivationRef::new(id)],
    shape~,
    required_capability=capability,
    workspace_bytes=0L,
    runtime_inputs=[StepCounts],
  )
}

///|
fn projection_fixture_plan() -> @plan.ModelPlan raise {
  let layer = @plan.KvLayerId::new(0)
  let operations = [
    projection_fixture_operation(
      0,
      Embedding,
      [ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
      EmbeddingShape(vocabulary_size=16, width=4),
      @plan.KernelCapabilityId::embedding_lookup(),
    ),
    projection_fixture_operation(
      1,
      QkvProjection,
      [
        Activation(@plan.ActivationRef::new(0)),
        Tensor(@plan.TensorRef::new(1)),
        Tensor(@plan.TensorRef::new(2)),
        Tensor(@plan.TensorRef::new(3)),
      ],
      ProjectionShape(input_width=4, output_width=8),
      @plan.KernelCapabilityId::qkv_projection(),
    ),
    @plan.PlanOperation::new_with_execution(
      id=@plan.OperationId::new(2),
      kind=RotaryEmbedding,
      inputs=[Activation(@plan.ActivationRef::new(1))],
      outputs=[@plan.ActivationRef::new(2)],
      shape=RotaryShape(
        query_heads=2,
        kv_heads=1,
        head_dimension=2,
        rope_theta=10000.0,
      ),
      required_capability=@plan.KernelCapabilityId::positioned_rotary_embedding(),
      workspace_bytes=0L,
      kv_layer=layer,
      runtime_inputs=[StepCounts, QueryPositions],
      persistent_state=NoPersistentState,
    ),
    @plan.PlanOperation::new_with_execution(
      id=@plan.OperationId::new(3),
      kind=CausalAttention,
      inputs=[Activation(@plan.ActivationRef::new(2))],
      outputs=[@plan.ActivationRef::new(3)],
      shape=AttentionShape(query_heads=2, kv_heads=1, head_dimension=2),
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
    projection_fixture_operation(
      4,
      OutputProjection,
      [Activation(@plan.ActivationRef::new(3)), Tensor(@plan.TensorRef::new(4))],
      ProjectionShape(input_width=4, output_width=4),
      @plan.KernelCapabilityId::dense_projection(),
    ),
    projection_fixture_operation(
      5,
      GatedMlp,
      [
        Activation(@plan.ActivationRef::new(4)),
        Tensor(@plan.TensorRef::new(5)),
        Tensor(@plan.TensorRef::new(6)),
        Tensor(@plan.TensorRef::new(7)),
      ],
      GatedMlpShape(hidden_size=4, intermediate_size=8),
      @plan.KernelCapabilityId::gated_mlp(),
    ),
    projection_fixture_operation(
      6,
      LanguageModelHead,
      [Activation(@plan.ActivationRef::new(5)), Tensor(@plan.TensorRef::new(8))],
      ProjectionShape(input_width=4, output_width=16),
      @plan.KernelCapabilityId::dense_projection(),
    ),
  ]
  @plan.ModelPlan::new_with_kv_execution(
    content_digest=@spec.ContentDigest::from_sha256("1".repeat(64)),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows=2,
      max_sequence_tokens=4,
      vocabulary_size=16,
      hidden_size=4,
      intermediate_size=8,
      attention_head_count=2,
      kv_head_count=1,
      head_dimension=2,
    ),
    kv_cache_geometry=@plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=1,
      head_dimension=2,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations~,
    final_output=@plan.ActivationRef::new(6),
    kv_execution=PagedKeyValue,
    numeric_schema=@plan_test_fixture.synthetic_bf16_numeric_schema(operations),
  )
}

///|
fn projection_fixture_target(
  bf16? : Bool = true,
) -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16=bf16,
    supports_cublas_lt=true,
  )
}

///|
fn projection_fixture_layout() -> @device_layout.DeviceKvLayout raise {
  let plan = projection_fixture_plan()
  @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=2,
      total_page_count=8,
      max_arena_bytes=4096L,
    ),
  )
}

///|
fn projection_fixture_entry(id : Int) -> @catalog.AotKernelEntryPoint raise {
  @catalog.AotKernelEntryPoint::new(
    family=@catalog.AotKernelFamily::new(
      module_digest=@catalog.AotArtifactDigest::from_sha256(
        (id + 1).to_string().repeat(64),
      ),
      family_id=@catalog.AotKernelFamilyId::from_stable_int(id + 1),
    ),
    entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(id + 1),
  )
}

///|
fn projection_fixture_semantics(
  operation : @plan.PlanOperation,
) -> @catalog.KernelExecutionSemantics raise {
  @catalog.KernelExecutionSemantics::paged_v3(
    kv_layer=operation.kv_layer(),
    runtime_inputs=operation.runtime_inputs(),
    persistent_state=operation.persistent_state(),
    kv_layout=if operation.persistent_state() is KvCacheReadWrite(_) {
      Some(
        @catalog.PagedKvLayoutContract::new(
          layout=LayerMajorSplitKeyValueV1,
          stable_version=1,
          tokens_per_page=2,
        ),
      )
    } else {
      None
    },
  )
}

///|
fn projection_fixture_catalog() -> @catalog.ResolvedKernelCatalog raise {
  let plan = projection_fixture_plan()
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in plan.operations() {
    entries.push(
      @catalog.CatalogEntry::new_paged_v3(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=projection_fixture_semantics(operation),
        compute_major=9,
        compute_minor=0,
        implementation=ContentAddressedAot(
          projection_fixture_entry(index).family(),
        ),
        workspace_bytes=0L,
        workspace_alignment=1L,
      ),
    )
  }
  @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v3(), entries~).resolve_paged(
    plan,
    projection_fixture_layout(),
    projection_fixture_target(),
  )
}

///|
fn projection_fixture_operand(
  role : @launch_contract.AotOperandRole,
  byte_count : Int64,
  alignment : Int64,
) -> @launch_contract.AotOperand raise {
  @launch_contract.AotOperand::new(role~, byte_count~, alignment~)
}
