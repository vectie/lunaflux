///|
fn pointwise_manifest_plan() -> @plan.ModelPlan raise {
  let layer = @plan.KvLayerId::new(0)
  let operations = [
    @plan.PlanOperation::new_with_runtime_inputs(
      id=@plan.OperationId::new(0),
      kind=Embedding,
      inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
      outputs=[@plan.ActivationRef::new(0)],
      shape=EmbeddingShape(vocabulary_size=128, width=16),
      required_capability=@plan.KernelCapabilityId::embedding_lookup(),
      workspace_bytes=0L,
      runtime_inputs=[StepCounts],
    ),
    @plan.PlanOperation::new_with_execution(
      id=@plan.OperationId::new(1),
      kind=RotaryEmbedding,
      inputs=[Activation(@plan.ActivationRef::new(0))],
      outputs=[@plan.ActivationRef::new(1)],
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
    @plan.PlanOperation::new_with_execution(
      id=@plan.OperationId::new(2),
      kind=CausalAttention,
      inputs=[Activation(@plan.ActivationRef::new(1))],
      outputs=[@plan.ActivationRef::new(2)],
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
  @plan.ModelPlan::new_with_kv_execution(
    content_digest=@spec.ContentDigest::from_sha256(
      "5555555555555555555555555555555555555555555555555555555555555555",
    ),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows=2,
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
    final_output=@plan.ActivationRef::new(2),
    kv_execution=PagedKeyValue,
    numeric_schema=@plan_test_fixture.synthetic_bf16_numeric_schema(operations),
  )
}

///|
fn pointwise_manifest_layout() -> @device_layout.DeviceKvLayout raise {
  let plan = pointwise_manifest_plan()
  @device_layout.plan(
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
fn pointwise_manifest_semantics(
  kind : @plan.OperationKind,
) -> @catalog.KernelExecutionSemantics raise {
  let layer = @plan.KvLayerId::new(0)
  match kind {
    Embedding =>
      @catalog.KernelExecutionSemantics::paged_v3(
        kv_layer=None,
        runtime_inputs=[StepCounts],
        persistent_state=NoPersistentState,
        kv_layout=None,
      )
    RotaryEmbedding =>
      @catalog.KernelExecutionSemantics::paged_v3(
        kv_layer=Some(layer),
        runtime_inputs=[StepCounts, QueryPositions],
        persistent_state=NoPersistentState,
        kv_layout=None,
      )
    CausalAttention =>
      @catalog.KernelExecutionSemantics::paged_v3(
        kv_layer=Some(layer),
        runtime_inputs=[
          StepCounts,
          QueryPositions,
          QueryRowOffsets,
          SequenceLengths,
          PageTableOffsets,
          PageTablePageIndices,
        ],
        persistent_state=KvCacheReadWrite(layer),
        kv_layout=Some(
          @catalog.PagedKvLayoutContract::new(
            layout=LayerMajorSplitKeyValueV1,
            stable_version=1,
            tokens_per_page=4,
          ),
        ),
      )
    _ => abort("unsupported manifest fixture")
  }
}

///|
fn pointwise_manifest_entry(
  operation : Int,
) -> @catalog.AotKernelEntryPoint raise {
  @catalog.AotKernelEntryPoint::new(
    family=@catalog.AotKernelFamily::new(
      module_digest=@catalog.AotArtifactDigest::from_sha256(
        "6666666666666666666666666666666666666666666666666666666666666666",
      ),
      family_id=@catalog.AotKernelFamilyId::from_stable_int(operation + 1),
    ),
    entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(
      operation + 1,
    ),
  )
}

///|
fn pointwise_manifest_catalog() -> @catalog.ResolvedKernelCatalog raise {
  let shapes = [
    @plan.EmbeddingShape(vocabulary_size=128, width=16),
    RotaryShape(query_heads=4, kv_heads=2, head_dimension=4, rope_theta=10000.0),
    AttentionShape(query_heads=4, kv_heads=2, head_dimension=4),
  ]
  let kinds = [@plan.Embedding, RotaryEmbedding, CausalAttention]
  let capabilities = [
    @plan.KernelCapabilityId::embedding_lookup(),
    @plan.KernelCapabilityId::positioned_rotary_embedding(),
    @plan.KernelCapabilityId::paged_causal_attention(),
  ]
  let entries : Array[@catalog.CatalogEntry] = []
  for index, kind in kinds {
    entries.push(
      @catalog.CatalogEntry::new_paged_v3(
        operation_kind=kind,
        operation_shape=shapes[index],
        capability=capabilities[index],
        execution_semantics=pointwise_manifest_semantics(kind),
        compute_major=9,
        compute_minor=0,
        implementation=ContentAddressedAot(
          pointwise_manifest_entry(index).family(),
        ),
        workspace_bytes=0L,
        workspace_alignment=1L,
      ),
    )
  }
  @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v3(), entries~).resolve_paged(
    pointwise_manifest_plan(),
    pointwise_manifest_layout(),
    pointwise_test_target(),
  )
}

///|
fn pointwise_manifest_operands(
  operation : Int,
) -> Array[@launch_contract.AotOperand] raise {
  let layer = @plan.KvLayerId::new(0)
  match operation {
    0 =>
      [
        pointwise_operand(RuntimeInput(StepCounts), 20L, alignment=4L),
        pointwise_operand(ExternalInput(TokenIds), 16L, alignment=4L),
        pointwise_operand(WeightInput(@plan.TensorRef::new(0)), 4096L),
        pointwise_operand(ActivationOutput(@plan.ActivationRef::new(0)), 128L),
      ]
    1 =>
      [
        pointwise_operand(RuntimeInput(StepCounts), 20L, alignment=4L),
        pointwise_operand(RuntimeInput(QueryPositions), 16L, alignment=4L),
        pointwise_operand(ActivationInput(@plan.ActivationRef::new(0)), 256L),
        pointwise_operand(ActivationOutput(@plan.ActivationRef::new(1)), 256L),
      ]
    _ =>
      [
        pointwise_operand(RuntimeInput(StepCounts), 20L, alignment=4L),
        pointwise_operand(RuntimeInput(QueryPositions), 16L, alignment=4L),
        pointwise_operand(RuntimeInput(QueryRowOffsets), 12L, alignment=4L),
        pointwise_operand(RuntimeInput(SequenceLengths), 8L, alignment=4L),
        pointwise_operand(RuntimeInput(PageTableOffsets), 12L, alignment=4L),
        pointwise_operand(RuntimeInput(PageTablePageIndices), 16L, alignment=4L),
        pointwise_operand(ActivationInput(@plan.ActivationRef::new(1)), 256L),
        pointwise_operand(ActivationOutput(@plan.ActivationRef::new(2)), 128L),
        pointwise_operand(PersistentKvComponent(layer, Key), 2048L),
        pointwise_operand(PersistentKvComponent(layer, Value), 2048L),
      ]
  }
}

///|
fn pointwise_manifest_contracts(
  embedding_grid? : Int = 1,
) -> @launch_contract.PagedKvLaunchContractSet raise {
  let profile = pointwise_test_profile()
  let inputs : Array[@launch_contract.PagedKvAotLaunchContractInput] = []
  for operation in 0..<3 {
    inputs.push(
      @launch_contract.PagedKvAotLaunchContractInput::new(
        profile_id=profile.id(),
        operation_id=@plan.OperationId::new(operation),
        entry_point=pointwise_manifest_entry(operation),
        dimensions=@launch_contract.AotLaunchDimensions::new(
          grid_x=if operation == 0 { embedding_grid } else { 1 },
          grid_y=1,
          grid_z=1,
          block_x=256,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        operands=pointwise_manifest_operands(operation),
      ),
    )
  }
  @launch_contract.admit_paged_graph(
    pointwise_manifest_plan(),
    pointwise_manifest_catalog(),
    pointwise_manifest_layout(),
    [profile],
    inputs,
    @launch_contract.LaunchContractLimits::new(
      max_profiles=1,
      max_contracts=3,
      max_operands_per_contract=16,
    ),
  )
}
