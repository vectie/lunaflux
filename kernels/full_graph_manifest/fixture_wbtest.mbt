///|
fn full_manifest_content_digest() -> @spec.ContentDigest raise {
  @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
}

///|
fn full_manifest_plan() -> @plan.ModelPlan raise {
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
    content_digest=full_manifest_content_digest(),
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
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=64L),
    operations~,
    final_output=@plan.ActivationRef::new(2),
    kv_execution=PagedKeyValue,
    numeric_schema=@plan_test_fixture.synthetic_bf16_numeric_schema(operations),
  )
}

///|
fn full_manifest_target() -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
}

///|
fn full_manifest_digest(operation : Int) -> @catalog.AotArtifactDigest raise {
  @catalog.AotArtifactDigest::from_sha256(
    if operation == 1 {
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    } else {
      "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
    },
  )
}

///|
fn full_manifest_entry(operation : Int) -> @catalog.AotKernelEntryPoint raise {
  @catalog.AotKernelEntryPoint::new(
    family=@catalog.AotKernelFamily::new(
      module_digest=full_manifest_digest(operation),
      family_id=@catalog.AotKernelFamilyId::from_stable_int(operation + 11),
    ),
    entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(
      operation + 1,
    ),
  )
}

///|
fn full_manifest_semantics(
  operation : Int,
) -> @catalog.KernelExecutionSemantics raise {
  let layer = @plan.KvLayerId::new(0)
  match operation {
    1 =>
      @catalog.KernelExecutionSemantics::paged_v3(
        kv_layer=Some(layer),
        runtime_inputs=[StepCounts, QueryPositions],
        persistent_state=NoPersistentState,
        kv_layout=None,
      )
    2 =>
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
    _ =>
      @catalog.KernelExecutionSemantics::paged_v3(
        kv_layer=None,
        runtime_inputs=[StepCounts],
        persistent_state=NoPersistentState,
        kv_layout=None,
      )
  }
}

///|
fn full_manifest_layout() -> @device_layout.DeviceKvLayout raise {
  let plan = full_manifest_plan()
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
fn full_manifest_resolved() -> @catalog.ResolvedKernelCatalog raise {
  let plan = full_manifest_plan()
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in plan.operations() {
    entries.push(
      @catalog.CatalogEntry::new_paged_v3(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=full_manifest_semantics(index),
        compute_major=8,
        compute_minor=0,
        implementation=ContentAddressedAot(full_manifest_entry(index).family()),
        workspace_bytes=0L,
        workspace_alignment=1L,
      ),
    )
  }
  @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v3(), entries~).resolve_paged(
    plan,
    full_manifest_layout(),
    full_manifest_target(),
  )
}

///|
fn full_manifest_operand(
  role : @launch_contract.AotOperandRole,
  bytes : Int64,
  alignment : Int64,
) -> @launch_contract.AotOperand raise {
  @launch_contract.AotOperand::new(role~, byte_count=bytes, alignment~)
}

///|
fn full_manifest_operands(
  operation : Int,
) -> Array[@launch_contract.AotOperand] raise {
  let layer = @plan.KvLayerId::new(0)
  match operation {
    0 =>
      [
        full_manifest_operand(RuntimeInput(StepCounts), 20L, 4L),
        full_manifest_operand(ExternalInput(TokenIds), 16L, 4L),
        full_manifest_operand(WeightInput(@plan.TensorRef::new(0)), 4096L, 16L),
        full_manifest_operand(
          ActivationOutput(@plan.ActivationRef::new(0)),
          128L,
          16L,
        ),
      ]
    1 =>
      [
        full_manifest_operand(RuntimeInput(StepCounts), 20L, 4L),
        full_manifest_operand(RuntimeInput(QueryPositions), 16L, 4L),
        full_manifest_operand(
          ActivationInput(@plan.ActivationRef::new(0)),
          128L,
          16L,
        ),
        full_manifest_operand(
          ActivationOutput(@plan.ActivationRef::new(1)),
          128L,
          16L,
        ),
      ]
    _ =>
      [
        full_manifest_operand(RuntimeInput(StepCounts), 20L, 4L),
        full_manifest_operand(RuntimeInput(QueryPositions), 16L, 4L),
        full_manifest_operand(RuntimeInput(QueryRowOffsets), 12L, 4L),
        full_manifest_operand(RuntimeInput(SequenceLengths), 8L, 4L),
        full_manifest_operand(RuntimeInput(PageTableOffsets), 12L, 4L),
        full_manifest_operand(RuntimeInput(PageTablePageIndices), 16L, 4L),
        full_manifest_operand(
          ActivationInput(@plan.ActivationRef::new(1)),
          128L,
          16L,
        ),
        full_manifest_operand(
          ActivationOutput(@plan.ActivationRef::new(2)),
          128L,
          16L,
        ),
        full_manifest_operand(PersistentKvComponent(layer, Key), 2048L, 256L),
        full_manifest_operand(PersistentKvComponent(layer, Value), 2048L, 256L),
      ]
  }
}
